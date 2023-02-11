// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <memory>  // std::unique_ptr

#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/DevicePointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>

namespace noa::TNL {
namespace Matrices {

template< typename Vector, typename Matrix >
__global__
void
SparseMatrixSetRowLengthsVectorKernel( Vector* rowLengths,
                                       const Matrix* matrix,
                                       typename Matrix::IndexType rows,
                                       typename Matrix::IndexType cols )
{
#ifdef __CUDACC__
   using IndexType = typename Matrix::IndexType;

   IndexType rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   while( rowIdx < rows ) {
      const auto row = matrix->getRow( rowIdx );
      IndexType length = 0;
      for( IndexType c_j = 0; c_j < row.getSize(); c_j++ )
         if( row.getColumnIndex( c_j ) < cols )
            length++;
         else
            break;
      rowLengths[ rowIdx ] = length;
      rowIdx += gridSize;
   }
#endif
}

template< typename Matrix1, typename Matrix2 >
__global__
void
SparseMatrixCopyKernel( Matrix1* A,
                        const Matrix2* B,
                        const typename Matrix2::IndexType* rowLengths,
                        typename Matrix2::IndexType rows )
{
#ifdef __CUDACC__
   using IndexType = typename Matrix2::IndexType;

   IndexType rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   while( rowIdx < rows ) {
      const auto length = rowLengths[ rowIdx ];
      const auto rowB = B->getRow( rowIdx );
      auto rowA = A->getRow( rowIdx );
      for( IndexType c = 0; c < length; c++ )
         rowA.setElement( c, rowB.getColumnIndex( c ), rowB.getValue( c ) );
      rowIdx += gridSize;
   }
#endif
}

// copy on the same device
template< typename Matrix1, typename Matrix2 >
typename std::enable_if< std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value >::type
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   static_assert( std::is_same< typename Matrix1::RealType, typename Matrix2::RealType >::value,
                  "The matrices must have the same RealType." );
   static_assert( std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value,
                  "The matrices must be allocated on the same device." );
   static_assert( std::is_same< typename Matrix1::IndexType, typename Matrix2::IndexType >::value,
                  "The matrices must have the same IndexType." );

   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = B.getRows();
   const IndexType cols = B.getColumns();

   A.setDimensions( rows, cols );

   if constexpr( std::is_same< DeviceType, Devices::Host >::value ) {
      // set row lengths
      typename Matrix1::RowsCapacitiesType rowLengths;
      rowLengths.setSize( rows );
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < rows; i++ ) {
         const auto row = B.getRow( i );
         IndexType length = 0;
         for( IndexType c_j = 0; c_j < row.getSize(); c_j++ )
            if( row.getColumnIndex( c_j ) < cols )
               length++;
            else
               break;
         rowLengths[ i ] = length;
      }
      A.setRowCapacities( rowLengths );

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < rows; i++ ) {
         const auto length = rowLengths[ i ];
         const auto rowB = B.getRow( i );
         auto rowA = A.getRow( i );
         for( IndexType c = 0; c < length; c++ )
            rowA.setElement( c, rowB.getColumnIndex( c ), rowB.getValue( c ) );
      }
   }

   if constexpr( std::is_same< DeviceType, Devices::Cuda >::value ) {
      Cuda::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      const IndexType desGridSize = 32 * Cuda::DeviceInfo::getCudaMultiprocessors( Cuda::DeviceInfo::getActiveDevice() );
      launch_config.gridSize.x = min( desGridSize, Cuda::getNumberOfBlocks( rows, launch_config.blockSize.x ) );

      typename Matrix1::RowsCapacitiesType rowLengths;
      rowLengths.setSize( rows );

      Pointers::DevicePointer< Matrix1 > Apointer( A );
      const Pointers::DevicePointer< const Matrix2 > Bpointer( B );

      // set row lengths
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      constexpr auto kernelRowLenghts =
         SparseMatrixSetRowLengthsVectorKernel< typename Matrix1::RowsCapacitiesType::ValueType, Matrix2 >;
      Cuda::launchKernelSync( kernelRowLenghts,
                              launch_config,
                              rowLengths.getData(),
                              &Bpointer.template getData< TNL::Devices::Cuda >(),
                              rows,
                              cols );
      Apointer->setRowCapacities( rowLengths );

      // copy rows
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      constexpr auto kernelCopy = SparseMatrixCopyKernel< Matrix1, Matrix2 >;
      Cuda::launchKernelSync( kernelCopy,
                              launch_config,
                              &Apointer.template modifyData< TNL::Devices::Cuda >(),
                              &Bpointer.template getData< TNL::Devices::Cuda >(),
                              rowLengths.getData(),
                              rows );
   }
}

// cross-device copy (host -> gpu)
template< typename Matrix1, typename Matrix2 >
typename std::enable_if< ! std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value
                         && std::is_same< typename Matrix2::DeviceType, Devices::Host >::value >::type
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   using CudaMatrix2 = typename Matrix2::template Self< typename Matrix2::RealType, Devices::Cuda >;
   CudaMatrix2 B_tmp;
   B_tmp = B;
   copySparseMatrix_impl( A, B_tmp );
}

// cross-device copy (gpu -> host)
template< typename Matrix1, typename Matrix2 >
typename std::enable_if< ! std::is_same< typename Matrix1::DeviceType, typename Matrix2::DeviceType >::value
                         && std::is_same< typename Matrix2::DeviceType, Devices::Cuda >::value >::type
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   using CudaMatrix1 = typename Matrix1::template Self< typename Matrix1::RealType, Devices::Cuda >;
   CudaMatrix1 A_tmp;
   copySparseMatrix_impl( A_tmp, B );
   A = A_tmp;
}

template< typename Matrix1, typename Matrix2 >
void
copySparseMatrix( Matrix1& A, const Matrix2& B )
{
   copySparseMatrix_impl( A, B );
}

template< typename Matrix, typename AdjacencyMatrix >
void
copyAdjacencyStructure( const Matrix& A, AdjacencyMatrix& B, bool has_symmetric_pattern, bool ignore_diagonal )
{
   static_assert( std::is_same< typename Matrix::DeviceType, Devices::Host >::value,
                  "The function is not implemented for CUDA matrices - it would require atomic insertions "
                  "of elements into the sparse format." );
   static_assert( std::is_same< typename Matrix::DeviceType, typename AdjacencyMatrix::DeviceType >::value,
                  "The matrices must be allocated on the same device." );
   static_assert( std::is_same< typename Matrix::IndexType, typename AdjacencyMatrix::IndexType >::value,
                  "The matrices must have the same IndexType." );
   //   static_assert( std::is_same< typename AdjacencyMatrix::RealType, bool >::value,
   //                  "The RealType of the adjacency matrix must be bool." );

   using IndexType = typename Matrix::IndexType;

   if( A.getRows() != A.getColumns() ) {
      throw std::logic_error( "The matrix is not square: " + std::to_string( A.getRows() ) + " rows, "
                              + std::to_string( A.getColumns() ) + " columns." );
   }

   const IndexType N = A.getRows();
   B.setDimensions( N, N );

   // set row lengths
   typename AdjacencyMatrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( N );
   rowLengths.setValue( 0 );
   for( IndexType i = 0; i < A.getRows(); i++ ) {
      const auto row = A.getRow( i );
      IndexType length = 0;
      for( int c_j = 0; c_j < row.getSize(); c_j++ ) {
         const IndexType j = row.getColumnIndex( c_j );
         if( j >= A.getColumns() )
            break;
         length++;
         if( ! has_symmetric_pattern && i != j )
            if( A.getElement( j, i ) == 0 )
               rowLengths[ j ]++;
      }
      if( ignore_diagonal )
         length--;
      rowLengths[ i ] += length;
   }
   B.setRowCapacities( rowLengths );

   // set non-zeros
   for( IndexType i = 0; i < A.getRows(); i++ ) {
      const auto row = A.getRow( i );
      for( int c_j = 0; c_j < row.getSize(); c_j++ ) {
         const IndexType j = row.getColumnIndex( c_j );
         if( j >= A.getColumns() )
            break;
         if( ! ignore_diagonal || i != j )
            if( A.getElement( i, j ) != 0 ) {
               B.setElement( i, j, true );
               if( ! has_symmetric_pattern )
                  B.setElement( j, i, true );
            }
      }
   }
}

template< typename Matrix1, typename Matrix2, typename PermutationArray >
void
reorderSparseMatrix( const Matrix1& matrix1, Matrix2& matrix2, const PermutationArray& perm, const PermutationArray& iperm )
{
   // TODO: implement on GPU
   static_assert( std::is_same< typename Matrix1::DeviceType, Devices::Host >::value,
                  "matrix reordering is implemented only for host" );
   static_assert( std::is_same< typename Matrix2::DeviceType, Devices::Host >::value,
                  "matrix reordering is implemented only for host" );
   static_assert( std::is_same< typename PermutationArray::DeviceType, Devices::Host >::value,
                  "matrix reordering is implemented only for host" );

   using IndexType = typename Matrix1::IndexType;

   matrix2.setDimensions( matrix1.getRows(), matrix1.getColumns() );

   // set row lengths
   typename Matrix2::RowsCapacitiesType rowLengths;
   rowLengths.setSize( matrix1.getRows() );
   for( IndexType i = 0; i < matrix1.getRows(); i++ ) {
      const auto row = matrix1.getRow( perm[ i ] );
      IndexType length = 0;
      for( IndexType j = 0; j < row.getSize(); j++ )
         if( row.getColumnIndex( j ) < matrix1.getColumns() )
            length++;
      rowLengths[ i ] = length;
   }
   matrix2.setRowCapacities( rowLengths );

   // set row elements
   for( IndexType i = 0; i < matrix2.getRows(); i++ ) {
      const IndexType rowLength = rowLengths[ i ];

      // extract sparse row
      const auto row1 = matrix1.getRow( perm[ i ] );

      // permute
      std::unique_ptr< typename Matrix2::IndexType[] > columns{ new typename Matrix2::IndexType[ rowLength ] };
      std::unique_ptr< typename Matrix2::RealType[] > values{ new typename Matrix2::RealType[ rowLength ] };
      for( IndexType j = 0; j < rowLength; j++ ) {
         columns[ j ] = iperm[ row1.getColumnIndex( j ) ];
         values[ j ] = row1.getValue( j );
      }

      // sort
      std::unique_ptr< IndexType[] > indices{ new IndexType[ rowLength ] };
      for( IndexType j = 0; j < rowLength; j++ )
         indices[ j ] = j;
      auto comparator = [ &columns ]( IndexType a, IndexType b )
      {
         return columns[ a ] < columns[ b ];
      };
      std::sort( indices.get(), indices.get() + rowLength, comparator );

      // set the row
      auto row2 = matrix2.getRow( i );
      for( IndexType j = 0; j < rowLength; j++ )
         row2.setElement( j, columns[ indices[ j ] ], values[ indices[ j ] ] );
   }
}

template< typename Array1, typename Array2, typename PermutationArray >
void
reorderArray( const Array1& src, Array2& dest, const PermutationArray& perm )
{
   static_assert( std::is_same< typename Array1::DeviceType, typename Array2::DeviceType >::value,
                  "Arrays must reside on the same device." );
   static_assert( std::is_same< typename Array1::DeviceType, typename PermutationArray::DeviceType >::value,
                  "Arrays must reside on the same device." );
   TNL_ASSERT_EQ( src.getSize(), perm.getSize(), "Source array and permutation must have the same size." );
   TNL_ASSERT_EQ( dest.getSize(), perm.getSize(), "Destination array and permutation must have the same size." );

   using DeviceType = typename Array1::DeviceType;
   using IndexType = typename Array1::IndexType;

   auto kernel = [] __cuda_callable__( IndexType i,
                                       const typename Array1::ValueType* src,
                                       typename Array2::ValueType* dest,
                                       const typename PermutationArray::ValueType* perm )
   {
      dest[ i ] = src[ perm[ i ] ];
   };

   Algorithms::ParallelFor< DeviceType >::exec(
      (IndexType) 0, src.getSize(), kernel, src.getData(), dest.getData(), perm.getData() );
}

}  // namespace Matrices
}  // namespace noa::TNL
