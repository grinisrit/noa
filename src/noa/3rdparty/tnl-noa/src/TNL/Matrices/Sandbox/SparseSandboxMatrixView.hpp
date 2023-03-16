// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <functional>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Sandbox/SparseSandboxMatrixView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/AtomicOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/details/SparseMatrix.h>

namespace noa::TNL {
namespace Matrices {
namespace Sandbox {

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::SparseSandboxMatrixView() {}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::SparseSandboxMatrixView(
   const IndexType rows,
   const IndexType columns,
   const ValuesViewType& values,
   const ColumnsIndexesViewType& columnIndexes,
   const RowPointersView& rowPointers )
: MatrixView< Real, Device, Index >( rows, columns, values ), columnIndexes( columnIndexes ), rowPointers( rowPointers )
{}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
auto
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getView() -> ViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getView(),
            this->getColumnIndexes().getView(),
            this->getSegments().getView() };
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
auto
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getConstView() const -> ConstViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getConstView(),
            this->getColumnsIndexes().getConstView(),
            this->getSegments().getConstView() };
}

template< typename Real, typename Device, typename Index, typename MatrixType >
std::string
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getSerializationType()
{
   return "Matrices::Sandbox::SparseMatrix< " + TNL::getSerializationType< RealType >() + ", "
        + TNL::getSerializationType< IndexType >() + ", " + MatrixType::getSerializationType()
        + ", [any_allocator], [any_allocator] >";
}

template< typename Real, typename Device, typename Index, typename MatrixType >
std::string
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Vector >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getCompressedRowLengths( Vector& rowLengths ) const
{
   details::set_size_if_resizable( rowLengths, this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return ( value != 0.0 );
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Vector >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getRowCapacities( Vector& rowLengths ) const
{
   details::set_size_if_resizable( rowLengths, this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return 1;
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
Index
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getRowCapacity( const IndexType row ) const
{
   return this->segments.getSegmentSize( row );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
Index
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getNonzeroElementsCount() const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
   if( ! isSymmetric() ) {
      auto fetch = [ = ] __cuda_callable__( const IndexType i ) -> IndexType
      {
         return ( columns_view[ i ] != paddingIndex );
      };
      return Algorithms::reduce< DeviceType >( (IndexType) 0, this->columnIndexes.getSize(), fetch, std::plus<>{}, 0 );
   }
   else {
      const auto rows = this->getRows();
      const auto columns = this->getColumns();
      Containers::Vector< IndexType, DeviceType, IndexType > row_sums( this->getRows(), 0 );
      auto row_sums_view = row_sums.getView();
      auto row_pointers_view = this->rowPointers.getConstView();
      const auto columnIndexesView = this->columnIndexes.getConstView();
      // SANDBOX_TODO: Replace the following lambda function (or more) with code compute number of nonzero matrix elements
      //               of symmetric matrix. Note, that this is required only by symmetric matrices and that the highest
      //               performance is not a priority here.
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         auto begin = row_pointers_view[ rowIdx ];
         auto end = row_pointers_view[ rowIdx + 1 ];
         IndexType sum( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
            const IndexType column = columnIndexesView[ globalIdx ];
            if( column != paddingIndex )
               sum += 1 + ( column != rowIdx && column < rows && rowIdx < columns );
         }
         row_sums_view[ rowIdx ] = sum;
      };
      TNL::Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, this->getRows(), f );
      return sum( row_sums );
   }
   return 0;
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
auto
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getRow( const IndexType& rowIdx ) const -> ConstRowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   // SANDBOX_TODO: Replace the following with creation of RowView corresponding with your sparse matrix format.
   return ConstRowView( rowIdx,                       // row index
                        this->rowPointers[ rowIdx ],  // row begining
                        this->rowPointers[ rowIdx + 1 ]
                           - this->rowPointers[ rowIdx ],  // number of elemnts allocated for given row
                        this->values,
                        this->columnIndexes );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
auto
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getRow( const IndexType& rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   // SANDBOX_TODO: Replace this with RowView constructor by your needs.
   return RowView( rowIdx,                                             // row index
                   rowPointers[ rowIdx ],                              // index of the first nonzero element in the row
                   rowPointers[ rowIdx + 1 ] - rowPointers[ rowIdx ],  // number of nonzero elements in the row
                   this->values,
                   this->columnIndexes );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::setElement( const IndexType row,
                                                                        const IndexType column,
                                                                        const RealType& value )
{
   this->addElement( row, column, value, 0.0 );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::addElement( IndexType row,
                                                                        IndexType column,
                                                                        const RealType& value,
                                                                        const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "Sparse matrix row index cannot be negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Sparse matrix row index is larger than number of matrix rows." );
   TNL_ASSERT_GE( column, 0, "Sparse matrix column index cannot be negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Sparse matrix column index is larger than number of matrix columns." );

   if( isSymmetric() && row < column ) {
      swap( row, column );
      TNL_ASSERT_LT( row, this->getRows(), "Column index is out of the symmetric part of the matrix after transposition." );
      TNL_ASSERT_LT( column, this->getColumns(), "Row index is out of the symmetric part of the matrix after transposition." );
   }

   // SANDBOX_TODO: Replace the following line with a code that computes number of matrix elements allocated for
   //               matrix row with indedx `row`. Note that the code works on both host and GPU kernel. To achieve
   //               the same effect, you may use macro __CUDA_ARCH__ as can be seen bellow in this method.
   const IndexType rowSize = this->rowPointers.getElement( row + 1 ) - this->rowPointers.getElement( row );
   IndexType col( this->getPaddingIndex() );
   IndexType i;
   IndexType globalIdx = 0;
   for( i = 0; i < rowSize; i++ ) {
      // SANDBOX_TODO: Replace the following line with a code that computes a global index of `i`-th nonzero matrix element
      //               in the `row`-th matrix row. The global index is a pointer to arrays `values` and `columnIndexes` storing
      //               the matrix elements values and column indexes respectively.
      globalIdx = this->rowPointers.getElement( row ) + i;
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      col = this->columnIndexes.getElement( globalIdx );
      if( col == column ) {
         if( ! isBinary() )
            this->values.setElement( globalIdx, thisElementMultiplicator * this->values.getElement( globalIdx ) + value );
         return;
      }
      if( col == this->getPaddingIndex() || col > column )
         break;
   }
   if( i == rowSize ) {
#ifndef __CUDA_ARCH__
      std::stringstream msg;
      msg << "The capacity of the sparse matrix row number " << row << " was exceeded.";
      throw std::logic_error( msg.str() );
#else
      TNL_ASSERT_TRUE( false, "" );
      return;
#endif
   }
   if( col == this->getPaddingIndex() ) {
      this->columnIndexes.setElement( globalIdx, column );
      if( ! isBinary() )
         this->values.setElement( globalIdx, value );
      return;
   }
   else {
      IndexType j = rowSize - 1;
      while( j > i ) {
         // SANDBOX_TODO: Replace the following two lines with a code that computes a global indexes of `j`-th and `j-1`-th
         // nonzero matrix elements
         //               in the `row`-th matrix row. The global index is a pointer to arrays `values` and `columnIndexes`
         //               storing the matrix elements values and column indexes respectively.
         const IndexType globalIdx1 = this->rowPointers.getElement( row ) + j;
         const IndexType globalIdx2 = globalIdx1 - 1;
         // End of code replacement.
         TNL_ASSERT_LT( globalIdx1, this->columnIndexes.getSize(), "" );
         TNL_ASSERT_LT( globalIdx2, this->columnIndexes.getSize(), "" );
         this->columnIndexes.setElement( globalIdx1, this->columnIndexes.getElement( globalIdx2 ) );
         if( ! isBinary() )
            this->values.setElement( globalIdx1, this->values.getElement( globalIdx2 ) );
         j--;
      }

      this->columnIndexes.setElement( globalIdx, column );
      if( ! isBinary() )
         this->values.setElement( globalIdx, value );
      return;
   }
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
auto
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getElement( IndexType row, IndexType column ) const -> RealType
{
   TNL_ASSERT_GE( row, 0, "Sparse matrix row index cannot be negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Sparse matrix row index is larger than number of matrix rows." );
   TNL_ASSERT_GE( column, 0, "Sparse matrix column index cannot be negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Sparse matrix column index is larger than number of matrix columns." );

   if( isSymmetric() && row < column ) {
      swap( row, column );
      if( row >= this->getRows() || column >= this->getColumns() )
         return 0.0;
   }

   // SANDBOX_TODO: Replace the following lines with a code for getting number of elements allocated for given row.
   const IndexType rowSize = this->rowPointers.getElement( row + 1 ) - this->rowPointers.getElement( row );
   for( IndexType i = 0; i < rowSize; i++ ) {
      // SANDBOX_TODO: Replace the following line with a code for getting index of the matrix element in arrays `values` and
      // `columnIdexes`.
      const IndexType globalIdx = this->rowPointers.getElement( row ) + i;
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      const IndexType col = this->columnIndexes.getElement( globalIdx );
      if( col == column ) {
         if( isBinary() )
            return 1;
         else
            return this->values.getElement( globalIdx );
      }
   }
   return 0.0;
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename InVector, typename OutVector >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::vectorProduct( const InVector& inVector,
                                                                           OutVector& outVector,
                                                                           RealType matrixMultiplicator,
                                                                           RealType outVectorMultiplicator,
                                                                           IndexType firstRow,
                                                                           IndexType lastRow ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   using OutVectorReal = typename OutVector::RealType;
   static_assert(
      ! MatrixType::isSymmetric() || ! std::is_same< Device, Devices::Cuda >::value
         || ( std::is_same< OutVectorReal, float >::value || std::is_same< OutVectorReal, double >::value
              || std::is_same< OutVectorReal, int >::value || std::is_same< OutVectorReal, long long int >::value ),
      "Given Real type is not supported by atomic operations on GPU which are necessary for symmetric operations." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   const auto columnIndexesView = this->columnIndexes.getConstView();
   const auto rowPointersView = this->rowPointers.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
#define HAVE_SANDBOX_SIMPLE_SPMV
   // SANDBOX_TODO: The following is simple direct implementation of SpMV operation with CSR format. We recommend to start by
   //               replacing this part with SpMV based on your sparse format.
   if( std::is_same< DeviceType, TNL::Devices::Host >::value )  // this way you may easily specialize for different device types
   {
      // SANDBOX_TODO: This simple and naive implementation for CPU.
      for( IndexType rowIdx = firstRow; rowIdx < lastRow; rowIdx++ ) {
         const auto begin = rowPointers[ rowIdx ];
         const auto end = rowPointers[ rowIdx + 1 ];
         RealType sum( 0.0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
            const auto columnIdx = this->columnIndexes[ globalIdx ];
            if( columnIdx != paddingIndex )
               sum += this->values[ globalIdx ] * inVector[ columnIdx ];
         }
         // SANDBOX_TODO:The following is quite inefficient, its better to specialized the code for cases when
         // `outVectorMultiplicator` is zero or `matrixMultiplicator` is one - see. the full implementation bellow.
         outVector[ rowIdx ] = outVector[ rowIdx ] * outVectorMultiplicator + matrixMultiplicator * sum;
      }
   }
   else {
      // SANDBOX_TODO: The following is general implementation based on ParallelFor and lambda function. It would work even on
      // CPU.
      auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
      {
         const auto begin = rowPointersView[ rowIdx ];
         const auto end = rowPointersView[ rowIdx + 1 ];
         RealType sum( 0.0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
            sum += valuesView[ globalIdx ] * inVectorView[ columnIndexesView[ globalIdx ] ];
         outVectorView[ rowIdx ] = outVectorView[ rowIdx ] * outVectorMultiplicator + matrixMultiplicator * sum;
      };
      TNL::Algorithms::ParallelFor< DeviceType >::exec( firstRow, lastRow, f );
   }
#ifdef HAVE_SANDBOX_SIMPLE_SPMV
#else
   // SANDBOX_TODO: The following is fully functional implementation based on method `reduceRows`.
   if( isSymmetric() )
      outVector *= outVectorMultiplicator;
   auto symmetricFetch =
      [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType globalIdx, bool& compute ) mutable -> RealType
   {
      const IndexType column = columnIndexesView[ globalIdx ];
      compute = ( column != paddingIndex );
      if( ! compute )
         return 0.0;
      if( isSymmetric() && column < row ) {
         if( isBinary() )
            Algorithms::AtomicOperations< DeviceType >::add( outVectorView[ column ],
                                                             (OutVectorReal) matrixMultiplicator * inVectorView[ row ] );
         else
            Algorithms::AtomicOperations< DeviceType >::add(
               outVectorView[ column ], (OutVectorReal) matrixMultiplicator * valuesView[ globalIdx ] * inVectorView[ row ] );
      }
      if( isBinary() )
         return inVectorView[ column ];
      return valuesView[ globalIdx ] * inVectorView[ column ];
   };
   auto fetch = [ = ] __cuda_callable__( IndexType globalIdx, bool& compute ) mutable -> RealType
   {
      const IndexType column = columnIndexesView[ globalIdx ];
      if( isBinary() )
         return inVectorView[ column ];
      return valuesView[ globalIdx ] * inVectorView[ column ];
   };

   auto keeperGeneral = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      if( isSymmetric() ) {
         typename OutVector::RealType aux = matrixMultiplicator * value;
         Algorithms::AtomicOperations< DeviceType >::add( outVectorView[ row ], aux );
      }
      else {
         if( outVectorMultiplicator == 0.0 )
            outVectorView[ row ] = matrixMultiplicator * value;
         else
            outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + matrixMultiplicator * value;
      }
   };
   auto keeperDirect = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = value;
   };
   auto keeperMatrixMult = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = matrixMultiplicator * value;
   };
   auto keeperVectorMult = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + value;
   };

   if( lastRow == 0 )
      lastRow = this->getRows();
   if( isSymmetric() )
      this->reduceRows( firstRow, lastRow, symmetricFetch, std::plus<>{}, keeperGeneral, (RealType) 0.0 );
   else {
      if( outVectorMultiplicator == 0.0 ) {
         if( matrixMultiplicator == 1.0 )
            this->reduceRows( firstRow, lastRow, fetch, std::plus<>{}, keeperDirect, (RealType) 0.0 );
         else
            this->reduceRows( firstRow, lastRow, fetch, std::plus<>{}, keeperMatrixMult, (RealType) 0.0 );
      }
      else {
         if( matrixMultiplicator == 1.0 )
            this->reduceRows( firstRow, lastRow, fetch, std::plus<>{}, keeperVectorMult, (RealType) 0.0 );
         else
            this->reduceRows( firstRow, lastRow, fetch, std::plus<>{}, keeperGeneral, (RealType) 0.0 );
      }
   }
#endif
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::reduceRows( IndexType begin,
                                                                        IndexType end,
                                                                        Fetch& fetch,
                                                                        const Reduce& reduce,
                                                                        Keep& keep,
                                                                        const FetchValue& zero )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   auto row_pointers_view = this->rowPointers.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   // SANDBOX_TODO: Replace the following code with the one for computing reduction in rows by your format.
   //               Note, that this method can be used for implementation of SpMV.
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      const auto begin = row_pointers_view[ rowIdx ];
      const auto end = row_pointers_view[ rowIdx + 1 ];
      FetchValue sum = zero;
      for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
         IndexType& columnIdx = columns_view[ globalIdx ];
         if( columnIdx != paddingIndex_ ) {
            if( isBinary() )
               sum = reduce( sum, fetch( rowIdx, columnIdx, 1 ) );
            else
               sum = reduce( sum, fetch( rowIdx, columnIdx, values_view[ globalIdx ] ) );
         }
      }
      keep( rowIdx, sum );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::reduceRows( IndexType begin,
                                                                        IndexType end,
                                                                        Fetch& fetch,
                                                                        const Reduce& reduce,
                                                                        Keep& keep,
                                                                        const FetchValue& zero ) const
{
   auto columns_view = this->columnIndexes.getConstView();
   auto values_view = this->values.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   // SANDBOX_TODO: Replace the following code with the one for computing reduction in rows by your format.
   //               Note, that this method can be used for implementation of SpMV.
   auto row_pointers_view = this->rowPointers.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      const auto begin = row_pointers_view[ rowIdx ];
      const auto end = row_pointers_view[ rowIdx + 1 ];
      FetchValue sum = zero;
      for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
         const IndexType& columnIdx = columns_view[ globalIdx ];
         if( columnIdx != paddingIndex_ ) {
            if( isBinary() )
               sum = reduce( sum, fetch( rowIdx, columnIdx, 1 ) );
            else
               sum = reduce( sum, fetch( rowIdx, columnIdx, values_view[ globalIdx ] ) );
         }
      }
      keep( rowIdx, sum );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::reduceAllRows( Fetch& fetch,
                                                                           const Reduce& reduce,
                                                                           Keep& keep,
                                                                           const FetchReal& zero )
{
   this->reduceRows( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::reduceAllRows( Fetch& fetch,
                                                                           const Reduce& reduce,
                                                                           Keep& keep,
                                                                           const FetchReal& zero ) const
{
   this->reduceRows( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forElements( IndexType begin,
                                                                         IndexType end,
                                                                         Function& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   // SANDBOX_TODO: Replace the following code with the one for iterating over all allocated matrix elements.
   auto row_pointers_view = this->rowPointers.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      const auto begin = row_pointers_view[ rowIdx ];
      const auto end = row_pointers_view[ rowIdx + 1 ];
      IndexType localIdx( 0 );
      for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
         if( isBinary() )
            function( rowIdx, localIdx, columns_view[ globalIdx ], (RealType) 1 );
         else
            function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
         localIdx++;
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forElements( IndexType begin, IndexType end, Function& function )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   // SANDBOX_TODO: Replace the following code with the one for iterating over all allocated matrix elements.
   auto row_pointers_view = this->rowPointers.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      const auto begin = row_pointers_view[ rowIdx ];
      const auto end = row_pointers_view[ rowIdx + 1 ];
      IndexType localIdx( 0 );
      RealType one( 1.0 );
      for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
         if( isBinary() )
            function( rowIdx, localIdx, columns_view[ globalIdx ], one );  // TODO: Fix this without using `one`.
         else
            function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
         localIdx++;
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forAllElements( Function& function ) const
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forAllElements( Function& function )
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forRows( IndexType begin, IndexType end, Function&& function )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   // SANDBOX_TODO: Replace the following code with the one for iteration over matrix rows.
   auto row_pointers_view = this->rowPointers.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      auto rowView = RowView( rowIdx,                       // row index
                              row_pointers_view[ rowIdx ],  // row begining
                              row_pointers_view[ rowIdx + 1 ]
                                 - row_pointers_view[ rowIdx ],  // number of elemnts allocated for given matrix row
                              values_view,
                              columns_view );
      function( rowView );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forRows( IndexType begin, IndexType end, Function&& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   // SANDBOX_TODO: Replace the following code with the one for iteration over matrix rows.
   auto row_pointers_view = this->rowPointers.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx )
   {
      auto rowView = ConstRowView( rowIdx,                       // row index
                                   row_pointers_view[ rowIdx ],  // row begining
                                   row_pointers_view[ rowIdx + 1 ]
                                      - row_pointers_view[ rowIdx ],  // number of elemnts allocated for given matrix row
                                   values_view,
                                   columns_view );
      function( rowView );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forAllRows( Function&& function )
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::forAllRows( Function&& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::sequentialForRows( IndexType begin,
                                                                               IndexType end,
                                                                               Function& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::sequentialForRows( IndexType begin,
                                                                               IndexType end,
                                                                               Function& function )
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Function >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::sequentialForAllRows( Function& function )
{
   this->sequentialForRows( 0, this->getRows(), function );
}

/*template< typename Real,
          template< typename, typename > class SegmentsView,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, template< typename, typename > class Segments2, typename Index2, typename RealAllocator2, typename
IndexAllocator2 > void SparseSandboxMatrixView< Real, Device, Index, MatrixType >:: addMatrix( const SparseSandboxMatrixView<
Real2, Segments2, Device, Index2, RealAllocator2, IndexAllocator2 >& matrix, const RealType& matrixMultiplicator, const
RealType& thisMatrixMultiplicator )
{

}

template< typename Real,
          template< typename, typename > class SegmentsView,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, typename Index2 >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::
getTransposition( const SparseSandboxMatrixView< Real2, Device, Index2 >& matrix,
                  const RealType& matrixMultiplicator )
{

}*/

template< typename Real, typename Device, typename Index, typename MatrixType >
SparseSandboxMatrixView< Real, Device, Index, MatrixType >&
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::operator=(
   const SparseSandboxMatrixView< Real, Device, Index, MatrixType >& matrix )
{
   MatrixView< Real, Device, Index >::operator=( matrix );
   this->columnIndexes.bind( matrix.columnIndexes );
   // SANDBOX_TODO: Replace the following line with assignment of metadata required by your
   //               sparse format.
   this->rowPointers.bind( matrix.rowPointers );
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Matrix >
bool
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::operator==( const Matrix& m ) const
{
   const auto& view1 = *this;
   // FIXME: getConstView does not work
   // const auto view2 = m.getConstView();
   const auto view2 = m.getView();
   auto fetch = [ = ] __cuda_callable__( const IndexType i ) -> bool
   {
      return view1.getRow( i ) == view2.getRow( i );
   };
   return Algorithms::reduce< DeviceType >( 0, this->getRows(), fetch, std::logical_and<>{}, true );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
template< typename Matrix >
bool
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::operator!=( const Matrix& m ) const
{
   return ! operator==( m );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
   file << this->columnIndexes << this->rowPointers;  // SANDBOX_TODO: Replace this with medata required by your format
}

template< typename Real, typename Device, typename Index, typename MatrixType >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real, typename Device, typename Index, typename MatrixType >
void
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::print( std::ostream& str ) const
{
   if( isSymmetric() ) {
      for( IndexType row = 0; row < this->getRows(); row++ ) {
         str << "Row: " << row << " -> ";
         for( IndexType column = 0; column < this->getColumns(); column++ ) {
            auto value = this->getElement( row, column );
            if( value != (RealType) 0 )
               str << " Col:" << column << "->" << value << "\t";
         }
         str << std::endl;
      }
   }
   else
      for( IndexType row = 0; row < this->getRows(); row++ ) {
         str << "Row: " << row << " -> ";
         // SANDBOX_TODO: Replace the followinf line with a code for computing number of elements allocated for given matrix
         // row.
         const auto rowLength = this->rowPointers.getElement( row + 1 ) - this->rowPointers.getElement( row );
         for( IndexType i = 0; i < rowLength; i++ ) {
            // SANDBOX_TODO: Replace the following line with a code for getting index of the matrix element in arrays `values`
            // and `columnIdexes`.
            const IndexType globalIdx = this->rowPointers.getElement( row ) + i;
            const IndexType column = this->columnIndexes.getElement( globalIdx );
            if( column == this->getPaddingIndex() )
               break;
            std::decay_t< RealType > value;
            if( isBinary() )
               value = (RealType) 1.0;
            else
               value = this->values.getElement( globalIdx );
            if( value ) {
               std::stringstream str_;
               str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << value;
               str << std::setw( 10 ) << str_.str();
            }
         }
         str << std::endl;
      }
}

template< typename Real, typename Device, typename Index, typename MatrixType >
__cuda_callable__
Index
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getPaddingIndex() const
{
   return -1;
}

template< typename Real, typename Device, typename Index, typename MatrixType >
auto
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getColumnIndexes() const -> const ColumnsIndexesViewType&
{
   return this->columnIndexes;
}

template< typename Real, typename Device, typename Index, typename MatrixType >
auto
SparseSandboxMatrixView< Real, Device, Index, MatrixType >::getColumnIndexes() -> ColumnsIndexesViewType&
{
   return this->columnIndexes;
}

}  // namespace Sandbox
}  // namespace Matrices
}  // namespace  TNL
