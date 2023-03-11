// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <functional>
#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Matrices {

/**
 * The following kernel is an attempt to map more CUDA threads to one matrix row.
 */
template< int BlockSize, int ThreadsPerRow, typename Matrix, typename InVector, typename OutVector >
__global__
void
VectorColumnMajorDenseMatrixViewVectorMultiplicationKernel( const Matrix matrix,
                                                            const InVector inVector,
                                                            OutVector outVector,
                                                            const int begin,
                                                            const int end,
                                                            int gridIdx )
{
#ifdef __CUDACC__
   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   constexpr int inVectorCacheSize = 20480 / sizeof( Real );
   __shared__ Real inVectorCache[ inVectorCacheSize ];
   __shared__ Real result_[ BlockSize ];

   constexpr Index rowsPerBlock = 256 / ThreadsPerRow;
   const Index rowIdx = ( ( gridIdx * Cuda::getMaxGridXSize() + blockIdx.x ) * 256 + threadIdx.x ) / ThreadsPerRow + begin;
   const Index localColIdx = threadIdx.x / rowsPerBlock;
   const Index localRowIdx = threadIdx.x % rowsPerBlock;

   Real result( 0.0 );
   Index columnIdx( 0 );
   const auto& values = matrix.getValues();
   const auto& rowsCount = matrix.getRows();
   Index valuesPtr = rowIdx + localColIdx * rowsCount;

   while( columnIdx < matrix.getColumns() ) {
      const Index lastIdx = min( matrix.getColumns(), columnIdx + inVectorCacheSize );
      Index matrixColIdx = columnIdx + threadIdx.x;
      Index cacheColIdx = threadIdx.x;
      while( matrixColIdx < lastIdx ) {
         inVectorCache[ cacheColIdx ] = inVector[ matrixColIdx ];
         cacheColIdx += 256;
         matrixColIdx += 256;
      }
      __syncthreads();

      matrixColIdx = columnIdx + localColIdx;
      cacheColIdx = localColIdx;
      if( rowIdx < end )
         while( matrixColIdx < lastIdx ) {
            result += values[ valuesPtr ] * inVectorCache[ cacheColIdx ];
            cacheColIdx += ThreadsPerRow;
            matrixColIdx += ThreadsPerRow;
            valuesPtr += ThreadsPerRow * rowsCount;
         }
      columnIdx = lastIdx;
   }
   const int idx = localRowIdx * ThreadsPerRow + localColIdx;
   result_[ idx ] = result;
   if( ThreadsPerRow > 8 && localColIdx < ThreadsPerRow - 8 )
      result_[ idx ] += result_[ idx + 8 ];
   __syncwarp();
   if( ThreadsPerRow > 4 && localColIdx < ThreadsPerRow - 4 )
      result_[ idx ] += result_[ idx + 4 ];
   __syncwarp();
   if( ThreadsPerRow > 2 && localColIdx < ThreadsPerRow - 2 )
      result_[ idx ] += result_[ idx + 2 ];
   __syncwarp();
   if( ThreadsPerRow > 1 && localColIdx < ThreadsPerRow - 1 )
      result_[ idx ] += result_[ idx + 1 ];
   __syncwarp();

   if( rowIdx < end && localColIdx == 0 )
      outVector[ rowIdx ] = result_[ idx ];
#endif
}

template< typename Matrix, typename InVector, typename OutVector >
__global__
void
ColumnMajorDenseMatrixViewVectorMultiplicationKernel( const Matrix matrix,
                                                      const InVector inVector,
                                                      OutVector outVector,
                                                      const int begin,
                                                      const int end,
                                                      int gridIdx )
{
#ifdef __CUDACC__
   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   constexpr int inVectorCacheSize = 20480 / sizeof( Real );
   __shared__ Real inVectorCache[ inVectorCacheSize ];

   const int rowIdx = ( gridIdx * Cuda::getMaxGridXSize() + blockIdx.x ) * 256 + threadIdx.x + begin;

   Real result( 0.0 );
   Index columnIdx( 0 );
   const auto& values = matrix.getValues();
   const auto& rowsCount = matrix.getRows();
   Index valuesPtr = rowIdx;

   while( columnIdx < matrix.getColumns() ) {
      const Index lastIdx = min( matrix.getColumns(), columnIdx + inVectorCacheSize );
      Index matrixColIdx = columnIdx + threadIdx.x;
      Index cacheColIdx = threadIdx.x;
      while( matrixColIdx < lastIdx ) {
         inVectorCache[ cacheColIdx ] = inVector[ matrixColIdx ];
         cacheColIdx += 256;
         matrixColIdx += 256;
      }
      __syncthreads();

      matrixColIdx = columnIdx;
      cacheColIdx = 0;
      if( rowIdx < end )
         while( matrixColIdx < lastIdx ) {
            result += values[ valuesPtr ] * inVectorCache[ cacheColIdx ];
            cacheColIdx++;
            matrixColIdx++;
            valuesPtr += rowsCount;
         }
      columnIdx = lastIdx;
   }
   if( rowIdx < end )
      outVector[ rowIdx ] = result;
#endif
}

template< typename Matrix, typename InVector, typename OutVector >
__global__
void
RowMajorDenseMatrixViewVectorMultiplicationKernel( const Matrix matrix,
                                                   const InVector inVector,
                                                   OutVector outVector,
                                                   const int first,
                                                   const int last,
                                                   int gridIdx )
{
#ifdef __CUDACC__
   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   // constexpr int inVectorCacheSize = 20480 / sizeof( Real );
   //__shared__ Real inVectorCache[ inVectorCacheSize ];

   constexpr int threadsPerRow = 32;
   // const Index rowIdx = begin + ((gridIdx * TNL::Cuda::getMaxGridXSize() ) + (blockIdx.x * blockDim.x) + threadIdx.x) /
   // threadsPerRow;
   const Index rowIdx = first + ( ( gridIdx * Cuda::getMaxGridXSize() + blockIdx.x ) * 256 + threadIdx.x ) / threadsPerRow;

   Real result = 0.0;
   const Index laneID = threadIdx.x & 31;  // & is cheaper than %
   const Real* values = matrix.getValues().getData();

   Index columnIdx( 0 );
   /*while( columnIdx < matrix.getColumns() )
   {
      const Index lastIdx = min( matrix.getColumns(), columnIdx + inVectorCacheSize );
      Index matrixColIdx = columnIdx + threadIdx.x;
      Index cacheColIdx = threadIdx.x;
      while( matrixColIdx < lastIdx )
      {
         inVectorCache[ cacheColIdx ] = inVector[ matrixColIdx ];
         cacheColIdx += 256;
         matrixColIdx += 256;
      }
      __syncthreads();

      // Calculate result
      if( rowIdx < last )
      {
         const Index begin = rowIdx * matrix.getColumns() + columnIdx;
         const Index end = rowIdx * matrix.getColumns() + lastIdx;
         Index localColumn( 0 );

         for( Index i = begin + laneID; i < end; i += threadsPerRow, localColumn += threadsPerRow )
            result += values[ i ] * inVectorCache[ localColumn ];
      }
      columnIdx = lastIdx;
   }*/

   if( rowIdx < last ) {
      const Index begin = rowIdx * matrix.getColumns();
      const Index end = begin + matrix.getColumns();

      for( Index i = begin + laneID; i < end; i += threadsPerRow, columnIdx += threadsPerRow )
         result += values[ i ] * inVector[ columnIdx ];
   }

   if( rowIdx < last ) {
      // Reduction
      if( threadsPerRow > 16 )
         result += __shfl_down_sync( 0xFFFFFFFF, result, 16 );
      if( threadsPerRow > 8 )
         result += __shfl_down_sync( 0xFFFFFFFF, result, 8 );
      if( threadsPerRow > 4 )
         result += __shfl_down_sync( 0xFFFFFFFF, result, 4 );
      if( threadsPerRow > 2 )
         result += __shfl_down_sync( 0xFFFFFFFF, result, 2 );
      if( threadsPerRow > 1 )
         result += __shfl_down_sync( 0xFFFFFFFF, result, 1 );
      // Write result
      if( laneID == 0 )
         outVector[ rowIdx ] = result;
   }
#endif
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::DenseMatrixView() = default;

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::DenseMatrixView( IndexType rows,
                                                                       IndexType columns,
                                                                       const ValuesViewType& values )
: MatrixView< Real, Device, Index >( rows, columns, values ), segments( rows, columns )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Value_ >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::DenseMatrixView(
   IndexType rows,
   IndexType columns,
   const Containers::VectorView< Value_, Device, Index >& values )
: MatrixView< Real, Device, Index >( rows, columns, values ), segments( rows, columns, true )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::getView() -> ViewType
{
   return ViewType( this->getRows(), this->getColumns(), this->getValues().getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return ConstViewType(
      this->getRows(), this->getColumns(), this->getValues().getConstView(), this->getColumnsIndexes().getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::string
DenseMatrixView< Real, Device, Index, Organization >::getSerializationType()
{
   return "Matrices::DenseMatrix< " + TNL::getSerializationType< RealType >() + ", [any_device], "
        + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + " >";
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::string
DenseMatrixView< Real, Device, Index, Organization >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
DenseMatrixView< Real, Device, Index, Organization >::getRowCapacities( Vector& rowCapacities ) const
{
   rowCapacities.setSize( this->getRows() );
   rowCapacities = this->getColumns();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
DenseMatrixView< Real, Device, Index, Organization >::getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return ( value != 0.0 );
   };
   auto keep = [ = ] __cuda_callable__( IndexType rowIdx, IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::getAllocatedElementsCount() const
{
   return this->getRows() * this->getColumns();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> IndexType
   {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->values.getSize(), fetch, std::plus<>{}, 0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
DenseMatrixView< Real, Device, Index, Organization >::setValue( const Real& value )
{
   this->values = value;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::getRow( IndexType rowIdx ) const -> ConstRowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::getRow( IndexType rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Real&
DenseMatrixView< Real, Device, Index, Organization >::operator()( IndexType row, IndexType column )
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
const Real&
DenseMatrixView< Real, Device, Index, Organization >::operator()( IndexType row, IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixView< Real, Device, Index, Organization >::setElement( IndexType row, IndexType column, const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixView< Real, Device, Index, Organization >::addElement( IndexType row,
                                                                  IndexType column,
                                                                  const RealType& value,
                                                                  const RealType& thisElementMultiplicator )
{
   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->values.setElement( elementIndex, this->values.getElement( elementIndex ) + value );
   else
      this->values.setElement( elementIndex, thisElementMultiplicator * this->values.getElement( elementIndex ) + value );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Real
DenseMatrixView< Real, Device, Index, Organization >::getElement( IndexType row, IndexType column ) const
{
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
DenseMatrixView< Real, Device, Index, Organization >::reduceRows( IndexType begin,
                                                                  IndexType end,
                                                                  Fetch& fetch,
                                                                  const Reduce& reduce,
                                                                  Keep& keep,
                                                                  const FetchValue& identity )
{
   auto values_view = this->values.getView();
   auto fetch_ = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable
      -> decltype( fetch( IndexType(), IndexType(), RealType() ) )
   {
      return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      return identity;
   };
   this->segments.reduceSegments( begin, end, fetch_, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
DenseMatrixView< Real, Device, Index, Organization >::reduceRows( IndexType begin,
                                                                  IndexType end,
                                                                  Fetch& fetch,
                                                                  const Reduce& reduce,
                                                                  Keep& keep,
                                                                  const FetchValue& identity ) const
{
   const auto values_view = this->values.getConstView();
   auto fetch_ = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable
      -> decltype( fetch( IndexType(), IndexType(), RealType() ) )
   {
      return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      return identity;
   };
   this->segments.reduceSegments( begin, end, fetch_, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
DenseMatrixView< Real, Device, Index, Organization >::reduceAllRows( Fetch& fetch,
                                                                     const Reduce& reduce,
                                                                     Keep& keep,
                                                                     const FetchReal& identity )
{
   this->reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
DenseMatrixView< Real, Device, Index, Organization >::reduceAllRows( Fetch& fetch,
                                                                     const Reduce& reduce,
                                                                     Keep& keep,
                                                                     const FetchReal& identity ) const
{
   this->reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forElements( IndexType begin, IndexType end, Function&& function ) const
{
   const auto values_view = this->values.getConstView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx ) mutable
   {
      function( rowIdx, columnIdx, columnIdx, values_view[ globalIdx ] );
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forElements( IndexType begin, IndexType end, Function&& function )
{
   auto values_view = this->values.getView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx ) mutable
   {
      function( rowIdx, columnIdx, globalIdx, values_view[ globalIdx ] );
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forAllElements( Function&& function ) const
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forAllElements( Function&& function )
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forRows( IndexType begin, IndexType end, Function&& function )
{
   auto values_view = this->values.getView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
   {
      auto rowView = RowView( segmentView, values_view );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forRows( IndexType begin, IndexType end, Function&& function ) const
{
   const auto values_view = this->values.getConstView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( SegmentViewType && segmentView ) mutable
   {
      const auto rowView = RowViewType( segmentView, values_view );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forAllRows( Function&& function )
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::forAllRows( Function&& function ) const
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::sequentialForRows( IndexType begin,
                                                                         IndexType end,
                                                                         Function&& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::sequentialForRows( IndexType begin, IndexType end, Function&& function )
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::sequentialForAllRows( Function&& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::sequentialForAllRows( Function&& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename InVector, typename OutVector >
void
DenseMatrixView< Real, Device, Index, Organization >::vectorProduct( const InVector& inVector,
                                                                     OutVector& outVector,
                                                                     const RealType& matrixMultiplicator,
                                                                     const RealType& outVectorMultiplicator,
                                                                     IndexType begin,
                                                                     IndexType end ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns count differs with input vector size." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows count differs with output vector size." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   if( end == 0 )
      end = this->getRows();

   if( matrixMultiplicator == 1.0 && outVectorMultiplicator == 0.0 ) {
      if constexpr( std::is_same< DeviceType, Devices::Cuda >::value ) {
         if constexpr( Organization == Algorithms::Segments::ColumnMajorOrder ) {
            Cuda::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            constexpr int ThreadsPerRow = 1;
            const std::size_t threadsCount = ( end - begin ) * ThreadsPerRow;
            const std::size_t blocksCount = roundUpDivision( threadsCount, launch_config.blockSize.x );
            const std::size_t gridsCount = roundUpDivision( blocksCount, Cuda::getMaxGridXSize() );
            launch_config.dynamicSharedMemorySize = 20480;
            for( std::size_t gridIdx = 0; gridIdx < gridsCount; gridIdx++ ) {
               launch_config.gridSize.x = Cuda::getMaxGridXSize();
               if( gridIdx == gridsCount - 1 )
                  launch_config.gridSize.x = blocksCount % Cuda::getMaxGridXSize();
               constexpr auto kernel = ColumnMajorDenseMatrixViewVectorMultiplicationKernel< DenseMatrixView,
                                                                                             decltype( inVectorView ),
                                                                                             decltype( outVectorView ) >;
               Cuda::launchKernelAsync( kernel, launch_config, *this, inVectorView, outVectorView, begin, end, gridIdx );
            }
            cudaStreamSynchronize( launch_config.stream );
            TNL_CHECK_CUDA_DEVICE;
            return;
         }
         if constexpr( Organization == Algorithms::Segments::RowMajorOrder ) {
            Cuda::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            constexpr int ThreadsPerRow = 32;
            const std::size_t threadsCount = ( end - begin ) * ThreadsPerRow;
            const std::size_t blocksCount = roundUpDivision( threadsCount, launch_config.blockSize.x );
            const std::size_t gridsCount = roundUpDivision( blocksCount, Cuda::getMaxGridXSize() );
            launch_config.dynamicSharedMemorySize = 20480;
            for( std::size_t gridIdx = 0; gridIdx < gridsCount; gridIdx++ ) {
               launch_config.gridSize.x = Cuda::getMaxGridXSize();
               if( gridIdx == gridsCount - 1 )
                  launch_config.gridSize.x = blocksCount % Cuda::getMaxGridXSize();
               constexpr auto kernel = RowMajorDenseMatrixViewVectorMultiplicationKernel< DenseMatrixView,
                                                                                          decltype( inVectorView ),
                                                                                          decltype( outVectorView ) >;
               Cuda::launchKernelAsync( kernel, launch_config, *this, inVectorView, outVectorView, begin, end, gridIdx );
            }
            cudaStreamSynchronize( launch_config.stream );
            TNL_CHECK_CUDA_DEVICE;
            return;
         }
      }
   }

   /***
    * The rest is general implementation based on segments
    */

   auto fetch = [ = ] __cuda_callable__( IndexType row, IndexType column, IndexType offset, bool& compute ) -> RealType
   {
      return valuesView[ offset ] * inVectorView[ column ];
   };
   auto keeperGeneral = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = matrixMultiplicator * value + outVectorMultiplicator * outVectorView[ row ];
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

   if( outVectorMultiplicator == 0.0 ) {
      if( matrixMultiplicator == 1.0 )
         this->segments.reduceSegments( begin, end, fetch, std::plus<>{}, keeperDirect, (RealType) 0.0 );
      else
         this->segments.reduceSegments( begin, end, fetch, std::plus<>{}, keeperMatrixMult, (RealType) 0.0 );
   }
   else {
      if( matrixMultiplicator == 1.0 )
         this->segments.reduceSegments( begin, end, fetch, std::plus<>{}, keeperVectorMult, (RealType) 0.0 );
      else
         this->segments.reduceSegments( begin, end, fetch, std::plus<>{}, keeperGeneral, (RealType) 0.0 );
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Matrix >
void
DenseMatrixView< Real, Device, Index, Organization >::addMatrix( const Matrix& matrix,
                                                                 const RealType& matrixMultiplicator,
                                                                 const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT( this->getColumns() == matrix.getColumns() && this->getRows() == matrix.getRows(),
               std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                         << "This matrix rows: " << this->getRows() << std::endl
                         << "That matrix columns: " << matrix.getColumns() << std::endl
                         << "That matrix rows: " << matrix.getRows() << std::endl );

   if( thisMatrixMultiplicator == 1.0 )
      this->values += matrixMultiplicator * matrix.values;
   else
      this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.values;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
DenseMatrixView< Real, Device, Index, Organization >&
DenseMatrixView< Real, Device, Index, Organization >::operator=( const DenseMatrixView& matrix )
{
   MatrixView< Real, Device, Index >::operator=( matrix );
   this->segments = matrix.segments;
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_ >
bool
DenseMatrixView< Real, Device, Index, Organization >::operator==(
   const DenseMatrixView< Real_, Device_, Index_, Organization >& matrix ) const
{
   return ( this->getRows() == matrix.getRows() && this->getColumns() == matrix.getColumns()
            && this->getValues() == matrix.getValues() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_ >
bool
DenseMatrixView< Real, Device, Index, Organization >::operator!=(
   const DenseMatrixView< Real_, Device_, Index_, Organization >& matrix ) const
{
   return ! ( *this == matrix );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Matrix >
bool
DenseMatrixView< Real, Device, Index, Organization >::operator==( const Matrix& m ) const
{
   const auto& view1 = *this;
   const auto view2 = m.getConstView();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> bool
   {
      return view1.getRow( i ) == view2.getRow( i );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->getRows(), fetch, std::logical_and<>{}, true );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Matrix >
bool
DenseMatrixView< Real, Device, Index, Organization >::operator!=( const Matrix& m ) const
{
   return ! ( *this == m );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
DenseMatrixView< Real, Device, Index, Organization >::save( const String& fileName ) const
{
   MatrixView< Real, Device, Index >::save( fileName );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
DenseMatrixView< Real, Device, Index, Organization >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
   this->segments.save( file );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
DenseMatrixView< Real, Device, Index, Organization >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ ) {
      str << "Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ ) {
         std::stringstream str_;
         str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left
              << this->getElement( row, column );
         str << std::setw( 10 ) << str_.str();
      }
      if( row < this->getRows() - 1 )
         str << std::endl;
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Index
DenseMatrixView< Real, Device, Index, Organization >::getElementIndex( IndexType row, IndexType column ) const
{
   return this->segments.getGlobalIndex( row, column );
}

}  // namespace Matrices
}  // namespace noa::TNL
