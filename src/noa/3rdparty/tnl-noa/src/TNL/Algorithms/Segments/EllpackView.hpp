// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/EllpackView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/detail/LambdaAdapter.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

#ifdef HAVE_CUDA
template< typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
__global__
void
EllpackCudaReductionKernelFull( Index first,
                                Index last,
                                Fetch fetch,
                                const Reduction reduction,
                                ResultKeeper keep,
                                const Real zero,
                                Index segmentSize )
{
   const int warpSize = 32;
   const int gridID = 0;
   const Index segmentIdx =
      first + ( ( gridID * TNL::Cuda::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / warpSize;
   if( segmentIdx >= last )
      return;

   Real result = zero;
   const Index laneID = threadIdx.x & 31;  // & is cheaper than %
   const Index begin = segmentIdx * segmentSize;
   const Index end = begin + segmentSize;

   /* Calculate result */
   Index localIdx( 0 );
   bool compute( true );
   for( Index i = begin + laneID; i < end; i += warpSize )
      result = reduction( result, fetch( segmentIdx, localIdx++, i, compute ) );

   /* Reduction */
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   /* Write result */
   if( laneID == 0 )
      keep( segmentIdx, result );
}

template< typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
__global__
void
EllpackCudaReductionKernelCompact( Index first,
                                   Index last,
                                   Fetch fetch,
                                   const Reduction reduction,
                                   ResultKeeper keep,
                                   const Real zero,
                                   Index segmentSize )
{
   const int warpSize = 32;
   const int gridID = 0;
   const Index segmentIdx =
      first + ( ( gridID * TNL::Cuda::getMaxGridXSize() ) + ( blockIdx.x * blockDim.x ) + threadIdx.x ) / warpSize;
   if( segmentIdx >= last )
      return;

   Real result = zero;
   const Index laneID = threadIdx.x & 31;  // & is cheaper than %
   const Index begin = segmentIdx * segmentSize;
   const Index end = begin + segmentSize;

   /* Calculate result */
   bool compute( true );
   for( Index i = begin + laneID; i < end; i += warpSize )
      result = reduction( result, fetch( i, compute ) );

   /* Reduction */
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
   result = reduction( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   /* Write result */
   if( laneID == 0 )
      keep( segmentIdx, result );
}
#endif

template< typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Real,
          bool FullFetch = detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() >
struct EllpackCudaReductionDispatcher
{
   static void
   exec( Index first,
         Index last,
         Fetch& fetch,
         const Reduction& reduction,
         ResultKeeper& keeper,
         const Real& zero,
         Index segmentSize )
   {
#ifdef HAVE_CUDA
      if( last <= first )
         return;
      const Index segmentsCount = last - first;
      const Index threadsCount = segmentsCount * 32;
      const Index blocksCount = Cuda::getNumberOfBlocks( threadsCount, 256 );
      dim3 blockSize( 256 );
      dim3 gridSize( blocksCount );
      EllpackCudaReductionKernelFull<<< gridSize,
         blockSize >>>( first, last, fetch, reduction, keeper, zero, segmentSize );
      cudaStreamSynchronize( 0 );
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
};

template< typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
struct EllpackCudaReductionDispatcher< Index, Fetch, Reduction, ResultKeeper, Real, false >
{
   static void
   exec( Index first,
         Index last,
         Fetch& fetch,
         const Reduction& reduction,
         ResultKeeper& keeper,
         const Real& zero,
         Index segmentSize )
   {
#ifdef HAVE_CUDA
      if( last <= first )
         return;
      const Index segmentsCount = last - first;
      const Index threadsCount = segmentsCount * 32;
      const Index blocksCount = Cuda::getNumberOfBlocks( threadsCount, 256 );
      dim3 blockSize( 256 );
      dim3 gridSize( blocksCount );
      EllpackCudaReductionKernelCompact<<< gridSize,
         blockSize >>>( first, last, fetch, reduction, keeper, zero, segmentSize );
      cudaStreamSynchronize( 0 );
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
};

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::EllpackView( IndexType segmentsCount,
                                                                    IndexType segmentSize,
                                                                    IndexType alignedSize )
: segmentSize( segmentSize ), segmentsCount( segmentsCount ), alignedSize( alignedSize )
{}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
EllpackView< Device, Index, Organization, Alignment >::EllpackView( IndexType segmentsCount, IndexType segmentSize )
: segmentSize( segmentSize ), segmentsCount( segmentsCount )
{
   if( Organization == RowMajorOrder )
      this->alignedSize = this->segmentsCount;
   else
      this->alignedSize = roundUpDivision( segmentsCount, this->getAlignment() ) * this->getAlignment();
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
std::string
EllpackView< Device, Index, Organization, Alignment >::getSerializationType()
{
   // FIXME: the serialized data DEPEND on the Organization and Alignment parameters, so it should be reflected in the
   // serialization type
   return "Ellpack< [any_device], " + TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
String
EllpackView< Device, Index, Organization, Alignment >::getSegmentsType()
{
   return "Ellpack";
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
typename EllpackView< Device, Index, Organization, Alignment >::ViewType
EllpackView< Device, Index, Organization, Alignment >::getView()
{
   return ViewType( segmentsCount, segmentSize, alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getConstView() const -> const ConstViewType
{
   return ConstViewType( segmentsCount, segmentSize, alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSegmentsCount() const -> IndexType
{
   return this->segmentsCount;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   return this->segmentSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSize() const -> IndexType
{
   return this->segmentsCount * this->segmentSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getStorageSize() const -> IndexType
{
   return this->alignedSize * this->segmentSize;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getGlobalIndex( const Index segmentIdx, const Index localIdx ) const
   -> IndexType
{
   if( Organization == RowMajorOrder )
      return segmentIdx * this->segmentSize + localIdx;
   else
      return segmentIdx + this->alignedSize * localIdx;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
__cuda_callable__
auto
EllpackView< Device, Index, Organization, Alignment >::getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( Organization == RowMajorOrder )
      return SegmentViewType( segmentIdx, segmentIdx * this->segmentSize, this->segmentSize, 1 );
   else
      return SegmentViewType( segmentIdx, segmentIdx, this->segmentSize, this->alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forElements( IndexType first, IndexType last, Function&& f ) const
{
   if( Organization == RowMajorOrder ) {
      const IndexType segmentSize = this->segmentSize;
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType begin = segmentIdx * segmentSize;
         const IndexType end = begin + segmentSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
            f( segmentIdx, localIdx++, globalIdx );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
   else {
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType begin = segmentIdx;
         const IndexType end = storageSize;
         IndexType localIdx( 0 );
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx += alignedSize )
            f( segmentIdx, localIdx++, globalIdx );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [ = ] __cuda_callable__( IndexType segmentIdx ) mutable
   {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Function >
void
EllpackView< Device, Index, Organization, Alignment >::forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
EllpackView< Device, Index, Organization, Alignment >::reduceSegments( IndexType first,
                                                                       IndexType last,
                                                                       Fetch& fetch,
                                                                       const Reduction& reduction,
                                                                       ResultKeeper& keeper,
                                                                       const Real& zero ) const
{
   // using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >() ) );
   using RealType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if( Organization == RowMajorOrder ) {
      if( std::is_same< Device, Devices::Cuda >::value )
         EllpackCudaReductionDispatcher< IndexType, Fetch, Reduction, ResultKeeper, Real >::exec(
            first, last, fetch, reduction, keeper, zero, segmentSize );
      else {
         const IndexType segmentSize = this->segmentSize;
         auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
         {
            const IndexType begin = segmentIdx * segmentSize;
            const IndexType end = begin + segmentSize;
            Real aux( zero );
            IndexType localIdx( 0 );
            bool compute( true );
            for( IndexType j = begin; j < end && compute; j++ )
               aux = reduction(
                  aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
            keeper( segmentIdx, aux );
         };
         Algorithms::ParallelFor< Device >::exec( first, last, l );
      }
   }
   else {
      const IndexType storageSize = this->getStorageSize();
      const IndexType alignedSize = this->alignedSize;
      auto l = [ = ] __cuda_callable__( const IndexType segmentIdx ) mutable
      {
         const IndexType begin = segmentIdx;
         const IndexType end = storageSize;
         RealType aux( zero );
         IndexType localIdx( 0 );
         bool compute( true );
         for( IndexType j = begin; j < end && compute; j += alignedSize )
            aux = reduction(
               aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, j, compute ) );
         keeper( segmentIdx, aux );
      };
      Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
EllpackView< Device, Index, Organization, Alignment >::reduceAllSegments( Fetch& fetch,
                                                                          const Reduction& reduction,
                                                                          ResultKeeper& keeper,
                                                                          const Real& zero ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
EllpackView< Device, Index, Organization, Alignment >&
EllpackView< Device, Index, Organization, Alignment >::operator=(
   const EllpackView< Device, Index, Organization, Alignment >& view )
{
   this->segmentSize = view.segmentSize;
   this->segmentsCount = view.segmentsCount;
   this->alignedSize = view.alignedSize;
   return *this;
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
void
EllpackView< Device, Index, Organization, Alignment >::save( File& file ) const
{
   file.save( &segmentSize );
   file.save( &segmentsCount );
   file.save( &alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
void
EllpackView< Device, Index, Organization, Alignment >::load( File& file )
{
   file.load( &segmentSize );
   file.load( &segmentsCount );
   file.load( &alignedSize );
}

template< typename Device, typename Index, ElementsOrganization Organization, int Alignment >
template< typename Fetch >
auto
EllpackView< Device, Index, Organization, Alignment >::print( Fetch&& fetch ) const -> SegmentsPrinter< EllpackView, Fetch >
{
   return SegmentsPrinter< EllpackView, Fetch >( *this, fetch );
}

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL
