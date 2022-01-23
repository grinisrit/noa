// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/BiEllpackView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/LambdaAdapter.h>
//#include <noa/3rdparty/TNL/Algorithms/Segments/detail/BiEllpack.h>
#include <noa/3rdparty/TNL/Cuda/SharedMemory.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, Organization, WarpSize >::
BiEllpackView( const IndexType size,
               const IndexType storageSize,
               const IndexType virtualRows,
               const OffsetsView& rowPermArray,
               const OffsetsView& groupPointers )
: size( size ),
  storageSize( storageSize ),
  virtualRows( virtualRows ),
  rowPermArray( rowPermArray ),
  groupPointers( groupPointers )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__
BiEllpackView< Device, Index, Organization, WarpSize >::
BiEllpackView( const IndexType size,
               const IndexType storageSize,
               const IndexType virtualRows,
               const OffsetsView&& rowPermArray,
               const OffsetsView&& groupPointers )
: size( size ),
  storageSize( storageSize ),
  virtualRows( virtualRows ),
  rowPermArray( std::move( rowPermArray ) ),
  groupPointers( std::move( groupPointers ) )
{
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
String
BiEllpackView< Device, Index, Organization, WarpSize >::
getSerializationType()
{
   // FIXME: the serialized data DEPEND on the Organization and WarpSize parameters, so it should be reflected in the serialization type
   return "BiEllpack< [any_device], " + noa::TNL::getSerializationType< IndexType >() + " >";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
String
BiEllpackView< Device, Index, Organization, WarpSize >::
getSegmentsType()
{
   return "BiEllpack";
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__
typename BiEllpackView< Device, Index, Organization, WarpSize >::ViewType
BiEllpackView< Device, Index, Organization, WarpSize >::
getView()
{
   return ViewType( size, storageSize, virtualRows, rowPermArray.getView(), groupPointers.getView() );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, Organization, WarpSize >::
getConstView() const -> const ConstViewType
{
   BiEllpackView* this_ptr = const_cast< BiEllpackView* >( this );
   return ConstViewType( size,
                         storageSize,
                         virtualRows,
                         this_ptr->rowPermArray.getView(),
                         this_ptr->groupPointers.getView() );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, Organization, WarpSize >::
getSegmentsCount() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, Organization, WarpSize >::
getSegmentSize( const IndexType segmentIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSizeDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSize(
         rowPermArray,
         groupPointers,
         segmentIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentSizeDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, Organization, WarpSize >::
getSize() const -> IndexType
{
   return this->size;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, Organization, WarpSize >::
getStorageSize() const -> IndexType
{
   return this->storageSize;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__ auto BiEllpackView< Device, Index, Organization, WarpSize >::
getGlobalIndex( const Index segmentIdx, const Index localIdx ) const -> IndexType
{
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndexDirect(
         rowPermArray,
         groupPointers,
         segmentIdx,
         localIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndex(
         rowPermArray,
         groupPointers,
         segmentIdx,
         localIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getGlobalIndexDirect(
         rowPermArray,
         groupPointers,
         segmentIdx,
         localIdx );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
__cuda_callable__
auto
BiEllpackView< Device, Index, Organization, WarpSize >::
getSegmentView( const IndexType segmentIdx ) const -> SegmentViewType
{
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef __CUDA_ARCH__
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentViewDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
#else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentView(
         rowPermArray,
         groupPointers,
         segmentIdx );
#endif
   }
   else
      return detail::BiEllpack< IndexType, DeviceType, Organization, WarpSize >::getSegmentViewDirect(
         rowPermArray,
         groupPointers,
         segmentIdx );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
forElements( IndexType first, IndexType last, Function&& f ) const
{
   const auto segmentsPermutationView = this->rowPermArray.getConstView();
   const auto groupPointersView = this->groupPointers.getConstView();
   auto work = [=] __cuda_callable__ ( IndexType segmentIdx ) mutable {
      const IndexType strip = segmentIdx / getWarpSize();
      const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
      const IndexType rowStripPerm = segmentsPermutationView[ segmentIdx ] - strip * getWarpSize();
      const IndexType groupsCount = detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect( segmentsPermutationView, segmentIdx );
      IndexType groupHeight = getWarpSize();
      //printf( "segmentIdx = %d strip = %d firstGroupInStrip = %d rowStripPerm = %d groupsCount = %d \n", segmentIdx, strip, firstGroupInStrip, rowStripPerm, groupsCount );
      IndexType localIdx( 0 );
      for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount; groupIdx++ )
      {
         IndexType groupOffset = groupPointersView[ groupIdx ];
         const IndexType groupSize = groupPointersView[ groupIdx + 1 ] - groupOffset;
         //printf( "groupSize = %d \n", groupSize );
         if( groupSize )
         {
            const IndexType groupWidth = groupSize / groupHeight;
            for( IndexType i = 0; i < groupWidth; i++ )
            {
               if( Organization == RowMajorOrder )
               {
                  f( segmentIdx, localIdx, groupOffset + rowStripPerm * groupWidth + i );
               }
               else
               {
                  /*printf( "segmentIdx = %d localIdx = %d globalIdx = %d groupIdx = %d groupSize = %d groupWidth = %d\n",
                     segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight,
                     groupIdx, groupSize, groupWidth );*/
                  f( segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight );
               }
               localIdx++;
            }
         }
         groupHeight /= 2;
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last , work );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
forAllElements( Function&& f ) const
{
   this->forElements( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
forSegments( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = this->getConstView();
   auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) mutable {
      auto segment = view.getSegmentView( segmentIdx );
      function( segment );
   };
   noa::TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Function >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
forAllSegments( Function&& f ) const
{
   this->forSegments( 0, this->getSegmentsCount(), f );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
reduceSegments( IndexType first, IndexType last, Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   using RealType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   if( this->getStorageSize() == 0 )
      return;
   if( std::is_same< DeviceType, Devices::Host >::value )
      for( IndexType segmentIdx = 0; segmentIdx < this->getSize(); segmentIdx++ )
      {
         const IndexType stripIdx = segmentIdx / getWarpSize();
         const IndexType groupIdx = stripIdx * ( getLogWarpSize() + 1 );
         const IndexType inStripIdx = rowPermArray[ segmentIdx ] - stripIdx * getWarpSize();
         const IndexType groupsCount = detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCount( rowPermArray, segmentIdx );
         IndexType globalIdx = groupPointers[ groupIdx ];
         IndexType groupHeight = getWarpSize();
         IndexType localIdx( 0 );
         RealType aux( zero );
         bool compute( true );
         //std::cerr << "segmentIdx = " << segmentIdx
         //          << " stripIdx = " << stripIdx
         //          << " inStripIdx = " << inStripIdx
         //          << " groupIdx = " << groupIdx
         //         << " groupsCount = " << groupsCount
         //          << std::endl;
         for( IndexType group = 0; group < groupsCount && compute; group++ )
         {
            const IndexType groupSize = detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getGroupSize( groupPointers, stripIdx, group );
            IndexType groupWidth = groupSize / groupHeight;
            const IndexType globalIdxBack = globalIdx;
            //std::cerr << "  groupSize = " << groupSize
            //          << " groupWidth = " << groupWidth
            //          << std::endl;
            if( Organization == RowMajorOrder )
               globalIdx += inStripIdx * groupWidth;
            else
               globalIdx += inStripIdx;
            for( IndexType j = 0; j < groupWidth && compute; j++ )
            {
               //std::cerr << "    segmentIdx = " << segmentIdx << " groupIdx = " << groupIdx
               //         << " groupWidth = " << groupWidth << " groupHeight = " << groupHeight
               //          << " localIdx = " << localIdx << " globalIdx = " << globalIdx
               //          << " fetch = " << detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) << std::endl;
               aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
               if( Organization == RowMajorOrder )
                  globalIdx ++;
               else
                  globalIdx += groupHeight;
            }
            globalIdx = globalIdxBack + groupSize;
            groupHeight /= 2;
         }
         keeper( segmentIdx, aux );
      }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      constexpr int BlockDim = 256;
      dim3 cudaBlockSize = BlockDim;
      const IndexType stripsCount = roundUpDivision( last - first, getWarpSize() );
      const IndexType cudaBlocks = roundUpDivision( stripsCount * getWarpSize(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      IndexType sharedMemory = 0;
      if( Organization == ColumnMajorOrder )
         sharedMemory = cudaBlockSize.x * sizeof( RealType );

      //printStructure( std::cerr );
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         dim3 cudaGridSize = Cuda::getMaxGridSize();
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         detail::BiEllpackreduceSegmentsKernel< ViewType, IndexType, Fetch, Reduction, ResultKeeper, Real, BlockDim  >
            <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
            ( *this, gridIdx, first, last, fetch, reduction, keeper, zero );
      }
      cudaStreamSynchronize(0);
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Fetch, typename Reduction, typename ResultKeeper, typename Real >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
reduceAllSegments( Fetch& fetch, const Reduction& reduction, ResultKeeper& keeper, const Real& zero ) const
{
   this->reduceSegments( 0, this->getSegmentsCount(), fetch, reduction, keeper, zero );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
BiEllpackView< Device, Index, Organization, WarpSize >&
BiEllpackView< Device, Index, Organization, WarpSize >::
operator=( const BiEllpackView& source )
{
   this->size = source.size;
   this->storageSize = source.storageSize;
   this->virtualRows = source.virtualRows;
   this->rowPermArray.bind( source.rowPermArray );
   this->groupPointers.bind( source.groupPointers );
   return *this;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
save( File& file ) const
{
   file.save( &this->size );
   file.save( &this->storageSize );
   file.save( &this->virtualRows );
   file << this->rowPermArray
        << this->groupPointers;
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
      template< typename Fetch >
auto
BiEllpackView< Device, Index, Organization, WarpSize >::
print( Fetch&& fetch ) const -> SegmentsPrinter< BiEllpackView, Fetch >
{
   return SegmentsPrinter< BiEllpackView, Fetch >( *this, fetch );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
void
BiEllpackView< Device, Index, Organization, WarpSize >::
printStructure( std::ostream& str ) const
{
   const IndexType stripsCount = roundUpDivision( this->getSize(), getWarpSize() );
   for( IndexType stripIdx = 0; stripIdx < stripsCount; stripIdx++ )
   {
      str << "Strip: " << stripIdx << std::endl;
      const IndexType firstGroupIdx = stripIdx * ( getLogWarpSize() + 1 );
      const IndexType lastGroupIdx = firstGroupIdx + getLogWarpSize() + 1;
      IndexType groupHeight = getWarpSize();
      for( IndexType groupIdx = firstGroupIdx; groupIdx < lastGroupIdx; groupIdx ++ )
      {
         const IndexType groupSize = groupPointers.getElement( groupIdx + 1 ) - groupPointers.getElement( groupIdx );
         const IndexType groupWidth = groupSize / groupHeight;
         str << "\tGroup: " << groupIdx << " size = " << groupSize << " width = " << groupWidth << " height = " << groupHeight
             << " offset = " << groupPointers.getElement( groupIdx ) << std::endl;
         groupHeight /= 2;
      }
   }
}

#ifdef HAVE_CUDA
template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             int BlockDim >
__device__
void
BiEllpackView< Device, Index, Organization, WarpSize >::
reduceSegmentsKernelWithAllParameters( IndexType gridIdx,
                                          IndexType first,
                                          IndexType last,
                                          Fetch fetch,
                                          Reduction reduction,
                                          ResultKeeper keeper,
                                          Real zero ) const
{
   using RealType = decltype( fetch( IndexType(), IndexType(), IndexType(), std::declval< bool& >() ) );
   const IndexType segmentIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x + first;
   if( segmentIdx >= last )
      return;

   const IndexType strip = segmentIdx / getWarpSize();
   const IndexType firstGroupInStrip = strip * ( getLogWarpSize() + 1 );
   const IndexType rowStripPerm = rowPermArray[ segmentIdx ] - strip * getWarpSize();
   const IndexType groupsCount = detail::BiEllpack< IndexType, DeviceType, Organization, getWarpSize() >::getActiveGroupsCountDirect( rowPermArray, segmentIdx );
   IndexType groupHeight = getWarpSize();
   bool compute( true );
   IndexType localIdx( 0 );
   RealType result( zero );
   for( IndexType groupIdx = firstGroupInStrip; groupIdx < firstGroupInStrip + groupsCount && compute; groupIdx++ )
   {
      IndexType groupOffset = groupPointers[ groupIdx ];
      const IndexType groupSize = groupPointers[ groupIdx + 1 ] - groupOffset;
      if( groupSize )
      {
         const IndexType groupWidth = groupSize / groupHeight;
         for( IndexType i = 0; i < groupWidth; i++ )
         {
            if( Organization == RowMajorOrder )
               result = reduction( result, fetch( segmentIdx, localIdx, groupOffset + rowStripPerm * groupWidth + i, compute ) );
            else
               result = reduction( result, fetch( segmentIdx, localIdx, groupOffset + rowStripPerm + i * groupHeight, compute ) );
            localIdx++;
         }
      }
      groupHeight /= 2;
   }
   keeper( segmentIdx, result );
}

template< typename Device,
          typename Index,
          ElementsOrganization Organization,
          int WarpSize >
   template< typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Real,
             int BlockDim >
__device__
void
BiEllpackView< Device, Index, Organization, WarpSize >::
reduceSegmentsKernel( IndexType gridIdx,
                         IndexType first,
                         IndexType last,
                         Fetch fetch,
                         Reduction reduction,
                         ResultKeeper keeper,
                         Real zero ) const
{
   using RealType = decltype( fetch( IndexType(), std::declval< bool& >() ) );
   Index segmentIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x + first;

   const IndexType strip = segmentIdx >> getLogWarpSize();
   const IndexType warpStart = strip << getLogWarpSize();
   const IndexType inWarpIdx = segmentIdx & ( getWarpSize() - 1 );

   if( warpStart >= last )
      return;

   const int warpIdx = threadIdx.x / WarpSize;
   const int warpsCount = BlockDim / WarpSize;
   constexpr int groupsInStrip = 6; //getLogWarpSize() + 1;
   //IndexType firstGroupIdx = strip * groupsInStrip;
   IndexType firstGroupInBlock = 8 * ( strip / 8 ) * groupsInStrip;
   IndexType groupHeight = getWarpSize();

   /////
   // Allocate shared memory
   __shared__ RealType results[ BlockDim ];
   results[ threadIdx.x ] = zero;
   __shared__ IndexType sharedGroupPointers[ groupsInStrip * warpsCount + 1 ];

   /////
   // Fetch group pointers to shared memory
   //bool b1 = ( threadIdx.x <= warpsCount * groupsInStrip );
   //bool b2 = ( firstGroupIdx + threadIdx.x % groupsInStrip < this->groupPointers.getSize() );
   //printf( "tid = %d warpsCount * groupsInStrip = %d firstGroupIdx + threadIdx.x = %d this->groupPointers.getSize() = %d read = %d %d\n",
   //   threadIdx.x, warpsCount * groupsInStrip,
   //   firstGroupIdx + threadIdx.x,
   //   this->groupPointers.getSize(), ( int ) b1, ( int ) b2 );
   if( threadIdx.x <= warpsCount * groupsInStrip &&
      firstGroupInBlock + threadIdx.x < this->groupPointers.getSize() )
   {
      sharedGroupPointers[ threadIdx.x ] = this->groupPointers[ firstGroupInBlock + threadIdx.x ];
      //printf( " sharedGroupPointers[ %d ] = %d \n",
      //   threadIdx.x, sharedGroupPointers[ threadIdx.x ] );
   }
   const IndexType sharedGroupOffset = warpIdx * groupsInStrip;
   __syncthreads();

   /////
   // Perform the reduction
   bool compute( true );
   if( Organization == RowMajorOrder )
   {
      for( IndexType group = 0; group < getLogWarpSize() + 1; group++ )
      {
         IndexType groupBegin = sharedGroupPointers[ sharedGroupOffset + group ];
         IndexType groupEnd = sharedGroupPointers[ sharedGroupOffset + group + 1 ];
         TNL_ASSERT_LT( groupBegin, this->getStorageSize(), "" );
         //if( groupBegin >= this->getStorageSize() )
         //   printf( "tid = %d sharedGroupOffset + group + 1 = %d strip = %d group = %d groupBegin = %d groupEnd = %d this->getStorageSize() = %d\n",
         //      threadIdx.x, sharedGroupOffset + group + 1, strip, group, groupBegin, groupEnd, this->getStorageSize() );
         TNL_ASSERT_LT( groupEnd, this->getStorageSize(), "" );
         if( groupEnd - groupBegin > 0 )
         {
            if( inWarpIdx < groupHeight )
            {
               const IndexType groupWidth = ( groupEnd - groupBegin ) / groupHeight;
               IndexType globalIdx = groupBegin + inWarpIdx * groupWidth;
               for( IndexType i = 0; i < groupWidth && compute; i++ )
               {
                  TNL_ASSERT_LT( globalIdx, this->getStorageSize(), "" );
                  results[ threadIdx.x ] = reduction( results[ threadIdx.x ], fetch( globalIdx++, compute ) );
                  //if( strip == 1 )
                  //  printf( "tid = %d i = %d groupHeight = %d groupWidth = %d globalIdx = %d fetch = %f results = %f \n",
                  //      threadIdx.x, i,
                  //      groupHeight, groupWidth,
                  //      globalIdx, fetch( globalIdx, compute ), results[ threadIdx.x ] );
               }
            }
         }
         groupHeight >>= 1;
      }
   }
   else
   {
      RealType* temp = Cuda::getSharedMemory< RealType >();
      for( IndexType group = 0; group < getLogWarpSize() + 1; group++ )
      {
         IndexType groupBegin = sharedGroupPointers[ sharedGroupOffset + group ];
         IndexType groupEnd = sharedGroupPointers[ sharedGroupOffset + group + 1 ];
         //if( threadIdx.x < 36 && strip == 1 )
         //   printf( " tid = %d strip = %d group = %d groupBegin = %d groupEnd = %d \n", threadIdx.x, strip, group, groupBegin, groupEnd );
         if( groupEnd - groupBegin > 0 )
         {
            temp[ threadIdx.x ] = zero;
            IndexType globalIdx = groupBegin + inWarpIdx;
            while( globalIdx < groupEnd )
            {
               temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], fetch( globalIdx, compute ) );
               //if( strip == 1 )
               //   printf( "tid %d fetch %f temp %f \n", threadIdx.x, fetch( globalIdx, compute ), temp[ threadIdx.x ] );
               globalIdx += getWarpSize();
            }
            // TODO: reduction via templates
            /*IndexType bisection2 = getWarpSize();
            for( IndexType i = 0; i < group; i++ )
            {
               bisection2 >>= 1;
               if( inWarpIdx < bisection2 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + bisection2 ] );
            }*/

            __syncwarp();
            if( group > 0 && inWarpIdx < 16 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 16 ] );
            __syncwarp();
            if( group > 1 && inWarpIdx < 8 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 8 ] );
            __syncwarp();
            if( group > 2 && inWarpIdx < 4 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 4 ] );
            __syncwarp();
            if( group > 3 && inWarpIdx < 2 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 2 ] );
            __syncwarp();
            if( group > 4 && inWarpIdx < 1 )
                  temp[ threadIdx.x ] = reduction( temp[ threadIdx.x ], temp[ threadIdx.x + 1 ] );
            __syncwarp();

            if( inWarpIdx < groupHeight )
               results[ threadIdx.x ] = reduction( results[ threadIdx.x ], temp[ threadIdx.x ] );
         }
         groupHeight >>= 1;
      }
   }
   __syncthreads();
   if( warpStart + inWarpIdx >= last )
      return;

   /////
   // Store the results
   //if( strip == 1 )
   //   printf( "Adding %f at %d \n", results[ this->rowPermArray[ warpStart + inWarpIdx ] & ( blockDim.x - 1 ) ], warpStart + inWarpIdx );
   keeper( warpStart + inWarpIdx, results[ this->rowPermArray[ warpStart + inWarpIdx ] & ( blockDim.x - 1 ) ] );
}
#endif

      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
