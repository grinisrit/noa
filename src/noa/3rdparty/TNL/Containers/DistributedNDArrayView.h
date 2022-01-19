// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Containers/NDArrayView.h>
#include <TNL/Containers/Subrange.h>
#include <TNL/MPI/Wrappers.h>

namespace TNL {
namespace Containers {

template< typename NDArrayView >
class DistributedNDArrayView
{
public:
   using ValueType = typename NDArrayView::ValueType;
   using DeviceType = typename NDArrayView::DeviceType;
   using IndexType = typename NDArrayView::IndexType;
   using SizesHolderType = typename NDArrayView::SizesHolderType;
   using PermutationType = typename NDArrayView::PermutationType;
   using LocalBeginsType = __ndarray_impl::LocalBeginsHolder< typename NDArrayView::SizesHolderType >;
   using LocalRangeType = Subrange< IndexType >;
   using OverlapsType = typename NDArrayView::OverlapsType;

   using ViewType = DistributedNDArrayView< NDArrayView >;
   using ConstViewType = DistributedNDArrayView< typename NDArrayView::ConstViewType >;
   using LocalViewType = NDArrayView;
   using ConstLocalViewType = typename NDArrayView::ConstViewType;

   DistributedNDArrayView() = default;

   // explicit initialization by local array view, global sizes and local begins and ends
   DistributedNDArrayView( NDArrayView localView, SizesHolderType globalSizes, LocalBeginsType localBegins, SizesHolderType localEnds, MPI_Comm communicator )
   : localView(localView), communicator(communicator), globalSizes(globalSizes), localBegins(localBegins), localEnds(localEnds) {}

   // copy-constructor does shallow copy
   DistributedNDArrayView( const DistributedNDArrayView& ) = default;

   // default move-constructor
   DistributedNDArrayView( DistributedNDArrayView&& ) = default;

   // Copy-assignment does deep copy, just like regular array, but the sizes
   // must match (i.e. copy-assignment cannot resize).
   DistributedNDArrayView& operator=( const DistributedNDArrayView& other ) = default;

   // There is no move-assignment operator, so expressions like `a = b.getView()`
   // are resolved as copy-assignment.

   // Templated copy-assignment
   template< typename OtherArray >
   DistributedNDArrayView& operator=( const OtherArray& other )
   {
      globalSizes = other.getSizes();
      localBegins = other.getLocalBegins();
      localEnds = other.getLocalEnds();
      communicator = other.getCommunicator();
      localView = other.getConstLocalView();
      return *this;
   }

   // methods for rebinding (reinitialization)
   void bind( DistributedNDArrayView view )
   {
      localView.bind( view.localView );
      communicator = view.communicator;
      globalSizes = view.globalSizes;
      localBegins = view.localBegins;
      localEnds = view.localEnds;
   }

   // binds to the given raw pointer and changes the indexer
   void bind( ValueType* data, typename LocalViewType::IndexerType indexer )
   {
      localView.bind( data, indexer );
   }

   // binds to the given raw pointer and preserves the current indexer
   void bind( ValueType* data )
   {
      localView.bind( data );
   }

   void reset()
   {
      localView.reset();
      communicator = MPI_COMM_NULL;
      globalSizes = SizesHolderType{};
      localBegins = LocalBeginsType{};
      localEnds = SizesHolderType{};
   }

   static constexpr std::size_t getDimension()
   {
      return NDArrayView::getDimension();
   }

   MPI_Comm getCommunicator() const
   {
      return communicator;
   }

   // Returns the *global* sizes
   const SizesHolderType& getSizes() const
   {
      return globalSizes;
   }

   // Returns the *global* size
   template< std::size_t level >
   IndexType getSize() const
   {
      return globalSizes.template getSize< level >();
   }

   LocalBeginsType getLocalBegins() const
   {
      return localBegins;
   }

   SizesHolderType getLocalEnds() const
   {
      return localEnds;
   }

   template< std::size_t level >
   LocalRangeType getLocalRange() const
   {
      return LocalRangeType( localBegins.template getSize< level >(), localEnds.template getSize< level >() );
   }

   // returns the local storage size
   IndexType getLocalStorageSize() const
   {
      return localView.getStorageSize();
   }

   LocalViewType getLocalView()
   {
      return localView;
   }

   ConstLocalViewType getConstLocalView() const
   {
      return localView.getConstView();
   }

   // returns the *local* storage index for given *global* indices
   template< typename... IndexTypes >
   IndexType
   getStorageIndex( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == SizesHolderType::getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      auto getStorageIndex = [this]( auto&&... indices )
      {
         return this->localView.getStorageIndex( std::forward< decltype(indices) >( indices )... );
      };
      return __ndarray_impl::host_call_with_unshifted_indices< LocalBeginsType >( localBegins, getStorageIndex, std::forward< IndexTypes >( indices )... );
   }

   ValueType* getData()
   {
      return localView.getData();
   }

   std::add_const_t< ValueType >* getData() const
   {
      return localView.getData();
   }


   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      return __ndarray_impl::call_with_unshifted_indices< LocalBeginsType >( localBegins, localView, std::forward< IndexTypes >( indices )... );
   }

   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      return __ndarray_impl::call_with_unshifted_indices< LocalBeginsType >( localBegins, localView, std::forward< IndexTypes >( indices )... );
   }

   // bracket operator for 1D arrays
   __cuda_callable__
   ValueType&
   operator[]( IndexType index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexType >( index ) );
      return localView[ index - localBegins.template getSize< 0 >() ];
   }

   __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexType >( index ) );
      return localView[ index - localBegins.template getSize< 0 >() ];
   }

   ViewType getView()
   {
      return ViewType( *this );
   }

   ConstViewType getConstView() const
   {
      return ConstViewType( localView, globalSizes, localBegins, localEnds, communicator );
   }

   // TODO: overlaps should be skipped, otherwise it works only after synchronization
   bool operator==( const DistributedNDArrayView& other ) const
   {
      // we can't run allreduce if the communicators are different
      if( communicator != other.getCommunicator() )
         return false;
      const bool localResult =
            globalSizes == other.globalSizes &&
            localBegins == other.localBegins &&
            localEnds == other.localEnds &&
            localView == other.localView;
      bool result = true;
      if( communicator != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, communicator );
      return result;
   }

   bool operator!=( const DistributedNDArrayView& other ) const
   {
      return ! (*this == other);
   }

   // iterate over all local elements
   template< typename Device2 = DeviceType, typename Func >
   void forAll( Func f ) const
   {
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, localEnds, f );
   }

   // iterate over local elements which are not neighbours of *global* boundaries
   template< typename Device2 = DeviceType, typename Func >
   void forInternal( Func f ) const
   {
      // add static sizes
      using Begins = __ndarray_impl::LocalBeginsHolder< SizesHolderType, 1 >;
      // add dynamic sizes
      Begins begins;
      __ndarray_impl::SetSizesAddHelper< 1, Begins, SizesHolderType, OverlapsType >::add( begins, SizesHolderType{} );
      __ndarray_impl::SetSizesMaxHelper< Begins, LocalBeginsType >::max( begins, localBegins );

      // subtract static sizes
      using Ends = typename __ndarray_impl::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      Ends ends;
      __ndarray_impl::SetSizesSubtractHelper< 1, Ends, SizesHolderType, OverlapsType >::subtract( ends, globalSizes );
      __ndarray_impl::SetSizesMinHelper< Ends, SizesHolderType >::min( ends, localEnds );

      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

   // iterate over local elements inside the given [begins, ends) range specified by global indices
   template< typename Device2 = DeviceType, typename Func, typename Begins, typename Ends >
   void forInternal( Func f, const Begins& begins, const Ends& ends ) const
   {
      // TODO: assert "localBegins <= begins <= localEnds", "localBegins <= ends <= localEnds"
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

   // iterate over local elements which are neighbours of *global* boundaries
   template< typename Device2 = DeviceType, typename Func >
   void forBoundary( Func f ) const
   {
      // add static sizes
      using SkipBegins = __ndarray_impl::LocalBeginsHolder< SizesHolderType, 1 >;
      // add dynamic sizes
      SkipBegins skipBegins;
      __ndarray_impl::SetSizesAddHelper< 1, SkipBegins, SizesHolderType, OverlapsType >::add( skipBegins, SizesHolderType{} );
      __ndarray_impl::SetSizesMaxHelper< SkipBegins, LocalBeginsType >::max( skipBegins, localBegins );

      // subtract static sizes
      using SkipEnds = typename __ndarray_impl::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      SkipEnds skipEnds;
      __ndarray_impl::SetSizesSubtractHelper< 1, SkipEnds, SizesHolderType, OverlapsType >::subtract( skipEnds, globalSizes );
      __ndarray_impl::SetSizesMinHelper< SkipEnds, SizesHolderType >::min( skipEnds, localEnds );

      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, skipBegins, skipEnds, localEnds, f );
   }

   // iterate over local elements outside the given [skipBegins, skipEnds) range specified by global indices
   template< typename Device2 = DeviceType, typename Func, typename SkipBegins, typename SkipEnds >
   void forBoundary( Func f, const SkipBegins& skipBegins, const SkipEnds& skipEnds ) const
   {
      // TODO: assert "localBegins <= skipBegins <= localEnds", "localBegins <= skipEnds <= localEnds"
      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, skipBegins, skipEnds, localEnds, f );
   }

   // iterate over local elements which are not neighbours of overlaps (if all overlaps are 0, it is equivalent to forAll)
   template< typename Device2 = DeviceType, typename Func >
   void forLocalInternal( Func f ) const
   {
      // add overlaps to dynamic sizes
      LocalBeginsType begins;
      __ndarray_impl::SetSizesAddOverlapsHelper< LocalBeginsType, SizesHolderType, OverlapsType >::add( begins, localBegins );

      // subtract overlaps from dynamic sizes
      SizesHolderType ends;
      __ndarray_impl::SetSizesSubtractOverlapsHelper< SizesHolderType, SizesHolderType, OverlapsType >::subtract( ends, localEnds );

      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

   // iterate over local elements which are neighbours of overlaps (if all overlaps are 0, it has no effect)
   template< typename Device2 = DeviceType, typename Func >
   void forLocalBoundary( Func f ) const
   {
      // add overlaps to dynamic sizes
      LocalBeginsType skipBegins;
      __ndarray_impl::SetSizesAddOverlapsHelper< LocalBeginsType, SizesHolderType, OverlapsType >::add( skipBegins, localBegins );

      // subtract overlaps from dynamic sizes
      SizesHolderType skipEnds;
      __ndarray_impl::SetSizesSubtractOverlapsHelper< SizesHolderType, SizesHolderType, OverlapsType >::subtract( skipEnds, localEnds );

      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, skipBegins, skipEnds, localEnds, f );
   }

   // iterate over elements of overlaps (if all overlaps are 0, it has no effect)
   template< typename Device2 = DeviceType, typename Func >
   void forOverlaps( Func f ) const
   {
      // subtract overlaps from dynamic sizes
      LocalBeginsType begins;
      __ndarray_impl::SetSizesSubtractOverlapsHelper< LocalBeginsType, SizesHolderType, OverlapsType >::subtract( begins, localBegins );

      // add overlaps to dynamic sizes
      SizesHolderType ends;
      __ndarray_impl::SetSizesAddOverlapsHelper< SizesHolderType, SizesHolderType, OverlapsType >::add( ends, localEnds );

      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, localBegins, localEnds, ends, f );
   }

protected:
   NDArrayView localView;
   MPI_Comm communicator = MPI_COMM_NULL;
   SizesHolderType globalSizes;
   // static sizes should have different type: localBegin is always 0, localEnd is always the full size
   LocalBeginsType localBegins;
   SizesHolderType localEnds;
};

} // namespace Containers
} // namespace TNL
