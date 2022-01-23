// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Containers/NDArray.h>
#include <noa/3rdparty/TNL/Containers/DistributedNDArrayView.h>

namespace noa::TNL {
namespace Containers {

template< typename NDArray >
class DistributedNDArray
{
public:
   using ValueType = typename NDArray::ValueType;
   using DeviceType = typename NDArray::DeviceType;
   using IndexType = typename NDArray::IndexType;
   using AllocatorType = typename NDArray::AllocatorType;
   using SizesHolderType = typename NDArray::SizesHolderType;
   using PermutationType = typename NDArray::PermutationType;
   using LocalBeginsType = __ndarray_impl::LocalBeginsHolder< typename NDArray::SizesHolderType >;
   using LocalRangeType = Subrange< IndexType >;
   using OverlapsType = typename NDArray::OverlapsType;

   using ViewType = DistributedNDArrayView< typename NDArray::ViewType >;
   using ConstViewType = DistributedNDArrayView< typename NDArray::ConstViewType >;
   using LocalViewType = typename NDArray::ViewType;
   using ConstLocalViewType = typename NDArray::ConstViewType;

   // all methods from NDArrayView

   DistributedNDArray() = default;

   DistributedNDArray( const AllocatorType& allocator );

   // Copy constructor (makes a deep copy).
   explicit DistributedNDArray( const DistributedNDArray& ) = default;

   // Copy constructor with a specific allocator (makes a deep copy).
   explicit DistributedNDArray( const DistributedNDArray& other, const AllocatorType& allocator )
   : localArray( allocator )
   {
      *this = other;
   }

   // Standard copy-semantics with deep copy, just like regular 1D array.
   // Mismatched sizes cause reallocations.
   DistributedNDArray& operator=( const DistributedNDArray& other ) = default;

   // default move-semantics
   DistributedNDArray( DistributedNDArray&& ) = default;
   DistributedNDArray& operator=( DistributedNDArray&& ) = default;

   // Templated copy-assignment
   template< typename OtherArray >
   DistributedNDArray& operator=( const OtherArray& other )
   {
      globalSizes = other.getSizes();
      localBegins = other.getLocalBegins();
      localEnds = other.getLocalEnds();
      communicator = other.getCommunicator();
      localArray = other.getConstLocalView();
      return *this;
   }

   static constexpr std::size_t getDimension()
   {
      return NDArray::getDimension();
   }

   AllocatorType getAllocator() const
   {
      return localArray.getAllocator();
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
      return localArray.getStorageSize();
   }

   LocalViewType getLocalView()
   {
      return localArray.getView();
   }

   ConstLocalViewType getConstLocalView() const
   {
      return localArray.getConstView();
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
         return this->localArray.getStorageIndex( std::forward< decltype(indices) >( indices )... );
      };
      return __ndarray_impl::host_call_with_unshifted_indices< LocalBeginsType >( localBegins, getStorageIndex, std::forward< IndexTypes >( indices )... );
   }

   ValueType* getData()
   {
      return localArray.getData();
   }

   std::add_const_t< ValueType >* getData() const
   {
      return localArray.getData();
   }


   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      return __ndarray_impl::call_with_unshifted_indices< LocalBeginsType >( localBegins, localArray, std::forward< IndexTypes >( indices )... );
   }

   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      return __ndarray_impl::call_with_unshifted_indices< LocalBeginsType >( localBegins, localArray, std::forward< IndexTypes >( indices )... );
   }

   // bracket operator for 1D arrays
   __cuda_callable__
   ValueType&
   operator[]( IndexType index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexType >( index ) );
      return localArray[ index - localBegins.template getSize< 0 >() ];
   }

   __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexType >( index ) );
      return localArray[ index - localBegins.template getSize< 0 >() ];
   }

   ViewType getView()
   {
      return ViewType( localArray.getView(), globalSizes, localBegins, localEnds, communicator );
   }

   ConstViewType getConstView() const
   {
      return ConstViewType( localArray.getConstView(), globalSizes, localBegins, localEnds, communicator );
   }

   // TODO: overlaps should be skipped, otherwise it works only after synchronization
   bool operator==( const DistributedNDArray& other ) const
   {
      // we can't run allreduce if the communicators are different
      if( communicator != other.getCommunicator() )
         return false;
      const bool localResult =
            globalSizes == other.globalSizes &&
            localBegins == other.localBegins &&
            localEnds == other.localEnds &&
            localArray == other.localArray;
      bool result = true;
      if( communicator != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, communicator );
      return result;
   }

   bool operator!=( const DistributedNDArray& other ) const
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


   // extra methods

   // Sets the *global* size, but does not allocate storage
   template< typename... IndexTypes >
   void setSizes( IndexTypes&&... sizes )
   {
      static_assert( sizeof...( sizes ) == getDimension(), "got wrong number of sizes" );
      __ndarray_impl::setSizesHelper( globalSizes, std::forward< IndexTypes >( sizes )... );
      // initialize localBegins and localEnds
      localBegins = LocalBeginsType{};
      localEnds = globalSizes;
   }

   template< std::size_t level >
   void setDistribution( IndexType begin, IndexType end, MPI_Comm communicator = MPI_COMM_WORLD )
   {
      static_assert( SizesHolderType::template getStaticSize< level >() == 0, "NDArray cannot be distributed in static dimensions." );
      TNL_ASSERT_GE( begin, 0, "begin must be non-negative" );
      TNL_ASSERT_LE( end, globalSizes.template getSize< level >(), "end must not be greater than global size" );
      TNL_ASSERT_LT( begin, end, "begin must be lesser than end" );
      localBegins.template setSize< level >( begin );
      localEnds.template setSize< level >( end );
      TNL_ASSERT( this->communicator == MPI_COMM_NULL || this->communicator == communicator,
                  std::cerr << "different communicators cannot be combined for different dimensions" );
      this->communicator = communicator;
   }

   // Computes the distributed storage size and allocates the local array
   void allocate()
   {
      SizesHolderType localSizes;
      Algorithms::staticFor< std::size_t, 0, SizesHolderType::getDimension() >(
         [&] ( auto level ) {
            if( SizesHolderType::template getStaticSize< level >() != 0 )
               return;

            const auto begin = localBegins.template getSize< level >();
            const auto end = localEnds.template getSize< level >();
            if( begin == end )
               localSizes.template setSize< level >( globalSizes.template getSize< level >() );
            else {
               TNL_ASSERT_GE( end - begin, (decltype(end)) __ndarray_impl::get<level>( OverlapsType{} ), "local size is less than the size of overlaps" );
               //localSizes.template setSize< level >( end - begin + 2 * __ndarray_impl::get<level>( OverlapsType{} ) );
               localSizes.template setSize< level >( end - begin );
            }
         }
      );
      localArray.setSize( localSizes );
   }

   void setLike( const DistributedNDArray& other )
   {
      localArray.setLike( other.localArray );
      communicator = other.getCommunicator();
      globalSizes = other.getSizes();
      localBegins = other.localBegins;
      localEnds = other.localEnds;
   }

   void reset()
   {
      localArray.reset();
      communicator = MPI_COMM_NULL;
      globalSizes = SizesHolderType{};
      localBegins = LocalBeginsType{};
      localEnds = SizesHolderType{};
   }

   // "safe" accessor - will do slow copy from device
   template< typename... IndexTypes >
   ValueType
   getElement( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      __ndarray_impl::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      auto getElement = [this]( auto&&... indices )
      {
         return this->localArray.getElement( std::forward< decltype(indices) >( indices )... );
      };
      return __ndarray_impl::host_call_with_unshifted_indices< LocalBeginsType >( localBegins, getElement, std::forward< IndexTypes >( indices )... );
   }

   void setValue( ValueType value )
   {
      localArray.setValue( value );
   }

protected:
   NDArray localArray;
   MPI_Comm communicator = MPI_COMM_NULL;
   SizesHolderType globalSizes;
   // static sizes should have different type: localBegin is always 0, localEnd is always the full size
   LocalBeginsType localBegins;
   SizesHolderType localEnds;
};

} // namespace Containers
} // namespace noa::TNL
