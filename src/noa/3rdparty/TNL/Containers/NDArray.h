// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Containers/Array.h>
#include <noa/3rdparty/TNL/Containers/StaticArray.h>

#include <noa/3rdparty/TNL/Containers/NDArrayView.h>

namespace noaTNL {
namespace Containers {

template< std::size_t slicedDimension = 0,
          std::size_t sliceSize = 0 >
struct SliceInfo
{
   // sliceSize == 0 means no slicing
   static constexpr std::size_t getSliceSize( std::size_t dimension )
   {
      return (dimension == slicedDimension) ? sliceSize : 0;
   }
};




template< typename Array,
          typename Indexer,
          typename Device = typename Array::DeviceType >
class NDArrayStorage
: public Indexer
{
public:
   using StorageArray = Array;
   using ValueType = typename Array::ValueType;
   using DeviceType = Device;
   using IndexType = typename Indexer::IndexType;
   using SizesHolderType = typename Indexer::SizesHolderType;
   using PermutationType = typename Indexer::PermutationType;
   using OverlapsType = typename Indexer::OverlapsType;
   using IndexerType = Indexer;
   using ViewType = NDArrayView< ValueType, DeviceType, IndexerType >;
   using ConstViewType = NDArrayView< std::add_const_t< ValueType >, DeviceType, IndexerType >;

   // all methods from NDArrayView

   NDArrayStorage() = default;

   // Copy constructor (makes a deep copy).
   explicit NDArrayStorage( const NDArrayStorage& ) = default;

   // Standard copy-semantics with deep copy, just like regular 1D array.
   // Mismatched sizes cause reallocations.
   NDArrayStorage& operator=( const NDArrayStorage& other ) = default;

   // default move-semantics
   NDArrayStorage( NDArrayStorage&& ) = default;
   NDArrayStorage& operator=( NDArrayStorage&& ) = default;

   // Templated copy-assignment
   template< typename OtherArray >
   NDArrayStorage& operator=( const OtherArray& other )
   {
      static_assert( std::is_same< PermutationType, typename OtherArray::PermutationType >::value,
                     "Arrays must have the same permutation of indices." );
      // update sizes
      __ndarray_impl::SetSizesCopyHelper< SizesHolderType, typename OtherArray::SizesHolderType >::copy( getSizes(), other.getSizes() );
      // (re)allocate storage if necessary
      array.setSize( getStorageSize() );
      // copy data
      getView() = other.getConstView();
      return *this;
   }

   bool operator==( const NDArrayStorage& other ) const
   {
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return getSizes() == other.getSizes() && array == other.array;
   }

   bool operator!=( const NDArrayStorage& other ) const
   {
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return getSizes() != other.getSizes() || array != other.array;
   }

   __cuda_callable__
   ValueType* getData()
   {
      return array.getData();
   }

   __cuda_callable__
   std::add_const_t< ValueType >* getData() const
   {
      return array.getData();
   }

   // methods from the base class
   using IndexerType::getDimension;
   using IndexerType::getSizes;
   using IndexerType::getSize;
   using IndexerType::getStride;
   using IndexerType::getOverlap;
   using IndexerType::getStorageSize;
   using IndexerType::getStorageIndex;

   __cuda_callable__
   const IndexerType& getIndexer() const
   {
      return *this;
   }

   __cuda_callable__
   ViewType getView()
   {
      return ViewType( array.getData(), getSizes() );
   }

   __cuda_callable__
   ConstViewType getConstView() const
   {
      return ConstViewType( array.getData(), getSizes() );
   }

   template< std::size_t... Dimensions, typename... IndexTypes >
   __cuda_callable__
   auto getSubarrayView( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      static_assert( 0 < sizeof...(Dimensions) && sizeof...(Dimensions) <= getDimension(), "got wrong number of dimensions" );
// FIXME: nvcc chokes on the variadic brace-initialization
#ifndef __NVCC__
      static_assert( __ndarray_impl::all_elements_in_range( 0, PermutationType::size(), {Dimensions...} ),
                     "invalid dimensions" );
      static_assert( __ndarray_impl::is_increasing_sequence( {Dimensions...} ),
                     "specifying permuted dimensions is not supported" );
#endif

      using Getter = __ndarray_impl::SubarrayGetter< typename Indexer::NDBaseType, PermutationType, Dimensions... >;
      using Subpermutation = typename Getter::Subpermutation;
      auto& begin = operator()( std::forward< IndexTypes >( indices )... );
      auto subarray_sizes = Getter::filterSizes( getSizes(), std::forward< IndexTypes >( indices )... );
      auto strides = Getter::getStrides( getSizes(), std::forward< IndexTypes >( indices )... );
      static_assert( Subpermutation::size() == sizeof...(Dimensions), "Bug - wrong subpermutation length." );
      static_assert( decltype(subarray_sizes)::getDimension() == sizeof...(Dimensions), "Bug - wrong dimension of the new sizes." );
      static_assert( decltype(strides)::getDimension() == sizeof...(Dimensions), "Bug - wrong dimension of the strides." );
      // TODO: select overlaps for the subarray
      using Subindexer = NDArrayIndexer< decltype(subarray_sizes), Subpermutation, typename Indexer::NDBaseType, decltype(strides) >;
      using SubarrayView = NDArrayView< ValueType, Device, Subindexer >;
      return SubarrayView{ &begin, subarray_sizes, strides };
   }

   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   // bracket operator for 1D arrays
   __cuda_callable__
   ValueType&
   operator[]( IndexType index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInBounds( getSizes(), OverlapsType{}, std::forward< IndexType >( index ) );
      return array[ index ];
   }

   __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      __ndarray_impl::assertIndicesInBounds( getSizes(), OverlapsType{}, std::forward< IndexType >( index ) );
      return array[ index ];
   }

   template< typename Device2 = DeviceType, typename Func >
   void forAll( Func f ) const
   {
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      dispatch( Begins{}, getSizes(), f );
   }

   template< typename Device2 = DeviceType, typename Func >
   void forInternal( Func f ) const
   {
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using Ends = typename __ndarray_impl::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      Ends ends;
      __ndarray_impl::SetSizesSubtractHelper< 1, Ends, SizesHolderType >::subtract( ends, getSizes() );
      dispatch( Begins{}, ends, f );
   }

   template< typename Device2 = DeviceType, typename Func, typename Begins, typename Ends >
   void forInternal( Func f, const Begins& begins, const Ends& ends ) const
   {
      // TODO: assert "begins <= sizes", "ends <= sizes"
      __ndarray_impl::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, f );
   }

   template< typename Device2 = DeviceType, typename Func >
   void forBoundary( Func f ) const
   {
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      using SkipBegins = ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using SkipEnds = typename __ndarray_impl::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      SkipEnds skipEnds;
      __ndarray_impl::SetSizesSubtractHelper< 1, SkipEnds, SizesHolderType >::subtract( skipEnds, getSizes() );

      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, SkipBegins{}, skipEnds, getSizes(), f );
   }

   template< typename Device2 = DeviceType, typename Func, typename SkipBegins, typename SkipEnds >
   void forBoundary( Func f, const SkipBegins& skipBegins, const SkipEnds& skipEnds ) const
   {
      // TODO: assert "skipBegins <= sizes", "skipEnds <= sizes"
      using Begins = ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      __ndarray_impl::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, skipBegins, skipEnds, getSizes(), f );
   }


   // extra methods

   // TODO: rename to setSizes and make sure that overloading with the following method works
   void setSize( const SizesHolderType& sizes )
   {
      getSizes() = sizes;
      array.setSize( getStorageSize() );
   }

   template< typename... IndexTypes >
   void setSizes( IndexTypes&&... sizes )
   {
      static_assert( sizeof...( sizes ) == getDimension(), "got wrong number of sizes" );
      __ndarray_impl::setSizesHelper( getSizes(), std::forward< IndexTypes >( sizes )... );
      array.setSize( getStorageSize() );
   }

   void setLike( const NDArrayStorage& other )
   {
      getSizes() = other.getSizes();
      array.setSize( getStorageSize() );
   }

   void reset()
   {
      getSizes() = SizesHolderType{};
      TNL_ASSERT_EQ( getStorageSize(), 0, "Failed to reset the sizes." );
      array.reset();
   }

   // "safe" accessor - will do slow copy from device
   template< typename... IndexTypes >
   ValueType
   getElement( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      return array.getElement( getStorageIndex( std::forward< IndexTypes >( indices )... ) );
   }

   const StorageArray& getStorageArray() const
   {
      return array;
   }

   StorageArray& getStorageArray()
   {
      return array;
   }

   void setValue( ValueType value )
   {
      array.setValue( value );
   }

protected:
   StorageArray array;
   IndexerType indexer;
};

template< typename Value,
          typename SizesHolder,
          typename Permutation = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename Device = Devices::Host,
          typename Index = typename SizesHolder::IndexType,
          typename Overlaps = __ndarray_impl::make_constant_index_sequence< SizesHolder::getDimension(), 0 >,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Value > >
class NDArray
: public NDArrayStorage< Array< Value, Device, Index, Allocator >,
                         NDArrayIndexer< SizesHolder,
                                         Permutation,
                                         __ndarray_impl::NDArrayBase< SliceInfo< 0, 0 > >,
                                         __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() >,
                                         Overlaps > >
{
   using Base = NDArrayStorage< Array< Value, Device, Index, Allocator >,
                                NDArrayIndexer< SizesHolder,
                                                Permutation,
                                                __ndarray_impl::NDArrayBase< SliceInfo< 0, 0 > >,
                                                __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() >,
                                                Overlaps > >;

public:
   // inherit all constructors and assignment operators
   using Base::Base;
   using Base::operator=;

   // default constructor
   NDArray() = default;

   // implement dynamic array interface
   using AllocatorType = Allocator;

   NDArray( const AllocatorType& allocator )
   {
      // set empty array containing the specified allocator
      this->getStorageArray() = Array< Value, Device, Index, Allocator >( allocator );
   }

   // Copy constructor with a specific allocator (makes a deep copy).
   explicit NDArray( const NDArray& other, const AllocatorType& allocator )
   {
      // set empty array containing the specified allocator
      this->array = Array< Value, Device, Index, Allocator >( allocator );
      // copy the data
      *this = other;
   }

   AllocatorType getAllocator() const
   {
      return this->array.getAllocator();
   }
};

template< typename Value,
          typename SizesHolder,
          typename Permutation = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename Index = typename SizesHolder::IndexType >
class StaticNDArray
: public NDArrayStorage< StaticArray< __ndarray_impl::StaticStorageSizeGetter< SizesHolder >::get(), Value >,
                         NDArrayIndexer< SizesHolder,
                                         Permutation,
                                         __ndarray_impl::NDArrayBase< SliceInfo< 0, 0 > > >,
                         Devices::Sequential >
{
   using Base = NDArrayStorage< StaticArray< __ndarray_impl::StaticStorageSizeGetter< SizesHolder >::get(), Value >,
                                NDArrayIndexer< SizesHolder,
                                                Permutation,
                                                __ndarray_impl::NDArrayBase< SliceInfo< 0, 0 > > >,
                                Devices::Sequential >;
   static_assert( __ndarray_impl::StaticStorageSizeGetter< SizesHolder >::get() > 0,
                  "All dimensions of a static array must to be positive." );

public:
   // inherit all assignment operators
   using Base::operator=;
};

template< typename Value,
          typename SizesHolder,
          typename Permutation = std::make_index_sequence< SizesHolder::getDimension() >,  // identity by default
          typename SliceInfo = SliceInfo<>,  // no slicing by default
          typename Device = Devices::Host,
          typename Index = typename SizesHolder::IndexType,
          typename Overlaps = __ndarray_impl::make_constant_index_sequence< SizesHolder::getDimension(), 0 >,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Value > >
class SlicedNDArray
: public NDArrayStorage< Array< Value, Device, Index, Allocator >,
                         NDArrayIndexer< SizesHolder,
                                         Permutation,
                                         __ndarray_impl::SlicedNDArrayBase< SliceInfo >,
                                         __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() >,
                                         Overlaps > >
{
   using Base = NDArrayStorage< Array< Value, Device, Index, Allocator >,
                                NDArrayIndexer< SizesHolder,
                                                Permutation,
                                                __ndarray_impl::SlicedNDArrayBase< SliceInfo >,
                                                __ndarray_impl::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() >,
                                                Overlaps > >;

public:
   // inherit all constructors and assignment operators
   using Base::Base;
   using Base::operator=;

   // default constructor
   SlicedNDArray() = default;

   // implement dynamic array interface
   using AllocatorType = Allocator;

   SlicedNDArray( const AllocatorType& allocator )
   {
      // set empty array containing the specified allocator
      this->getStorageArray() = Array< Value, Device, Index, Allocator >( allocator );
   }

   // Copy constructor with a specific allocator (makes a deep copy).
   explicit SlicedNDArray( const SlicedNDArray& other, const AllocatorType& allocator )
   {
      // set empty array containing the specified allocator
      this->array = Array< Value, Device, Index, Allocator >( allocator );
      // copy the data
      *this = other;
   }

   AllocatorType getAllocator() const
   {
      return this->array.getAllocator();
   }
};

} // namespace Containers
} // namespace noaTNL
