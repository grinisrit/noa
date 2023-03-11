// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/NDArrayIndexer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/SizesHolder.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/Subarrays.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/Executors.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/BoundaryExecutors.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/Operations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MemoryOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MultiDeviceMemoryOperations.h>

namespace noa::TNL {
namespace Containers {

/**
 * \brief Simple data structure which provides a non-owning encapsulation of
 * N-dimensional array data.
 *
 * \tparam Value Type of the values stored in the array.
 * \tparam Device Type of the \ref TNL::Devices "device" that will be used for
 *                running operations on the array.
 * \tparam Type of the N-dimensional indexer, \ref NDArrayIndexer.
 *
 * \ingroup ndarray
 */
template< typename Value, typename Device, typename Indexer >
class NDArrayView : public Indexer
{
public:
   //! \brief Type of the values stored in the array.
   using ValueType = Value;

   //! \brief Type of the \ref TNL::Devices "device" used for running operations on the array.
   using DeviceType = Device;

   //! \brief Type of indices used for addressing the array elements.
   using IndexType = typename Indexer::IndexType;

   //! \brief Type of the underlying object which represents the sizes of the N-dimensional array.
   using SizesHolderType = typename Indexer::SizesHolderType;

   //! \brief Type of the base class which represents the strides of the N-dimensional array.
   using StridesHolderType = typename Indexer::StridesHolderType;

   //! \brief Permutation that is applied to indices when accessing the array elements.
   using PermutationType = typename Indexer::PermutationType;

   //! \brief Sequence of integers representing the overlaps in each dimension
   //! of a distributed N-dimensional array.
   using OverlapsType = typename Indexer::OverlapsType;

   //! \brief Type of the N-dimensional indexer, \ref NDArrayIndexer.
   using IndexerType = Indexer;

   //! Compatible \ref NDArrayView type.
   using ViewType = NDArrayView< ValueType, DeviceType, IndexerType >;

   //! Compatible constant \ref NDArrayView type.
   using ConstViewType = NDArrayView< std::add_const_t< ValueType >, DeviceType, IndexerType >;

   //! \brief Constructs an array view with zero size.
   __cuda_callable__
   NDArrayView() = default;

   //! \brief Constructs an array view initialized by a raw data pointer and
   //! sizes and strides.
   __cuda_callable__
   NDArrayView( Value* data, SizesHolderType sizes, StridesHolderType strides = StridesHolderType{} )
   : IndexerType( sizes, strides ), array( data )
   {}

   //! \brief Constructs an array view initialized by a raw data pointer and an
   //! indexer.
   __cuda_callable__
   NDArrayView( Value* data, IndexerType indexer ) : IndexerType( indexer ), array( data ) {}

   /**
    * \brief A shallow-copy copy-constructor.
    *
    * This allows views to be passed-by-value into CUDA kernels and be
    * captured-by-value in __cuda_callable__ lambda functions.
    */
   __cuda_callable__
   NDArrayView( const NDArrayView& ) = default;

   //! \brief Move constructor for initialization from \e rvalues.
   __cuda_callable__
   NDArrayView( NDArrayView&& ) noexcept = default;

   /**
    * \brief Copy-assignment operator for deep-copying data from another array.
    *
    * This is just like the operator on a regular array, but the sizes must
    * match (i.e. copy-assignment cannot resize).
    *
    * Note that there is no move-assignment operator, so expressions like
    * `a = b.getView()` are resolved as copy-assignment.
    */
   TNL_NVCC_HD_WARNING_DISABLE
   __cuda_callable__
   NDArrayView&
   operator=( const NDArrayView& other )
   {
      TNL_ASSERT_EQ( getSizes(), other.getSizes(), "The sizes of the array views must be equal, views are not resizable." );
      if( getStorageSize() > 0 )
         Algorithms::MemoryOperations< DeviceType >::copy( array, other.array, getStorageSize() );
      return *this;
   }

   //! \brief Templated copy-assignment operator for deep-copying data from another array.
   TNL_NVCC_HD_WARNING_DISABLE
   template< typename OtherView >
   __cuda_callable__
   NDArrayView&
   operator=( const OtherView& other )
   {
      static_assert( std::is_same< PermutationType, typename OtherView::PermutationType >::value,
                     "Arrays must have the same permutation of indices." );
      static_assert( NDArrayView::isContiguous() && OtherView::isContiguous(),
                     "Non-contiguous array views cannot be assigned." );
      TNL_ASSERT_TRUE( detail::sizesWeakCompare( getSizes(), other.getSizes() ),
                       "The sizes of the array views must be equal, views are not resizable." );
      if( getStorageSize() > 0 ) {
         TNL_ASSERT_TRUE( array, "Attempted to assign to an empty view." );
         Algorithms::MultiDeviceMemoryOperations< DeviceType, typename OtherView::DeviceType >::copy(
            array, other.getData(), getStorageSize() );
      }
      return *this;
   }

   //! \brief Re-binds (re-initializes) the array view to a different view.
   __cuda_callable__
   void
   bind( NDArrayView view )
   {
      IndexerType::operator=( view );
      array = view.array;
   }

   //! \brief Re-binds (re-initializes) the array view to the given raw pointer
   //! and changes the indexer.
   __cuda_callable__
   void
   bind( Value* data, IndexerType indexer )
   {
      IndexerType::operator=( indexer );
      array = data;
   }

   //! \brief Re-binds (re-initializes) the array view to the given raw pointer
   //! and preserves the current indexer.
   __cuda_callable__
   void
   bind( Value* data )
   {
      array = data;
   }

   //! \brief Resets the array view to the empty state.
   __cuda_callable__
   void
   reset()
   {
      IndexerType::operator=( IndexerType{} );
      array = nullptr;
   }

   //! \brief Compares the array view with another N-dimensional array view.
   TNL_NVCC_HD_WARNING_DISABLE
   __cuda_callable__
   bool
   operator==( const NDArrayView& other ) const
   {
      if( getSizes() != other.getSizes() )
         return false;
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      // TODO: overlaps should be skipped, otherwise it works only after synchronization
      return Algorithms::MemoryOperations< Device >::compare( array, other.array, getStorageSize() );
   }

   //! \brief Compares the array view with another N-dimensional array view.
   TNL_NVCC_HD_WARNING_DISABLE
   __cuda_callable__
   bool
   operator!=( const NDArrayView& other ) const
   {
      if( getSizes() != other.getSizes() )
         return true;
      // FIXME: uninitialized data due to alignment in NDArray and padding in SlicedNDArray
      return ! Algorithms::MemoryOperations< Device >::compare( array, other.array, getStorageSize() );
   }

   //! \brief Returns a raw pointer to the data.
   __cuda_callable__
   ValueType*
   getData()
   {
      return array;
   }

   //! \brief Returns a \e const-qualified raw pointer to the data.
   __cuda_callable__
   std::add_const_t< ValueType >*
   getData() const
   {
      return array;
   }

   // methods from the base class
   using IndexerType::getDimension;
   using IndexerType::getOverlap;
   using IndexerType::getSize;
   using IndexerType::getSizes;
   using IndexerType::getStorageIndex;
   using IndexerType::getStorageSize;
   using IndexerType::getStride;
   using IndexerType::isContiguousBlock;

   //! Returns a const-qualified reference to the underlying indexer.
   __cuda_callable__
   const IndexerType&
   getIndexer() const
   {
      return *this;
   }

   //! \brief Returns a modifiable view of the array.
   __cuda_callable__
   ViewType
   getView()
   {
      return ViewType( *this );
   }

   //! \brief Returns a non-modifiable view of the array.
   __cuda_callable__
   ConstViewType
   getConstView() const
   {
      return ConstViewType( array, getIndexer() );
   }

   /**
    * \brief Returns a modifiable view of a subarray.
    *
    * \tparam Dimensions Sequence of integers representing the dimensions of the
    *                    array which selects the subset of dimensions to appear
    *                    in the subarray.
    * \param indices Indices of the _origin_ of the subarray in the whole array.
    *                The number of indices supplied must be equal to \e N, i.e.
    *                \ref NDArrayIndexer::getDimension "getDimension()".
    * \returns \ref NDArrayView instantiated for \ref ValueType, \ref DeviceType
    *          and an \ref NDArrayIndexer matching the specified dimensions and
    *          subarray sizes.
    */
   template< std::size_t... Dimensions, typename... IndexTypes >
   __cuda_callable__
   auto
   getSubarrayView( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      static_assert( 0 < sizeof...( Dimensions ) && sizeof...( Dimensions ) <= getDimension(),
                     "got wrong number of dimensions" );
// FIXME: nvcc chokes on the variadic brace-initialization
#ifndef __NVCC__
      static_assert( detail::all_elements_in_range( 0, PermutationType::size(), { Dimensions... } ), "invalid dimensions" );
      static_assert( detail::is_increasing_sequence( { Dimensions... } ), "specifying permuted dimensions is not supported" );
#endif

      using Getter = detail::SubarrayGetter< typename Indexer::NDBaseType, PermutationType, Dimensions... >;
      using Subpermutation = typename Getter::Subpermutation;
      ValueType* begin = getData() + getStorageIndex( std::forward< IndexTypes >( indices )... );
      auto subarray_sizes = Getter::filterSizes( getSizes(), std::forward< IndexTypes >( indices )... );
      auto strides = Getter::getStrides( getSizes(), std::forward< IndexTypes >( indices )... );
      static_assert( Subpermutation::size() == sizeof...( Dimensions ), "Bug - wrong subpermutation length." );
      static_assert( decltype( subarray_sizes )::getDimension() == sizeof...( Dimensions ),
                     "Bug - wrong dimension of the new sizes." );
      static_assert( decltype( strides )::getDimension() == sizeof...( Dimensions ), "Bug - wrong dimension of the strides." );
      // TODO: select overlaps for the subarray
      using Subindexer =
         NDArrayIndexer< decltype( subarray_sizes ), Subpermutation, typename Indexer::NDBaseType, decltype( strides ) >;
      using SubarrayView = NDArrayView< ValueType, Device, Subindexer >;
      return SubarrayView{ begin, subarray_sizes, strides };
   }

   /**
    * \brief Accesses an element of the array.
    *
    * \param indices Indices of the element in the N-dimensional array. The
    *                number of indices supplied must be equal to \e N, i.e.
    *                \ref NDArrayIndexer::getDimension "getDimension()".
    * \returns Reference to the array element.
    */
   template< typename... IndexTypes >
   __cuda_callable__
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   /**
    * \brief Accesses an element of the array.
    *
    * \param indices Indices of the element in the N-dimensional array. The
    *                number of indices supplied must be equal to \e N, i.e.
    *                \ref NDArrayIndexer::getDimension "getDimension()".
    * \returns Constant reference to the array element.
    */
   template< typename... IndexTypes >
   __cuda_callable__
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      return array[ getStorageIndex( std::forward< IndexTypes >( indices )... ) ];
   }

   /**
    * \brief Accesses an element in a one-dimensional array.
    *
    * \warning This function can be used only when the dimension of the array is
    *          equal to 1.
    *
    * \param index Index of the element in the one-dimensional array.
    * \returns Reference to the array element.
    */
   __cuda_callable__
   ValueType&
   operator[]( IndexType&& index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      detail::assertIndicesInBounds( getSizes(), OverlapsType{}, std::forward< IndexType >( index ) );
      return array[ index ];
   }

   /**
    * \brief Accesses an element in a one-dimensional array.
    *
    * \warning This function can be used only when the dimension of the array is
    *          equal to 1.
    *
    * \param index Index of the element in the one-dimensional array.
    * \returns Constant reference to the array element.
    */
   __cuda_callable__
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      detail::assertIndicesInBounds( getSizes(), OverlapsType{}, std::forward< IndexType >( index ) );
      return array[ index ];
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements of the
    * array view.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array elements. For example,
    * given a 3D array view `a` with `int` as \ref IndexType, the function can
    * be defined and used as follows:
    *
    * \code{.cpp}
    * auto setter = [&a] ( int i, int j, int k )
    * {
    *    a( i, j, k ) = 1;
    * };
    * a.forAll( setter );
    * \endcode
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forAll( Func f,
           const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = detail::ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      dispatch( Begins{}, getSizes(), launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all internal elements
    * of the array view.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array elements. For example,
    * given a 3D array view `a` with `int` as \ref IndexType, the function can
    * be defined and used as follows:
    *
    * \code{.cpp}
    * auto setter = [&a] ( int i, int j, int k )
    * {
    *    a( i, j, k ) = 1;
    * };
    * a.forInterior( setter );
    * \endcode
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forInterior(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      using Begins = detail::ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using Ends = typename detail::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      Ends ends;
      detail::SetSizesSubtractHelper< 1, Ends, SizesHolderType >::subtract( ends, getSizes() );
      dispatch( Begins{}, ends, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements of the
    * array view inside the N-dimensional interval `[begins, ends)`.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array view elements.
    */
   template< typename Device2 = DeviceType, typename Begins, typename Ends, typename Func >
   void
   forInterior(
      const Begins& begins,
      const Ends& ends,
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // TODO: assert "begins <= getSizes()", "ends <= getSizes()"
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all boundary elements
    * of the array view.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array view elements. For
    * example, given a 3D array view `a` with `int` as \ref IndexType, the
    * function can be defined and used as follows:
    *
    * \code{.cpp}
    * auto setter = [&a] ( int i, int j, int k )
    * {
    *    a( i, j, k ) = 1;
    * };
    * a.forBoundary( setter );
    * \endcode
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forBoundary(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      using Begins = detail::ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      using SkipBegins = detail::ConstStaticSizesHolder< IndexType, getDimension(), 1 >;
      // subtract static sizes
      using SkipEnds = typename detail::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      SkipEnds skipEnds;
      detail::SetSizesSubtractHelper< 1, SkipEnds, SizesHolderType >::subtract( skipEnds, getSizes() );

      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, SkipBegins{}, skipEnds, getSizes(), launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements of the
    * array view outside the N-dimensional interval `[skipBegins, skipEnds)`.
    *
    * The function `f` is called as `f(indices...)`, where `indices...` are
    * substituted by the actual indices of all array view elements.
    */
   template< typename Device2 = DeviceType, typename SkipBegins, typename SkipEnds, typename Func >
   void
   forBoundary(
      const SkipBegins& skipBegins,
      const SkipEnds& skipEnds,
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // TODO: assert "skipBegins <= getSizes()", "skipEnds <= getSizes()"
      using Begins = detail::ConstStaticSizesHolder< IndexType, getDimension(), 0 >;
      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( Begins{}, skipBegins, skipEnds, getSizes(), launch_configuration, f );
   }

protected:
   //! \brief Pointer to the first element of the bound N-dimensional array.
   Value* array = nullptr;

   //! \brief Object which transforms the multi-dimensional indices to a
   //! one-dimensional index.
   IndexerType indexer;
};

}  // namespace Containers
}  // namespace noa::TNL
