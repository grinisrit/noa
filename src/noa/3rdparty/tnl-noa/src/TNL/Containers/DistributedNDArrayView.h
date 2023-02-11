// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/NDArrayView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Subrange.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Wrappers.h>

namespace noa::TNL {
namespace Containers {

/**
 * \brief Distributed N-dimensional array view.
 *
 * \tparam NDArrayView Type of the N-dimensional array view which is used to
 *                     access the local elements. It must be an instance of the
 *                     \ref NDArrayView template.
 *
 * \ingroup ndarray
 */
template< typename NDArrayView >
class DistributedNDArrayView
{
public:
   //! \brief Type of the values stored in the array.
   using ValueType = typename NDArrayView::ValueType;

   //! \brief Type of the \ref TNL::Devices "device" used for running operations on the array.
   using DeviceType = typename NDArrayView::DeviceType;

   //! \brief Type of indices used for addressing the array elements.
   using IndexType = typename NDArrayView::IndexType;

   //! \brief Type of the underlying object which represents the sizes of the N-dimensional array.
   using SizesHolderType = typename NDArrayView::SizesHolderType;

   //! \brief Permutation that is applied to indices when accessing the array elements.
   using PermutationType = typename NDArrayView::PermutationType;

   //! \brief Type which represents the position of the first local element in
   //! the global N-dimensional array. It has all static sizes set to 0.
   using LocalBeginsType = detail::LocalBeginsHolder< typename NDArrayView::SizesHolderType >;

   //! \brief Type which represents an integer range `[a, b)`.
   using LocalRangeType = Subrange< IndexType >;

   //! \brief Sequence of integers representing the overlaps in each dimension
   //! of a distributed N-dimensional array.
   using OverlapsType = typename NDArrayView::OverlapsType;

   //! \brief Compatible \ref DistributedNDArrayView type.
   using ViewType = DistributedNDArrayView< NDArrayView >;

   //! \brief Compatible constant \ref DistributedNDArrayView type.
   using ConstViewType = DistributedNDArrayView< typename NDArrayView::ConstViewType >;

   //! \brief Compatible \ref NDArrayView of the local array.
   using LocalViewType = NDArrayView;

   //! \brief Compatible constant \ref NDArrayView of the local array.
   using ConstLocalViewType = typename NDArrayView::ConstViewType;

   //! \brief Constructs an empty array view with zero size.
   DistributedNDArrayView() = default;

   //! \brief Constructs an array view initialized by local array view, global
   //! sizes, local begins and ends, and MPI communicator.
   DistributedNDArrayView( NDArrayView localView,
                           SizesHolderType globalSizes,
                           LocalBeginsType localBegins,
                           SizesHolderType localEnds,
                           MPI::Comm communicator )
   : localView( localView ), communicator( std::move( communicator ) ), globalSizes( globalSizes ), localBegins( localBegins ),
     localEnds( localEnds )
   {}

   //! \brief A shallow-copy copy-constructor.
   DistributedNDArrayView( const DistributedNDArrayView& ) = default;

   //! \brief Move constructor for initialization from \e rvalues.
   DistributedNDArrayView( DistributedNDArrayView&& ) noexcept = default;

   /**
    * \brief Copy-assignment operator for deep-copying data from another array.
    *
    * This is just like the operator on a regular array, but the sizes must
    * match (i.e. copy-assignment cannot resize).
    *
    * Note that there is no move-assignment operator, so expressions like
    * `a = b.getView()` are resolved as copy-assignment.
    */
   DistributedNDArrayView&
   operator=( const DistributedNDArrayView& other ) = default;

   //! \brief Templated copy-assignment operator for deep-copying data from another array.
   template< typename OtherArray >
   DistributedNDArrayView&
   operator=( const OtherArray& other )
   {
      globalSizes = other.getSizes();
      localBegins = other.getLocalBegins();
      localEnds = other.getLocalEnds();
      communicator = other.getCommunicator();
      localView = other.getConstLocalView();
      return *this;
   }

   //! \brief Re-binds (re-initializes) the array view to a different view.
   void
   bind( DistributedNDArrayView view )
   {
      localView.bind( view.localView );
      communicator = view.communicator;
      globalSizes = view.globalSizes;
      localBegins = view.localBegins;
      localEnds = view.localEnds;
   }

   //! \brief Re-binds (re-initializes) the array view to the given raw pointer
   //! and changes the indexer.
   void
   bind( ValueType* data, typename LocalViewType::IndexerType indexer )
   {
      localView.bind( data, indexer );
   }

   //! \brief Re-binds (re-initializes) the array view to the given raw pointer
   //! and preserves the current indexer.
   void
   bind( ValueType* data )
   {
      localView.bind( data );
   }

   //! \brief Resets the array view to the empty state.
   void
   reset()
   {
      localView.reset();
      communicator = MPI_COMM_NULL;
      globalSizes = SizesHolderType{};
      localBegins = LocalBeginsType{};
      localEnds = SizesHolderType{};
   }

   //! \brief Returns the dimension of the \e N-dimensional array, i.e. \e N.
   static constexpr std::size_t
   getDimension()
   {
      return NDArrayView::getDimension();
   }

   //! \brief Returns the MPI communicator associated with the array.
   const MPI::Comm&
   getCommunicator() const
   {
      return communicator;
   }

   //! \brief Returns the N-dimensional sizes of the **global** array.
   const SizesHolderType&
   getSizes() const
   {
      return globalSizes;
   }

   /**
    * \brief Returns a specific component of the N-dimensional **global** sizes.
    *
    * \tparam level Integer specifying the component of the sizes to be returned.
    */
   template< std::size_t level >
   IndexType
   getSize() const
   {
      return globalSizes.template getSize< level >();
   }

   //! \brief Returns the beginning position of the local array in the global
   //! N-dimensional array.
   LocalBeginsType
   getLocalBegins() const
   {
      return localBegins;
   }

   //! \brief Returns the ending position of the local array in the global
   //! N-dimensional array.
   SizesHolderType
   getLocalEnds() const
   {
      return localEnds;
   }

   /**
    * \brief Returns a specific `[begin, end)` subrange of the local array in
    * the global N-dimensional array.
    *
    * \tparam level Integer specifying the component of the sizes to be returned.
    */
   template< std::size_t level >
   LocalRangeType
   getLocalRange() const
   {
      return LocalRangeType( localBegins.template getSize< level >(), localEnds.template getSize< level >() );
   }

   //! \brief Returns the size (number of elements) needed to store the local
   //! N-dimensional array.
   IndexType
   getLocalStorageSize() const
   {
      return localView.getStorageSize();
   }

   //! \brief Returns a modifiable view of the local array.
   LocalViewType
   getLocalView()
   {
      return localView;
   }

   //! \brief Returns a non-modifiable view of the local array.
   ConstLocalViewType
   getConstLocalView() const
   {
      return localView.getConstView();
   }

   /**
    * \brief Returns the **local** storage index for given **global** indices.
    *
    * \param indices Global indices of the element in the N-dimensional array.
    *                The number of indices supplied must be equal to \e N, i.e.
    *                \ref getDimension().
    * \returns An index that can be used to address the element in a local
    *          one-dimensional array.
    */
   template< typename... IndexTypes >
   IndexType
   getStorageIndex( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == SizesHolderType::getDimension(), "got wrong number of indices" );
      detail::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      auto getStorageIndex = [ this ]( auto&&... indices )
      {
         return this->localView.getStorageIndex( std::forward< decltype( indices ) >( indices )... );
      };
      return detail::host_call_with_unshifted_indices< LocalBeginsType >(
         localBegins, getStorageIndex, std::forward< IndexTypes >( indices )... );
   }

   //! \brief Returns a raw pointer to the local data.
   ValueType*
   getData()
   {
      return localView.getData();
   }

   //! \brief Returns a \e const-qualified raw pointer to the local data.
   std::add_const_t< ValueType >*
   getData() const
   {
      return localView.getData();
   }

   /**
    * \brief Accesses an element of the array.
    *
    * Only local elements or elements in the overlapping region can be
    * accessed.
    *
    * \param indices Global indices of the element in the N-dimensional array.
    *                The number of indices supplied must be equal to \e N, i.e.
    *                \ref getDimension().
    * \returns Reference to the array element.
    */
   template< typename... IndexTypes >
   ValueType&
   operator()( IndexTypes&&... indices )
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      detail::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      return detail::call_with_unshifted_indices< LocalBeginsType >(
         localBegins, localView, std::forward< IndexTypes >( indices )... );
   }

   /**
    * \brief Accesses an element of the array.
    *
    * Only local elements or elements in the overlapping region can be
    * accessed.
    *
    * \param indices Global indices of the element in the N-dimensional array.
    *                The number of indices supplied must be equal to \e N, i.e.
    *                \ref getDimension().
    * \returns Constant reference to the array element.
    */
   template< typename... IndexTypes >
   const ValueType&
   operator()( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      detail::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexTypes >( indices )... );
      return detail::call_with_unshifted_indices< LocalBeginsType >(
         localBegins, localView, std::forward< IndexTypes >( indices )... );
   }

   /**
    * \brief Accesses an element in a one-dimensional array.
    *
    * Only local elements or elements in the overlapping region can be
    * accessed.
    *
    * \warning This function can be used only when the dimension of the array is
    *          equal to 1.
    *
    * \param index Global index of the element in the one-dimensional array.
    * \returns Reference to the array element.
    */
   ValueType&
   operator[]( IndexType index )
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      detail::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexType >( index ) );
      return localView[ index - localBegins.template getSize< 0 >() ];
   }

   /**
    * \brief Accesses an element in a one-dimensional array.
    *
    * Only local elements or elements in the overlapping region can be
    * accessed.
    *
    * \warning This function can be used only when the dimension of the array is
    *          equal to 1.
    *
    * \param index Global index of the element in the one-dimensional array.
    * \returns Reference to the array element.
    */
   const ValueType&
   operator[]( IndexType index ) const
   {
      static_assert( getDimension() == 1, "the access via operator[] is provided only for 1D arrays" );
      detail::assertIndicesInRange( localBegins, localEnds, OverlapsType{}, std::forward< IndexType >( index ) );
      return localView[ index - localBegins.template getSize< 0 >() ];
   }

   //! \brief Returns a modifiable view of the array.
   ViewType
   getView()
   {
      return ViewType( *this );
   }

   //! \brief Returns a non-modifiable view of the array.
   ConstViewType
   getConstView() const
   {
      return ConstViewType( localView, globalSizes, localBegins, localEnds, communicator );
   }

   //! \brief Compares the array with another distributed N-dimensional array.
   bool
   operator==( const DistributedNDArrayView& other ) const
   {
      // we can't run allreduce if the communicators are different
      if( communicator != other.getCommunicator() )
         return false;
      // TODO: overlaps should be skipped, otherwise it works only after synchronization
      const bool localResult = globalSizes == other.globalSizes && localBegins == other.localBegins
                            && localEnds == other.localEnds && localView == other.localView;
      bool result = true;
      if( communicator != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, communicator );
      return result;
   }

   //! \brief Compares the array with another distributed N-dimensional array.
   bool
   operator!=( const DistributedNDArrayView& other ) const
   {
      return ! ( *this == other );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements of the
    * array.
    *
    * Each MPI rank iterates over all of its local elements.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forAll( Func f,
           const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, localEnds, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all internal elements
    * of the array.
    *
    * Each MPI rank iterates over its local elements which are not neighbours
    * of **global** boundaries.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forInterior(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // add static sizes
      using Begins = detail::LocalBeginsHolder< SizesHolderType, 1 >;
      // add dynamic sizes
      Begins begins;
      detail::SetSizesAddHelper< 1, Begins, SizesHolderType, OverlapsType >::add( begins, SizesHolderType{} );
      detail::SetSizesMaxHelper< Begins, LocalBeginsType >::max( begins, localBegins );

      // subtract static sizes
      using Ends = typename detail::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      Ends ends;
      detail::SetSizesSubtractHelper< 1, Ends, SizesHolderType, OverlapsType >::subtract( ends, globalSizes );
      detail::SetSizesMinHelper< Ends, SizesHolderType >::min( ends, localEnds );

      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements inside the
    * given `[begins, ends)` range specified by global indices.
    *
    * Each MPI rank iterates over its local elements from the range.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename Begins, typename Ends, typename Func >
   void
   forInterior(
      const Begins& begins,
      const Ends& ends,
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // TODO: assert "localBegins <= begins <= localEnds", "localBegins <= ends <= localEnds"
      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all boundary elements
    * of the array.
    *
    * Each MPI rank iterates over its local elements which are neighbours of
    * **global** boundaries.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forBoundary(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // add static sizes
      using SkipBegins = detail::LocalBeginsHolder< SizesHolderType, 1 >;
      // add dynamic sizes
      SkipBegins skipBegins;
      detail::SetSizesAddHelper< 1, SkipBegins, SizesHolderType, OverlapsType >::add( skipBegins, SizesHolderType{} );
      detail::SetSizesMaxHelper< SkipBegins, LocalBeginsType >::max( skipBegins, localBegins );

      // subtract static sizes
      using SkipEnds = typename detail::SubtractedSizesHolder< SizesHolderType, 1 >::type;
      // subtract dynamic sizes
      SkipEnds skipEnds;
      detail::SetSizesSubtractHelper< 1, SkipEnds, SizesHolderType, OverlapsType >::subtract( skipEnds, globalSizes );
      detail::SetSizesMinHelper< SkipEnds, SizesHolderType >::min( skipEnds, localEnds );

      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, skipBegins, skipEnds, localEnds, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements outside
    * the given `[skipBegins, skipEnds)` range specified by global indices.
    *
    * Each MPI rank iterates over its local elements outside the range.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename SkipBegins, typename SkipEnds, typename Func >
   void
   forBoundary(
      const SkipBegins& skipBegins,
      const SkipEnds& skipEnds,
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // TODO: assert "localBegins <= skipBegins <= localEnds", "localBegins <= skipEnds <= localEnds"
      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, skipBegins, skipEnds, localEnds, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all local-internal
    * elements of the array.
    *
    * Each MPI rank iterates over its local elements which are not neighbours
    * of overlaps. If all overlaps are 0, it is equivalent to \ref forAll.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forLocalInterior(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // add overlaps to dynamic sizes
      LocalBeginsType begins;
      detail::SetSizesAddOverlapsHelper< LocalBeginsType, SizesHolderType, OverlapsType >::add( begins, localBegins );

      // subtract overlaps from dynamic sizes
      SizesHolderType ends;
      detail::SetSizesSubtractOverlapsHelper< SizesHolderType, SizesHolderType, OverlapsType >::subtract( ends, localEnds );

      detail::ExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, ends, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all local-boundary
    * elements of the array.
    *
    * Each MPI rank iterates over its local elements which are neighbours of
    * overlaps. If all overlaps are 0, it has no effect.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forLocalBoundary(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // add overlaps to dynamic sizes
      LocalBeginsType skipBegins;
      detail::SetSizesAddOverlapsHelper< LocalBeginsType, SizesHolderType, OverlapsType >::add( skipBegins, localBegins );

      // subtract overlaps from dynamic sizes
      SizesHolderType skipEnds;
      detail::SetSizesSubtractOverlapsHelper< SizesHolderType, SizesHolderType, OverlapsType >::subtract( skipEnds, localEnds );

      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( localBegins, skipBegins, skipEnds, localEnds, launch_configuration, f );
   }

   /**
    * \brief Evaluates the function `f` in parallel for all elements in the
    * ghost region.
    *
    * Each MPI rank iterates over elements which are in the overlapping region
    * (i.e., owned by a different MPI rank). If all overlaps are 0, it has no
    * effect.
    *
    * See \ref NDArrayView::forAll for the requirements on the function `f`.
    */
   template< typename Device2 = DeviceType, typename Func >
   void
   forGhosts(
      Func f,
      const typename Device2::LaunchConfiguration& launch_configuration = typename Device2::LaunchConfiguration{} ) const
   {
      // subtract overlaps from dynamic sizes
      LocalBeginsType begins;
      detail::SetSizesSubtractOverlapsHelper< LocalBeginsType, SizesHolderType, OverlapsType >::subtract( begins, localBegins );

      // add overlaps to dynamic sizes
      SizesHolderType ends;
      detail::SetSizesAddOverlapsHelper< SizesHolderType, SizesHolderType, OverlapsType >::add( ends, localEnds );

      detail::BoundaryExecutorDispatcher< PermutationType, Device2 > dispatch;
      dispatch( begins, localBegins, localEnds, ends, launch_configuration, f );
   }

protected:
   //! \brief View of the N-dimensional array containing the local elements and overlaps.
   NDArrayView localView;

   //! \brief MPI communicator associated with the array.
   MPI::Comm communicator = MPI_COMM_NULL;

   //! \brief Global sizes of the whole distributed N-dimensional array.
   SizesHolderType globalSizes;

   /**
    * \brief Global indices of the first local element in the whole
    * N-dimensional array.
    *
    * Note that `localBegins` and \ref localEnds have different static sizes
    * (and hence different C++ type): `localBegins` is always 0, `localEnds`
    * has always the full static size.
    */
   LocalBeginsType localBegins;

   //! \brief Global indices of the end-of-range element,
   //! `[localBegins, localEnds)`.
   SizesHolderType localEnds;
};

}  // namespace Containers
}  // namespace noa::TNL
