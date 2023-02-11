// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/Indexing.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/SizesHolderHelpers.h>  // StorageSizeGetter
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/Subarrays.h>           // DummyStrideBase

namespace noa::TNL {
namespace Containers {

// HACK for https://stackoverflow.com/q/74240374
#ifdef _MSC_VER
template< typename T >
constexpr std::size_t
getDimension()
{
   return T::getDimension();
}
#endif

/**
 * \brief Indexer for N-dimensional arrays. It does not store any data, only
 * the sizes of each dimension.
 *
 * \tparam SizesHolder Instance of \ref SizesHolder that will represent the
 *                     array sizes.
 * \tparam Permutation Permutation that will be applied to indices when
 *                     accessing the array elements. The identity permutation
 *                     is used by default.
 * \tparam Base Either `detail::NDArrayBase` or `detail::SlicedNDArrayBase`.
 * \tparam StridesHolder Type of the base class which represents the strides of
 *                       the N-dimensional array.
 * \tparam Overlaps Sequence of integers representing the overlaps in each
 *                  dimension a distributed N-dimensional array.
 *
 * \ingroup ndarray
 */
template< typename SizesHolder,
          typename Permutation,
          typename Base,
          typename StridesHolder = detail::DummyStrideBase< typename SizesHolder::IndexType, SizesHolder::getDimension() >,
          typename Overlaps = detail::make_constant_index_sequence< SizesHolder::getDimension(), 0 > >
class NDArrayIndexer : public StridesHolder
{
protected:
   using NDBaseType = Base;

public:
   //! \brief Type of the underlying object which represents the sizes of the N-dimensional array.
   using SizesHolderType = SizesHolder;

   //! \brief Type of the base class which represents the strides of the N-dimensional array.
   using StridesHolderType = StridesHolder;

   //! \brief Permutation that is applied to indices when accessing the array elements.
   using PermutationType = Permutation;

   //! \brief Sequence of integers representing the overlaps in each dimension
   //! of a distributed N-dimensional array.
   using OverlapsType = Overlaps;

   //! \brief Type of indices used for addressing the array elements.
   using IndexType = typename SizesHolder::IndexType;

// HACK for https://stackoverflow.com/q/74240374
#ifdef _MSC_VER
   static_assert( Containers::getDimension< StridesHolder >() == Containers::getDimension< SizesHolder >(),
                  "Dimension of strides does not match the dimension of sizes." );
   static_assert( Permutation::size() == Containers::getDimension< SizesHolder >(),
                  "Dimension of permutation does not match the dimension of sizes." );
   static_assert( Overlaps::size() == Containers::getDimension< SizesHolder >(),
                  "Dimension of overlaps does not match the dimension of sizes." );
#else
   static_assert( StridesHolder::getDimension() == SizesHolder::getDimension(),
                  "Dimension of strides does not match the dimension of sizes." );
   static_assert( Permutation::size() == SizesHolder::getDimension(),
                  "Dimension of permutation does not match the dimension of sizes." );
   static_assert( Overlaps::size() == SizesHolder::getDimension(),
                  "Dimension of overlaps does not match the dimension of sizes." );
#endif

   //! \brief Constructs an empty indexer with zero sizes and strides.
   __cuda_callable__
   NDArrayIndexer() = default;

   //! \brief Creates the indexer with given sizes and strides.
   __cuda_callable__
   NDArrayIndexer( SizesHolderType sizes, StridesHolderType strides ) : StridesHolder( strides ), sizes( sizes ) {}

   //! \brief Returns the dimension of the \e N-dimensional array, i.e. \e N.
   static constexpr std::size_t
   getDimension()
   {
// HACK for https://stackoverflow.com/q/74240374
#ifdef _MSC_VER
      return Containers::getDimension< SizesHolder >();
#else
      return SizesHolder::getDimension();
#endif
   }

   //! \brief Returns the N-dimensional array sizes held by the indexer.
   __cuda_callable__
   const SizesHolderType&
   getSizes() const
   {
      return sizes;
   }

   /**
    * \brief Returns a specific component of the N-dimensional sizes.
    *
    * \tparam level Integer specifying the component of the sizes to be returned.
    */
   template< std::size_t level >
   __cuda_callable__
   IndexType
   getSize() const
   {
      return sizes.template getSize< level >();
   }

   // method template from base class
   using StridesHolder::getStride;

   /**
    * \brief Returns the overlap of a distributed N-dimensional array along the
    *        specified axis.
    *
    * \tparam level Integer specifying the axis of the array.
    */
   template< std::size_t level >
   static constexpr std::size_t
   getOverlap()
   {
      return detail::get< level >( Overlaps{} );
   }

   /**
    * \brief Returns the size (number of elements) needed to store the N-dimensional array.
    *
    * \returns The product of the aligned sizes.
    */
   __cuda_callable__
   IndexType
   getStorageSize() const
   {
      using Alignment = typename Base::template Alignment< Permutation >;
      return detail::StorageSizeGetter< SizesHolder, Alignment, Overlaps >::get( sizes );
   }

   /**
    * \brief Computes the one-dimensional storage index for a specific element
    *        of the N-dimensional array.
    *
    * \param indices Indices of the element in the N-dimensional array. The
    *                number of indices supplied must be equal to \e N, i.e.
    *                \ref getDimension().
    * \returns An index that can be used to address the element in a
    *          one-dimensional array.
    */
   template< typename... IndexTypes >
   __cuda_callable__
   IndexType
   getStorageIndex( IndexTypes&&... indices ) const
   {
      static_assert( sizeof...( indices ) == getDimension(), "got wrong number of indices" );
      detail::assertIndicesInBounds( getSizes(), OverlapsType{}, std::forward< IndexTypes >( indices )... );
      const IndexType result = Base::template getStorageIndex< Permutation, Overlaps >(
         sizes, static_cast< const StridesHolder& >( *this ), std::forward< IndexTypes >( indices )... );
      TNL_ASSERT_GE( result, (IndexType) 0, "storage index out of bounds - either input error or a bug in the indexer" );
      // upper bound can be checked only for contiguous arrays/views
      if( StridesHolder::isContiguous() ) {
         TNL_ASSERT_LT( result, getStorageSize(), "storage index out of bounds - either input error or a bug in the indexer" );
      }
      return result;
   }

   template< typename BeginsHolder, typename EndsHolder >
   __cuda_callable__
   bool
   isContiguousBlock( const BeginsHolder& begins, const EndsHolder& ends )
   {
      static_assert( BeginsHolder::getDimension() == getDimension(), "invalid dimension of the begins parameter" );
      static_assert( EndsHolder::getDimension() == getDimension(), "invalid dimension of the ends parameter" );

      bool check = false;
      bool contiguous = true;

      Algorithms::staticFor< std::size_t, 0, getDimension() >(
         [ & ]( auto i )
         {
            constexpr int dim = TNL::Containers::detail::get< i >( Permutation{} );
            const auto size = getSize< dim >();
            const auto blockSize = ends.template getSize< dim >() - begins.template getSize< dim >();
            // blockSize can be different from size only in the first dimension,
            // then the sizes must match in all following dimensions
            if( check && blockSize != size )
               contiguous = false;
            if( blockSize > 1 )
               check = true;
         } );

      return contiguous;
   }

protected:
   /**
    * \brief Returns a non-constant reference to the underlying \ref sizes.
    *
    * The function is not public -- only subclasses like \ref NDArrayStorage
    * may modify the sizes.
    */
   __cuda_callable__
   SizesHolderType&
   getSizes()
   {
      return sizes;
   }

   //! \brief Underlying object which represents the sizes of the N-dimensional array.
   SizesHolderType sizes;
};

}  // namespace Containers
}  // namespace noa::TNL
