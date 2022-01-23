// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Containers/ndarray/SizesHolderHelpers.h>

namespace noa::TNL {
namespace Containers {
namespace __ndarray_impl {

template< typename OffsetsHolder,
          typename Sequence >
struct OffsetsHelper
{};

template< typename OffsetsHolder,
          std::size_t... N >
struct OffsetsHelper< OffsetsHolder, std::index_sequence< N... > >
{
   template< typename Func >
   __cuda_callable__
   static auto apply( const OffsetsHolder& offsets, Func&& f ) -> decltype(auto)
   {
      return f( offsets.template getSize< N >()... );
   }

   template< typename Func >
   static auto apply_host( const OffsetsHolder& offsets, Func&& f ) -> decltype(auto)
   {
      return f( offsets.template getSize< N >()... );
   }
};

template< typename OffsetsHolder,
          typename Func >
__cuda_callable__
auto call_with_offsets( const OffsetsHolder& offsets, Func&& f ) -> decltype(auto)
{
   return OffsetsHelper< OffsetsHolder, std::make_index_sequence< OffsetsHolder::getDimension() > >
          ::apply( offsets, std::forward< Func >( f ) );
}

template< typename OffsetsHolder,
          typename Func >
auto host_call_with_offsets( const OffsetsHolder& offsets, Func&& f ) -> decltype(auto)
{
   return OffsetsHelper< OffsetsHolder, std::make_index_sequence< OffsetsHolder::getDimension() > >
          ::apply_host( offsets, std::forward< Func >( f ) );
}


template< typename OffsetsHolder,
          typename Sequence >
struct IndexShiftHelper
{};

template< typename OffsetsHolder,
          std::size_t... N >
struct IndexShiftHelper< OffsetsHolder, std::index_sequence< N... > >
{
   template< typename Func,
             typename... Indices >
   __cuda_callable__
   static auto apply( const OffsetsHolder& offsets, Func&& f, Indices&&... indices ) -> decltype(auto)
   {
      return f( ( std::forward< Indices >( indices ) + offsets.template getSize< N >() )... );
   }

   template< typename Func,
             typename... Indices >
   static auto apply_host( const OffsetsHolder& offsets, Func&& f, Indices&&... indices ) -> decltype(auto)
   {
      return f( ( std::forward< Indices >( indices ) + offsets.template getSize< N >() )... );
   }
};

template< typename OffsetsHolder,
          typename Func,
          typename... Indices >
__cuda_callable__
auto call_with_shifted_indices( const OffsetsHolder& offsets, Func&& f, Indices&&... indices ) -> decltype(auto)
{
   return IndexShiftHelper< OffsetsHolder, std::make_index_sequence< sizeof...( Indices ) > >
          ::apply( offsets, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}

template< typename OffsetsHolder,
          typename Func,
          typename... Indices >
auto host_call_with_shifted_indices( const OffsetsHolder& offsets, Func&& f, Indices&&... indices ) -> decltype(auto)
{
   return IndexShiftHelper< OffsetsHolder, std::make_index_sequence< sizeof...( Indices ) > >
          ::apply_host( offsets, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}


template< typename SizesHolder,
          typename Overlaps,
          typename Sequence >
struct IndexUnshiftHelper
{};

template< typename SizesHolder,
          typename Overlaps,
          std::size_t... N >
struct IndexUnshiftHelper< SizesHolder, Overlaps, std::index_sequence< N... > >
{
   template< typename Func,
             typename... Indices >
   __cuda_callable__
   static auto apply( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
   {
      return f( ( get<N>( Overlaps{} ) + std::forward< Indices >( indices ) - begins.template getSize< N >() )... );
   }

   template< typename Func,
             typename... Indices >
   static auto apply_host( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
   {
      return f( ( get<N>( Overlaps{} ) + std::forward< Indices >( indices ) - begins.template getSize< N >() )... );
   }
};

template< typename SizesHolder,
          typename Overlaps = make_constant_index_sequence< SizesHolder::getDimension(), 0 >,
          typename Func,
          typename... Indices >
__cuda_callable__
auto call_with_unshifted_indices( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
{
   return IndexUnshiftHelper< SizesHolder, Overlaps, std::make_index_sequence< sizeof...( Indices ) > >
          ::apply( begins, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}

template< typename SizesHolder,
          typename Overlaps = make_constant_index_sequence< SizesHolder::getDimension(), 0 >,
          typename Func,
          typename... Indices >
auto host_call_with_unshifted_indices( const SizesHolder& begins, Func&& f, Indices&&... indices ) -> decltype(auto)
{
   return IndexUnshiftHelper< SizesHolder, Overlaps, std::make_index_sequence< sizeof...( Indices ) > >
          ::apply_host( begins, std::forward< Func >( f ), std::forward< Indices >( indices )... );
}


template< typename Permutation,
          typename Overlaps,
          typename Alignment,
          typename SliceInfo,
          std::size_t level = Permutation::size() - 1,
          bool _sliced_level = ( SliceInfo::getSliceSize( get< level >( Permutation{} ) ) > 0 ) >
struct SlicedIndexer
{};

template< typename Permutation,
          typename Overlaps,
          typename Alignment,
          typename SliceInfo,
          std::size_t level >
struct SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo, level, false >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static constexpr std::size_t idx = get< level >( Permutation{} );
      static constexpr std::size_t overlap = __ndarray_impl::get< idx >( Overlaps{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      const auto size = Alignment::template getAlignedSize< idx >( sizes ) + 2 * overlap;
      const auto previous = SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo, level - 1 >::getIndex( sizes, strides, std::forward< Indices >( indices )... );

      return strides.template getStride< idx >( alpha ) * ( alpha + overlap + size * previous );
   }
};

template< typename Permutation,
          typename Overlaps,
          typename Alignment,
          typename SliceInfo,
          std::size_t level >
struct SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo, level, true >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static_assert( SizesHolder::template getStaticSize< get< level >( Permutation{} ) >() == 0,
                     "Invalid SliceInfo: static dimension cannot be sliced." );

      static constexpr std::size_t idx = get< level >( Permutation{} );
      static constexpr std::size_t overlap = __ndarray_impl::get< idx >( Overlaps{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      static constexpr std::size_t S = SliceInfo::getSliceSize( idx );
      // TODO: check the calculation with strides and overlaps
      return strides.template getStride< idx >( alpha ) *
                  ( S * ((alpha + overlap) / S) * StorageSizeGetter< SizesHolder, Alignment, Overlaps, IndexTag< level - 1 > >::getPermuted( sizes, Permutation{} ) +
                    (alpha + overlap) % S ) +
             S * SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo, level - 1 >::getIndex( sizes, strides, std::forward< Indices >( indices )... );
   }
};

template< typename Permutation,
          typename Overlaps,
          typename Alignment,
          typename SliceInfo >
struct SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo, 0, false >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static constexpr std::size_t idx = get< 0 >( Permutation{} );
      static constexpr std::size_t overlap = __ndarray_impl::get< idx >( Overlaps{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      return strides.template getStride< idx >( alpha ) * (alpha + overlap);
   }
};

template< typename Permutation,
          typename Overlaps,
          typename Alignment,
          typename SliceInfo >
struct SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo, 0, true >
{
   template< typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getIndex( const SizesHolder& sizes,
             const StridesHolder& strides,
             Indices&&... indices )
   {
      static constexpr std::size_t idx = get< 0 >( Permutation{} );
      static constexpr std::size_t overlap = __ndarray_impl::get< idx >( Overlaps{} );
      const auto alpha = get_from_pack< idx >( std::forward< Indices >( indices )... );
      return strides.template getStride< idx >( alpha ) * (alpha + overlap);
   }
};


// SliceInfo should be always empty (i.e. sliceSize == 0)
template< typename SliceInfo >
struct NDArrayBase
{
   template< typename Permutation >
   struct Alignment
   {
      template< std::size_t dimension, typename SizesHolder >
      __cuda_callable__
      static typename SizesHolder::IndexType
      getAlignedSize( const SizesHolder& sizes )
      {
         const auto size = sizes.template getSize< dimension >();
         // round up the last dynamic dimension to improve performance
         // TODO: aligning is good for GPU, but bad for CPU
//         static constexpr decltype(size) mult = 32;
//         if( dimension == get< Permutation::size() - 1 >( Permutation{} )
//                 && SizesHolder::template getStaticSize< dimension >() == 0 )
//             return mult * ( size / mult + ( size % mult != 0 ) );
         return size;
      }
   };

   template< typename Permutation, typename Overlaps, typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   typename SizesHolder::IndexType
   static getStorageIndex( const SizesHolder& sizes, const StridesHolder& strides, Indices&&... indices )
   {
      static_assert( check_slice_size( SizesHolder::getDimension(), 0 ), "BUG - invalid SliceInfo type passed to NDArrayBase" );
      using Alignment = Alignment< Permutation >;
      return SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo >::getIndex( sizes, strides, std::forward< Indices >( indices )... );
   }

private:
   static constexpr bool check_slice_size( std::size_t dim, std::size_t sliceSize )
   {
      for( std::size_t i = 0; i < dim; i++ )
         if( SliceInfo::getSliceSize( i ) != sliceSize )
            return false;
      return true;
   }
};


template< typename SliceInfo >
struct SlicedNDArrayBase
{
   template< typename Permutation >
   struct Alignment
   {
      template< std::size_t dimension, typename SizesHolder >
      __cuda_callable__
      static typename SizesHolder::IndexType
      getAlignedSize( const SizesHolder& sizes )
      {
         const auto size = sizes.template getSize< dimension >();
         if( SliceInfo::getSliceSize(dimension) > 0 )
// icpc does not consider the condition when printing warnings
#ifdef __INTEL_COMPILER
   #pragma warning push
   #pragma warning disable 39    // division by zero
   #pragma warning disable 179   // right operand of "%" is zero
#endif
            // round to multiple of SliceSize
            return SliceInfo::getSliceSize(dimension) * (
                        size / SliceInfo::getSliceSize(dimension) +
                        ( size % SliceInfo::getSliceSize(dimension) != 0 )
                     );
#ifdef __INTEL_COMPILER
   #pragma warning pop
#endif
         // unmodified
         return size;
      }
   };

   template< typename Permutation, typename Overlaps, typename SizesHolder, typename StridesHolder, typename... Indices >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getStorageIndex( const SizesHolder& sizes, const StridesHolder& strides, Indices&&... indices )
   {
      using Alignment = Alignment< Permutation >;
      return SlicedIndexer< Permutation, Overlaps, Alignment, SliceInfo >::getIndex( sizes, strides, std::forward< Indices >( indices )... );
   }
};

} // namespace __ndarray_impl
} // namespace Containers
} // namespace noa::TNL
