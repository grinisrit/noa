// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <algorithm>

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Algorithms/staticFor.h>
#include <noa/3rdparty/TNL/Containers/ndarray/Meta.h>

namespace noa::TNL {
namespace Containers {
namespace __ndarray_impl {

// Dynamic storage size with alignment
template< typename SizesHolder,
          typename Alignment,
          typename Overlaps,
          typename LevelTag = IndexTag< SizesHolder::getDimension() - 1 > >
struct StorageSizeGetter
{
   static typename SizesHolder::IndexType
   __cuda_callable__
   get( const SizesHolder& sizes )
   {
      static constexpr std::size_t overlap = __ndarray_impl::get< LevelTag::value >( Overlaps{} );
      const auto size = Alignment::template getAlignedSize< LevelTag::value >( sizes );
      return ( size + 2 * overlap )
             * StorageSizeGetter< SizesHolder, Alignment, Overlaps, IndexTag< LevelTag::value - 1 > >::get( sizes );
   }

   template< typename Permutation >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getPermuted( const SizesHolder& sizes, Permutation )
   {
      static constexpr std::size_t idx = __ndarray_impl::get< LevelTag::value >( Permutation{} );
      static constexpr std::size_t overlap = __ndarray_impl::get< idx >( Overlaps{} );
      const auto size = Alignment::template getAlignedSize< idx >( sizes );
      return ( size + 2 * overlap )
             * StorageSizeGetter< SizesHolder, Alignment, Overlaps, IndexTag< LevelTag::value - 1 > >::get( sizes );
   }
};

template< typename SizesHolder, typename Alignment, typename Overlaps >
struct StorageSizeGetter< SizesHolder, Alignment, Overlaps, IndexTag< 0 > >
{
   static typename SizesHolder::IndexType
   __cuda_callable__
   get( const SizesHolder& sizes )
   {
      static constexpr std::size_t overlap = __ndarray_impl::get< 0 >( Overlaps{} );
      return Alignment::template getAlignedSize< 0 >( sizes ) + 2 * overlap;
   }

   template< typename Permutation >
   __cuda_callable__
   static typename SizesHolder::IndexType
   getPermuted( const SizesHolder& sizes, Permutation )
   {
      static constexpr std::size_t idx = __ndarray_impl::get< 0 >( Permutation{} );
      static constexpr std::size_t overlap = __ndarray_impl::get< idx >( Overlaps{} );
      return Alignment::template getAlignedSize< idx >( sizes ) + 2 * overlap;
   }
};


// Static storage size without alignment, used in StaticNDArray
template< typename SizesHolder,
          typename LevelTag = IndexTag< SizesHolder::getDimension() - 1 > >
struct StaticStorageSizeGetter
{
   constexpr static std::size_t get()
   {
      return SizesHolder::template getStaticSize< LevelTag::value >() *
             StaticStorageSizeGetter< SizesHolder, IndexTag< LevelTag::value - 1 > >::get();
   }
};

template< typename SizesHolder >
struct StaticStorageSizeGetter< SizesHolder, IndexTag< 0 > >
{
   constexpr static std::size_t get()
   {
      return SizesHolder::template getStaticSize< 0 >();
   }
};


template< std::size_t level = 0,
          typename SizesHolder,
          typename Index,
          typename... IndexTypes >
void setSizesHelper( SizesHolder& holder,
                     Index&& size,
                     IndexTypes&&... otherSizes )
{
   holder.template setSize< level >( std::forward< Index >( size ) );
   setSizesHelper< level + 1 >( holder, std::forward< IndexTypes >( otherSizes )... );
}

template< std::size_t level = 0,
          typename SizesHolder,
          typename Index >
void setSizesHelper( SizesHolder& holder,
                     Index&& size )
{
   holder.template setSize< level >( std::forward< Index >( size ) );
}


// A variadic bounds-checker for indices
template< typename SizesHolder, typename Overlaps >
__cuda_callable__
void assertIndicesInBounds( const SizesHolder&, const Overlaps& overlaps )
{}

template< typename SizesHolder,
          typename Overlaps,
          typename Index,
          typename... IndexTypes >
__cuda_callable__
void assertIndicesInBounds( const SizesHolder& sizes, const Overlaps& overlaps, Index&& i, IndexTypes&&... indices )
{
#ifndef NDEBUG
   // sizes.template getSize<...>() cannot be inside the assert macro, but the variables
   // shouldn't be declared when compiling without assertions
   constexpr std::size_t level = SizesHolder::getDimension() - sizeof...(indices) - 1;
   const auto size = sizes.template getSize< level >();
   const decltype(size) overlap = get<level>( overlaps );
   TNL_ASSERT_LE( - overlap, (decltype(size)) i, "Input error - some index is below the lower bound." );
   TNL_ASSERT_LT( (decltype(size)) i, size + overlap, "Input error - some index is above the upper bound." );
#endif
   assertIndicesInBounds( sizes, overlaps, std::forward< IndexTypes >( indices )... );
}


// A variadic bounds-checker for distributed indices with overlaps
template< typename SizesHolder1, typename SizesHolder2, typename Overlaps >
__cuda_callable__
void assertIndicesInRange( const SizesHolder1&, const SizesHolder2&, const Overlaps& )
{}

template< typename SizesHolder1,
          typename SizesHolder2,
          typename Overlaps,
          typename Index,
          typename... IndexTypes >
__cuda_callable__
void assertIndicesInRange( const SizesHolder1& begins, const SizesHolder2& ends, const Overlaps& overlaps, Index&& i, IndexTypes&&... indices )
{
   static_assert( SizesHolder1::getDimension() == SizesHolder2::getDimension(),
                  "Inconsistent begins and ends." );
#ifndef NDEBUG
   // sizes.template getSize<...>() cannot be inside the assert macro, but the variables
   // shouldn't be declared when compiling without assertions
   constexpr std::size_t level = SizesHolder1::getDimension() - sizeof...(indices) - 1;
   const auto begin = begins.template getSize< level >();
   const auto end = ends.template getSize< level >();
   TNL_ASSERT_LE( begin - (decltype(begin)) get<level>( overlaps ), i, "Input error - some index is below the lower bound." );
   TNL_ASSERT_LT( i, end + (decltype(end)) get<level>( overlaps ), "Input error - some index is above the upper bound." );
#endif
   assertIndicesInRange( begins, ends, overlaps, std::forward< IndexTypes >( indices )... );
}


// helper for the assignment operator in NDArray
template< typename TargetHolder,
          typename SourceHolder,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesCopyHelper
{
   static void copy( TargetHolder& target,
                     const SourceHolder& source )
   {
      if( target.template getStaticSize< level >() == 0 ) {
         target.template setSize< level >( source.template getSize< level >() );
         SetSizesCopyHelper< TargetHolder, SourceHolder, level - 1 >::copy( target, source );
      }
      else if( source.template getSize< level >() < 0 ||
               target.template getStaticSize< level >() != (std::size_t) source.template getSize< level >() )
         throw std::logic_error( "Cannot copy sizes due to inconsistent underlying types (static sizes don't match)." );
   }
};

template< typename TargetHolder,
          typename SourceHolder >
struct SetSizesCopyHelper< TargetHolder, SourceHolder, 0 >
{
   static void copy( TargetHolder& target,
                     const SourceHolder& source )
   {
      if( target.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( source.template getSize< 0 >() );
      else if( source.template getSize< 0 >() < 0 ||
               target.template getStaticSize< 0 >() != (std::size_t) source.template getSize< 0 >() )
         throw std::logic_error( "Cannot copy sizes due to inconsistent underlying types (static sizes don't match)." );
   }
};


// helper for the assignment operator in NDArrayView
template< typename SizesHolder1,
          typename SizesHolder2 >
__cuda_callable__
bool sizesWeakCompare( const SizesHolder1& sizes1, const SizesHolder2& sizes2 )
{
   static_assert( SizesHolder1::getDimension() == SizesHolder2::getDimension(),
                  "Cannot compare sizes of different dimensions." );
   bool result = true;
   Algorithms::staticFor< std::size_t, 0, SizesHolder1::getDimension() >(
      [&result, &sizes1, &sizes2] ( auto level ) {
         result = result && sizes1.template getSize< level >() == sizes2.template getSize< level >();
      }
   );
   return result;
}


// helper for the forInternal and forBoundary methods (NDArray and DistributedNDArray)
template< std::size_t ConstValue,
          typename TargetHolder,
          typename SourceHolder,
          typename Overlaps = make_constant_index_sequence< TargetHolder::getDimension(), 0 >,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesSubtractHelper
{
   static void subtract( TargetHolder& target,
                         const SourceHolder& source )
   {
      if( source.template getStaticSize< level >() == 0 )
         target.template setSize< level >( source.template getSize< level >() - ConstValue * ! get< level >( Overlaps{} ) );
      SetSizesSubtractHelper< ConstValue, TargetHolder, SourceHolder, Overlaps, level - 1 >::subtract( target, source );
   }
};

template< std::size_t ConstValue,
          typename TargetHolder,
          typename SourceHolder,
          typename Overlaps >
struct SetSizesSubtractHelper< ConstValue, TargetHolder, SourceHolder, Overlaps, 0 >
{
   static void subtract( TargetHolder& target,
                         const SourceHolder& source )
   {
      if( source.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( source.template getSize< 0 >() - ConstValue * ! get< 0 >( Overlaps{} ) );
   }
};


// helper for the forInternal and forBoundary methods (DistributedNDArray)
template< std::size_t ConstValue,
          typename TargetHolder,
          typename SourceHolder,
          typename Overlaps = make_constant_index_sequence< TargetHolder::getDimension(), 0 >,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesAddHelper
{
   static void add( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< level >() == 0 )
         target.template setSize< level >( source.template getSize< level >() + ConstValue * ! get< level >( Overlaps{} ) );
      SetSizesAddHelper< ConstValue, TargetHolder, SourceHolder, Overlaps, level - 1 >::add( target, source );
   }
};

template< std::size_t ConstValue,
          typename TargetHolder,
          typename SourceHolder,
          typename Overlaps >
struct SetSizesAddHelper< ConstValue, TargetHolder, SourceHolder, Overlaps, 0 >
{
   static void add( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( source.template getSize< 0 >() + ConstValue * ! get< 0 >( Overlaps{} ) );
   }
};


// helper for the forLocalInternal, forLocalBoundary and forOverlaps methods (DistributedNDArray)
template< typename TargetHolder,
          typename SourceHolder,
          typename Overlaps = make_constant_index_sequence< TargetHolder::getDimension(), 0 >,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesSubtractOverlapsHelper
{
   static void subtract( TargetHolder& target,
                         const SourceHolder& source )
   {
      if( source.template getStaticSize< level >() == 0 )
         target.template setSize< level >( source.template getSize< level >() - get< level >( Overlaps{} ) );
      SetSizesSubtractOverlapsHelper< TargetHolder, SourceHolder, Overlaps, level - 1 >::subtract( target, source );
   }
};

template< typename TargetHolder,
          typename SourceHolder,
          typename Overlaps >
struct SetSizesSubtractOverlapsHelper< TargetHolder, SourceHolder, Overlaps, 0 >
{
   static void subtract( TargetHolder& target,
                         const SourceHolder& source )
   {
      if( source.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( source.template getSize< 0 >() - get< 0 >( Overlaps{} ) );
   }
};


// helper for the forLocalInternal, forLocalBoundary and forOverlaps methods (DistributedNDArray)
template< typename TargetHolder,
          typename SourceHolder,
          typename Overlaps = make_constant_index_sequence< TargetHolder::getDimension(), 0 >,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesAddOverlapsHelper
{
   static void add( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< level >() == 0 )
         target.template setSize< level >( source.template getSize< level >() + get< level >( Overlaps{} ) );
      SetSizesAddOverlapsHelper< TargetHolder, SourceHolder, Overlaps, level - 1 >::add( target, source );
   }
};

template< typename TargetHolder,
          typename SourceHolder,
          typename Overlaps >
struct SetSizesAddOverlapsHelper< TargetHolder, SourceHolder, Overlaps, 0 >
{
   static void add( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( source.template getSize< 0 >() + get< 0 >( Overlaps{} ) );
   }
};


// helper for the forInternal method (DistributedNDArray)
template< typename TargetHolder,
          typename SourceHolder,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesMaxHelper
{
   static void max( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< level >() == 0 )
         target.template setSize< level >( std::max( target.template getSize< level >(), source.template getSize< level >() ) );
      SetSizesMaxHelper< TargetHolder, SourceHolder, level - 1 >::max( target, source );
   }
};

template< typename TargetHolder,
          typename SourceHolder >
struct SetSizesMaxHelper< TargetHolder, SourceHolder, 0 >
{
   static void max( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( std::max( target.template getSize< 0 >(), source.template getSize< 0 >() ) );
   }
};


// helper for the forInternal method (DistributedNDArray)
template< typename TargetHolder,
          typename SourceHolder,
          std::size_t level = TargetHolder::getDimension() - 1 >
struct SetSizesMinHelper
{
   static void min( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< level >() == 0 )
         target.template setSize< level >( std::min( target.template getSize< level >(), source.template getSize< level >() ) );
      SetSizesMinHelper< TargetHolder, SourceHolder, level - 1 >::min( target, source );
   }
};

template< typename TargetHolder,
          typename SourceHolder >
struct SetSizesMinHelper< TargetHolder, SourceHolder, 0 >
{
   static void min( TargetHolder& target,
                    const SourceHolder& source )
   {
      if( source.template getStaticSize< 0 >() == 0 )
         target.template setSize< 0 >( std::min( target.template getSize< 0 >(), source.template getSize< 0 >() ) );
   }
};

} // namespace __ndarray_impl
} // namespace Containers
} // namespace noa::TNL
