// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Containers/ndarray/Meta.h>
#include <noa/3rdparty/TNL/Containers/ndarray/SizesHolder.h>
#include <noa/3rdparty/TNL/Containers/ndarray/Indexing.h>

namespace noa::TNL {
namespace Containers {
namespace __ndarray_impl {

template< typename Dimensions, typename Permutation >
class SubpermutationGetter;

template< std::size_t... dims, std::size_t... vals >
class SubpermutationGetter< std::index_sequence< dims... >, std::index_sequence< vals... > >
{
private:
   using Dimensions = std::index_sequence< dims... >;
   using Permutation = std::index_sequence< vals... >;
   using Subsequence = decltype(
            filter_sequence< Dimensions >( Permutation{} )
         );

   template< std::size_t... v >
   static constexpr auto
   get_subpermutation( std::index_sequence< v... > )
   {
      using Subpermutation = std::index_sequence< count_smaller( v, v... )... >;
      return Subpermutation{};
   }

public:
   using Subpermutation = decltype(
            get_subpermutation( Subsequence{} )
         );
};


template< typename Dimensions, typename SihesHolder >
class SizesFilter;

template< std::size_t... dims, typename Index, std::size_t... sizes >
class SizesFilter< std::index_sequence< dims... >, SizesHolder< Index, sizes... > >
{
private:
   using Dimensions = std::index_sequence< dims... >;
   using SizesSequence = std::index_sequence< sizes... >;
   using Subsequence = decltype(
            concat_sequences( std::index_sequence< get_from_pack< dims >( sizes... ) >{} ... )
         );

   template< std::size_t... v >
   static constexpr auto
   get_sizesholder( std::index_sequence< v... > )
   {
      using Sizes = SizesHolder< Index, v... >;
      return Sizes{};
   }

   template< std::size_t level = 0, typename = void >
   struct SizeSetterHelper
   {
      template< typename NewSizes,
                typename OldSizes >
      __cuda_callable__
      static void setSizes( NewSizes& newSizes,
                            const OldSizes& oldSizes )
      {
         if( oldSizes.template getStaticSize< level >() == 0 )
            newSizes.template setSize< level >( oldSizes.template getSize< get< level >( Dimensions{} ) >() );
         SizeSetterHelper< level + 1 >::setSizes( newSizes, oldSizes );
      }
   };

   template< typename _unused >
   struct SizeSetterHelper< Dimensions::size() - 1, _unused >
   {
      template< typename NewSizes,
                typename OldSizes >
      __cuda_callable__
      static void setSizes( NewSizes& newSizes,
                            const OldSizes& oldSizes )
      {
         static constexpr std::size_t level = Dimensions::size() - 1;
         if( oldSizes.template getStaticSize< level >() == 0 )
            newSizes.template setSize< level >( oldSizes.template getSize< get< level >( Dimensions{} ) >() );
      }
   };

   template< std::size_t level = 0, typename = void >
   struct IndexChecker
   {
      template< typename... IndexTypes >
      static bool check( IndexTypes&&... indices )
      {
         static constexpr std::size_t d = get< level >( Dimensions{} );
         if( get_from_pack< d >( std::forward< IndexTypes >( indices )... ) != 0 )
            return false;
         return IndexChecker< level + 1 >::check( std::forward< IndexTypes >( indices )... );
      }
   };

   template< typename _unused >
   struct IndexChecker< Dimensions::size() - 1, _unused >
   {
      template< typename... IndexTypes >
      static bool check( IndexTypes&&... indices )
      {
         static constexpr std::size_t d = get< Dimensions::size() - 1 >( Dimensions{} );
         if( get_from_pack< d >( std::forward< IndexTypes >( indices )... ) != 0 )
            return false;
         return true;
      }
   };

public:
   using Sizes = decltype(
            get_sizesholder( Subsequence{} )
         );

   template< typename... IndexTypes >
   __cuda_callable__
   static Sizes filterSizes( const SizesHolder< Index, sizes... >& oldSizes, IndexTypes&&... indices )
   {
      Sizes newSizes;

      // assert that indices are 0 for the dimensions in the subarray
      // (contraction of dimensions is not supported yet, and it does not
      // make sense for static dimensions anyway)
      TNL_ASSERT_TRUE( IndexChecker<>::check( std::forward< IndexTypes >( indices )... ),
                       "Static dimensions of the subarray must start at index 0 of the array." );

      // set dynamic sizes
      // pseudo-python-code:
      //      for d, D in enumerate(dims...):
      //          newSizes.setSize< d >( oldSizes.getSize< D >() )
      SizeSetterHelper<>::setSizes( newSizes, oldSizes );

      return newSizes;
   }
};


template< typename Index, std::size_t Dimension >
struct DummyStrideBase
{
   static constexpr std::size_t getDimension()
   {
      return Dimension;
   }

   static constexpr bool isContiguous()
   {
      return true;
   }

   template< std::size_t level >
   __cuda_callable__
   constexpr Index getStride( Index i = 0 ) const
   {
      return 1;
   }
};

template< typename Index,
          std::size_t... sizes >
class StridesHolder
: private SizesHolder< Index, sizes... >
{
   using BaseType = SizesHolder< Index, sizes... >;

public:
   using BaseType::getDimension;

   static constexpr bool isContiguous()
   {
      // a priori not contiguous (otherwise DummyStrideBase would be used)
      return false;
   }

   template< std::size_t level >
   static constexpr std::size_t getStaticStride( Index i = 0 )
   {
      return BaseType::template getStaticSize< level >();
   }

   template< std::size_t level >
   __cuda_callable__
   Index getStride( Index i = 0 ) const
   {
      return BaseType::template getSize< level >();
   }

   template< std::size_t level >
   __cuda_callable__
   void setStride( Index size )
   {
      BaseType::template setSize< level >( size );
   }
};

template< typename Base, typename Permutation, std::size_t... Dimensions >
class SubarrayGetter;

template< typename SliceInfo, typename Permutation, std::size_t... Dimensions >
class SubarrayGetter< NDArrayBase< SliceInfo >, Permutation, Dimensions... >
{
   // returns the number of factors in the stride product
   template< std::size_t dim, std::size_t... vals >
   static constexpr std::size_t get_end( std::index_sequence< vals... > _perm )
   {
      if( dim == get< Permutation::size() - 1 >( Permutation{} ) )
         return 0;
      std::size_t i = 0;
      std::size_t count = 0;
// FIXME: nvcc chokes on the variadic brace-initialization
#ifndef __NVCC__
      for( auto v : std::initializer_list< std::size_t >{ vals... } )
#else
      for( auto v : (std::size_t [sizeof...(vals)]){ vals... } )
#endif
      {
         if( i++ <= index_in_pack( dim, vals... ) )
            continue;
         if( is_in_sequence( v, std::index_sequence< Dimensions... >{} ) )
            break;
         count++;
      }
      return count;
   }

   // static calculation of the stride product
   template< typename SizesHolder,
             std::size_t start_dim,
             std::size_t end = get_end< start_dim >( Permutation{} ),
             std::size_t level = 0,
             typename = void >
   struct StaticStrideGetter
   {
      static constexpr std::size_t get()
      {
         constexpr std::size_t start_offset = index_in_sequence( start_dim, Permutation{} );
         constexpr std::size_t dim = __ndarray_impl::get< start_offset + level + 1 >( Permutation{} );
         return SizesHolder::template getStaticSize< dim >() * StaticStrideGetter< SizesHolder, start_dim, end, level + 1 >::get();
      }
   };

   template< typename SizesHolder, std::size_t start_dim, std::size_t end, typename _unused >
   struct StaticStrideGetter< SizesHolder, start_dim, end, end, _unused >
   {
      static constexpr std::size_t get()
      {
         return 1;
      }
   };

   // dynamic calculation of the stride product
   template< std::size_t start_dim,
             std::size_t end = get_end< start_dim >( Permutation{} ),
             std::size_t level = 0,
             typename = void >
   struct DynamicStrideGetter
   {
      template< typename SizesHolder >
      static constexpr std::size_t get( const SizesHolder& sizes )
      {
         constexpr std::size_t start_offset = index_in_sequence( start_dim, Permutation{} );
         constexpr std::size_t dim = __ndarray_impl::get< start_offset + level + 1 >( Permutation{} );
         return sizes.template getSize< dim >() * DynamicStrideGetter< start_dim, end, level + 1 >::get( sizes );
      }
   };

   template< std::size_t start_dim, std::size_t end, typename _unused >
   struct DynamicStrideGetter< start_dim, end, end, _unused >
   {
      template< typename SizesHolder >
      static constexpr std::size_t get( const SizesHolder& sizes )
      {
         return 1;
      }
   };

   // helper class for setting dynamic strides
   template< std::size_t level = 0, typename = void >
   struct StrideSetterHelper
   {
      template< typename StridesHolder, typename SizesHolder >
      __cuda_callable__
      static void setStrides( StridesHolder& strides, const SizesHolder& sizes )
      {
         static constexpr std::size_t dim = get_from_pack< level >( Dimensions... );
         if( StridesHolder::template getStaticStride< level >() == 0 )
            strides.template setStride< level >( DynamicStrideGetter< dim >::get( sizes ) );
         StrideSetterHelper< level + 1 >::setStrides( strides, sizes );
      }
   };

   template< typename _unused >
   struct StrideSetterHelper< sizeof...(Dimensions) - 1, _unused >
   {
      template< typename StridesHolder, typename SizesHolder >
      __cuda_callable__
      static void setStrides( StridesHolder& strides, const SizesHolder& sizes )
      {
         static constexpr std::size_t level = sizeof...(Dimensions) - 1;
         static constexpr std::size_t dim = get_from_pack< level >( Dimensions... );
         if( StridesHolder::template getStaticStride< level >() == 0 )
            strides.template setStride< level >( DynamicStrideGetter< dim >::get( sizes ) );
      }
   };

public:
   using Subpermutation = typename SubpermutationGetter< std::index_sequence< Dimensions... >, Permutation >::Subpermutation;

   template< typename SizesHolder, typename... IndexTypes >
   __cuda_callable__
   static auto filterSizes( const SizesHolder& sizes, IndexTypes&&... indices )
   {
      using Filter = SizesFilter< std::index_sequence< Dimensions... >, SizesHolder >;
      return Filter::filterSizes( sizes, std::forward< IndexTypes >( indices )... );
   }

   template< typename SizesHolder, typename... IndexTypes >
   __cuda_callable__
   static auto getStrides( const SizesHolder& sizes, IndexTypes&&... indices )
   {
      using Strides = StridesHolder< typename SizesHolder::IndexType,
                                     StaticStrideGetter< SizesHolder, Dimensions >::get()... >;
      Strides strides;

      // set dynamic strides
      // pseudo-python-code:
      //      for i, d in enumerate(Dimensions):
      //          if is_dynamic_dimension(d):
      //              strides.setStride< i >( dynamic_stride(d, sizes) )
      StrideSetterHelper<>::setStrides( strides, sizes );

      return strides;
   }
};

} // namespace __ndarray_impl
} // namespace Containers
} // namespace noa::TNL
