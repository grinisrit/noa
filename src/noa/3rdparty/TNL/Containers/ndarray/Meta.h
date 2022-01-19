// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <utility>
#include <initializer_list>

namespace TNL {
namespace Containers {
namespace __ndarray_impl {

/*
 * Generic function to get the N-th element from a variadic pack.
 * Reference:
 * http://stackoverflow.com/questions/20162903/template-parameter-packs-access-nth-type-and-nth-element/37836252#37836252
 */
template< std::size_t index, typename T, typename... Ts,
          typename = typename std::enable_if< index == 0 >::type >
constexpr T
get_from_pack( T&& arg, Ts&&... args )
{
   return arg;
}

template< std::size_t index, typename T, typename... Ts,
          typename = typename std::enable_if< (index > 0) && index <= sizeof...( Ts ) >::type >
constexpr auto
get_from_pack( T&& arg, Ts&&... args )
{
   return get_from_pack< index-1 >( std::forward< Ts >( args )... );
}

// complementary specialization for getting a more readable compilation error
// in case calling get with a bad index
template< long long index, typename T, typename... Ts,
          typename = typename std::enable_if< (index < 0) || (index > sizeof...( Ts )) >::type >
constexpr T
get_from_pack( T&& arg, Ts&&... args )
{
   static_assert( index >= 0 && index <= sizeof...( Ts ),
                  "invalid index passed to the get function" );
   return arg;
}


// Get N-th element from std::integer_sequence.
template< std::size_t N, typename Index, Index... vals >
constexpr Index
get( std::integer_sequence< Index, vals... > )
{
   return get_from_pack< N >( vals... );
}


// Test if a variadic pack contains a value.
template< typename Index, typename T >
constexpr bool
is_in_pack( Index value, T&& pack_value )
{
   return value == pack_value;
}

template< typename Index, typename T, typename... Ts >
constexpr bool
is_in_pack( Index value, T&& pack_value, Ts&&... vals )
{
   if( value == pack_value )
      return true;
   return is_in_pack( value, std::forward< Ts >( vals )... );
}


// Test if an std::integer_sequence contains an element.
template< typename Index, Index... vals >
constexpr bool
is_in_sequence( Index value, std::integer_sequence< Index, vals... > )
{
   return is_in_pack( value, vals... );
}


// Get index of the first occurrence of value in a variadic pack.
template< typename V >
constexpr std::size_t
index_in_pack( V&& value )
{
   return 0;
}

template< typename V, typename T, typename... Ts >
constexpr std::size_t
index_in_pack( V&& value, T&& arg, Ts&&... args )
{
   if( value == arg )
      return 0;
   return 1 + index_in_pack( value, std::forward< Ts >( args )... );
}


// Get index of the first occurrence of value in a std::integer_sequence
template< typename V, typename Index, Index... vals >
constexpr std::size_t
index_in_sequence( V&& value, std::integer_sequence< Index, vals... > )
{
   return index_in_pack( std::forward< V >( value ), vals... );
}


/*
 * Generic function to concatenate an arbitrary number of std::integer_sequence instances.
 * Useful mainly for getting the type of the resulting sequence with `decltype`.
 */
// concatenate a single, potentially empty sequence
template< typename Index, Index... s >
constexpr auto
concat_sequences( std::integer_sequence< Index, s... > )
{
   return std::integer_sequence< Index, s... >{};
}

// concatenate two sequences, each potentially empty
template< typename Index, Index... s, Index... t>
constexpr auto
concat_sequences( std::integer_sequence< Index, s... >, std::integer_sequence< Index, t... > )
{
   return std::integer_sequence< Index, s... , t... >{};
}

// concatenate more than 2 sequences
template< typename Index, Index... s, Index... t, typename... R >
constexpr auto
concat_sequences( std::integer_sequence< Index, s... >, std::integer_sequence< Index, t...>, R... )
{
   return concat_sequences( std::integer_sequence< Index, s..., t... >{}, R{}... );
}


// Integer wrapper necessary for C++ templates specializations.
// As the C++ standard says:
//    A partially specialized non-type argument expression shall not involve
//    a template parameter of the partial specialization except when the argument
//    expression is a simple identifier.
template< std::size_t v >
struct IndexTag
{
   static constexpr std::size_t value = v;
};


template< typename Permutation,
          typename Sequence >
struct CallPermutationHelper
{};

template< typename Permutation,
          std::size_t... N >
struct CallPermutationHelper< Permutation, std::index_sequence< N... > >
{
   template< typename Func,
             typename... Args >
   static constexpr auto apply( Func&& f, Args&&... args ) -> decltype(auto)
   {
      return std::forward< Func >( f )( get_from_pack<
                  get< N >( Permutation{} )
                >( std::forward< Args >( args )... )... );
   }
};

// Call specified function with permuted arguments.
// [used in ndarray_operations.h]
template< typename Permutation,
          typename Func,
          typename... Args >
constexpr auto
call_with_permuted_arguments( Func&& f, Args&&... args ) -> decltype(auto)
{
   return CallPermutationHelper< Permutation, std::make_index_sequence< sizeof...( Args ) > >
          ::apply( std::forward< Func >( f ), std::forward< Args >( args )... );
}


template< typename Permutation,
          typename Sequence >
struct CallInversePermutationHelper
{};

template< typename Permutation,
          std::size_t... N >
struct CallInversePermutationHelper< Permutation, std::index_sequence< N... > >
{
   template< typename Func,
             typename... Args >
   static constexpr auto apply( Func&& f, Args&&... args ) -> decltype(auto)
   {
      return std::forward< Func >( f )( get_from_pack<
                  index_in_sequence( N, Permutation{} )
                >( std::forward< Args >( args )... )... );
   }
};

// Call specified function with permuted arguments.
// [used in ndarray_operations.h]
template< typename Permutation,
          typename Func,
          typename... Args >
constexpr auto
call_with_unpermuted_arguments( Func&& f, Args&&... args ) -> decltype(auto)
{
   return CallInversePermutationHelper< Permutation, std::make_index_sequence< sizeof...( Args ) > >
          ::apply( std::forward< Func >( f ), std::forward< Args >( args )... );
}


// Check that all elements of the initializer list are equal to the specified value.
// [used in ndarray_operations.h]
constexpr bool
all_elements_equal_to_value( std::size_t value, std::initializer_list< std::size_t > list )
{
   for( auto elem : list )
      if( elem != value )
         return false;
   return true;
}


// Check that all elements of the initializer list are in the specified range [begin, end).
// [used in ndarray.h -- static assertions on permutations]
constexpr bool
all_elements_in_range( std::size_t begin, std::size_t end, std::initializer_list< std::size_t > list )
{
   for( auto elem : list )
      if( elem < begin || elem >= end )
         return false;
   return true;
}


// Check that the elements of the initializer list form an increasing sequence.
// [used in ndarray.h -- static assertion in getSubarrayView()]
constexpr bool
is_increasing_sequence( std::initializer_list< std::size_t > list )
{
   std::size_t prev = *list.begin();
   for( auto& elem : list ) {
      if( &elem == list.begin() )
         continue;
      if( elem <= prev )
         return false;
      prev = elem;
   }
   return true;
}


// Count elements of a variadic pack smaller than a specified value
// [used in ndarray_subarray.h to generate a subpermutation]
template< typename T, typename V >
constexpr std::size_t
count_smaller( T threshold, V&& value )
{
   return value < threshold ? 1 : 0;
}

template< typename T, typename V, typename... Values >
constexpr std::size_t
count_smaller( T threshold, V&& value, Values&&... vals )
{
   if( value < threshold )
      return 1 + count_smaller( threshold, vals... );
   return count_smaller( threshold, vals... );
}


// C++17 version using "if constexpr" and a general predicate (lambda function)
// Reference: https://stackoverflow.com/a/41723705
//template< typename Index, Index a, typename Predicate >
//constexpr auto
//FilterSingle( std::integer_sequence< Index, a >, Predicate pred )
//{
//   if constexpr (pred(a))
//      return std::integer_sequence< Index, a >{};
//   else
//      return std::integer_sequence< Index >{};
//}
//
//// empty sequence case
//template< typename Index, typename Predicate >
//constexpr auto
//filter_sequence( std::integer_sequence< Index >, [[maybe_unused]] Predicate pred )
//{
//   return std::integer_sequence< Index >{};
//}
//
//// non empty sequence case
//template< typename Index, Index... vals, typename Predicate >
//constexpr auto
//filter_sequence( std::integer_sequence< Index, vals... >, [[maybe_unused]] Predicate pred )
//{
//   return concat_sequences( FilterSingle( std::integer_sequence< Index, vals >{}, pred )... );
//}

// C++14 version, with hard-coded predicate
template< typename Mask, typename Index, Index val >
constexpr typename std::conditional_t< is_in_sequence( val, Mask{} ),
                                       std::integer_sequence< Index, val >,
                                       std::integer_sequence< Index > >
FilterSingle( std::integer_sequence< Index, val > )
{
   return {};
}

/*
 * Generic function returning a subsequence of a sequence obtained by omitting
 * the elements not contained in the specified mask.
 */
// empty sequence case
template< typename Mask, typename Index >
constexpr auto
filter_sequence( std::integer_sequence< Index > )
{
   return std::integer_sequence< Index >{};
}

// non empty sequence case
template< typename Mask, typename Index, Index... vals >
constexpr auto
filter_sequence( std::integer_sequence< Index, vals... > )
{
   return concat_sequences( FilterSingle< Mask >( std::integer_sequence< Index, vals >{} )... );
}


/*
 * make_constant_integer_sequence, make_constant_index_sequence - helper
 * templates for the generation of constant sequences like
 * std::make_integer_sequence, std::make_index_sequence
 */
template< typename T, typename N, T v > struct gen_const_seq;
template< typename T, typename N, T v > using gen_const_seq_t = typename gen_const_seq< T, N, v >::type;

template< typename T, typename N, T v >
struct gen_const_seq
{
   using type = decltype(concat_sequences(
                     gen_const_seq_t<T, std::integral_constant<T, N::value/2>, v>{},
                     gen_const_seq_t<T, std::integral_constant<T, N::value - N::value/2>, v>{}
                  ));
};

template< typename T, T v >
struct gen_const_seq< T, std::integral_constant<T, 0>, v >
{
   using type = std::integer_sequence<T>;
};

template< typename T, T v >
struct gen_const_seq< T, std::integral_constant<T, 1>, v >
{
   using type = std::integer_sequence<T, v>;
};

template< typename T, T N, T value >
using make_constant_integer_sequence = gen_const_seq_t< T, std::integral_constant<T, N>, value >;

template< std::size_t N, std::size_t value >
using make_constant_index_sequence = gen_const_seq_t< std::size_t, std::integral_constant<std::size_t, N>, value >;

} // namespace __ndarray_impl
} // namespace Containers
} // namespace TNL
