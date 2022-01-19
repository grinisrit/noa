// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Xuan Thang Nguyen

#pragma once

#include <TNL/Algorithms/Sorting/DefaultSorter.h>

namespace TNL {
   namespace Algorithms {

/**
 * \brief Function for sorting elements of array or vector in ascending order.
 *
 * \tparam Array is a type of container to be sorted. It can be, for example, \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *    \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView.
 * \tparam Sorter is an algorithm for sorting. It can be, \ref TNL::Algorithms::Sorting::STLSort for sorting on host and \ref TNL::Algorithms::Sorting::Quicksort
 *    or \ref TNL::Algorithms::Sorting::BitonicSort for sorting on CUDA GPU.
 *
 * \param array is an instance of array/array view/vector/vector view for sorting.
 * \param sorter is an instance of sorter.
 *
 * \par Example
 *
 * \includelineno SortingExample.cpp
 *
 * \par Output
 *
 * \include SortingExample.out
 *
 */
template< typename Array,
          typename Sorter = typename Sorting::DefaultSorter< typename Array::DeviceType >::SorterType >
void ascendingSort( Array& array, const Sorter& sorter = Sorter{} )
{
   using ValueType = typename Array::ValueType;
   sorter.sort( array, [] __cuda_callable__ ( const ValueType& a, const ValueType& b ) { return a < b; } );
}

/**
 * \brief Function for sorting elements of array or vector in descending order.
 *
 * \tparam Array is a type of container to be sorted. It can be, for example, \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *    \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView.
 * \tparam Sorter is an algorithm for sorting. It can be, \ref TNL::Algorithms::Sorting::STLSort for sorting on host and \ref TNL::Algorithms::Sorting::Quicksort
 *    or \ref TNL::Algorithms::Sorting::BitonicSort for sorting on CUDA GPU.
 *
 * \param array is an instance of array/array view/vector/vector view for sorting.
 * \param sorter is an instance of sorter.
 *
 * \par Example
 *
 * \includelineno SortingExample.cpp
 *
 * \par Output
 *
 * \include SortingExample.out
 *
 */
template< typename Array,
          typename Sorter = typename Sorting::DefaultSorter< typename Array::DeviceType >::SorterType >
void descendingSort( Array& array, const Sorter& sorter = Sorter{} )
{
   using ValueType = typename Array::ValueType;
   sorter.sort( array, [] __cuda_callable__ ( const ValueType& a, const ValueType& b ) { return a < b; } );
}

/**
 * \brief Function for sorting elements of array or vector based on a user defined comparison lambda function.
 *
 * \tparam Array is a type of container to be sorted. It can be, for example, \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *    \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView.
 * \tparam Compare is a lambda function for comparing of two elements. It returns true if the first argument should be ordered before the second. The
 *    lambda function is supposed to be defined as follows (`ValueType` is type of the array elements):
 *  ```
 *  auto compare = [] __cuda_callable__ ( const ValueType& a , const ValueType& b ) -> bool { return .... };
 *  ```
 * \tparam Sorter is an algorithm for sorting. It can be, \ref TNL::Algorithms::Sorting::STLSort for sorting on host and \ref TNL::Algorithms::Sorting::Quicksort
 *    or \ref TNL::Algorithms::Sorting::BitonicSort for sorting on CUDA GPU.
 *
 * \param array is an instance of array/array view/vector/vector view for sorting.
 * \param compare is an instance of the lambda function for comparison of two elements.
 * \param sorter is an instance of sorter.
 *
 * \par Example
 *
 * \includelineno SortingExample2.cpp
 *
 * \par Output
 *
 * \include SortingExample2.out
 *
 */
template< typename Array,
          typename Compare,
          typename Sorter = typename Sorting::DefaultSorter< typename Array::DeviceType >::SorterType >
void sort( Array& array, const Compare& compare, const Sorter& sorter = Sorter{} )
{
   sorter.sort( array, compare );
}

/**
 * \brief Function for general sorting based on lambda functions for comparison and swaping of two elements..
 *
 * \tparam Device is device on which the sorting algorithms should be executed.
 * \tparam Index is type used for indexing of the sorted data.
 * \tparam Compare is a lambda function for comparing of two elements. It returns true if the first argument should be ordered before the second - both are given
 *  by indices representing their positions. The lambda function is supposed to be defined as follows:
 *  ```
 *  auto compare = [=] __cuda_callable__ ( const Index& a , const Index& b ) -> bool { return .... };
 *  ```
 * \tparam Swap is a lambda function for swaping of two elements which are ordered wrong way. Both elements are represented by indices as well.  It supposed to
 *  be defined as:
 * ```
 * auto swap = [=] __cuda_callable__ (  const Index& a , const Index& b ) mutable { swap( ....); };
 * ```
 * \tparam Sorter is an algorithm for sorting. It can be \ref TNL::Algorithms::Sorting::BitonicSort for sorting on CUDA GPU. Currently there is no algorithm for CPU :(.
 *
 * \param array is an instance of array/array view/vector/vector view for sorting.
 * \param compare is an instance of the lambda function for comparison of two elements.
 * \param sorter is an instance of sorter.
 *
 * \par Example
 *
 * \includelineno SortingExample3.cpp
 *
 * \par Output
 *
 * \include SortingExample3.out
 *
 */
template< typename Device,
          typename Index,
          typename Compare,
          typename Swap,
          typename Sorter = typename Sorting::DefaultInplaceSorter< Device >::SorterType >
void sort( const Index begin, const Index end, Compare&& compare, Swap&& swap, const Sorter& sorter = Sorter{} )
{
   sorter.template inplaceSort< Device, Index >( begin, end, compare, swap );
}

/**
 * \brief Functions returning true if the array elements are sorted according to the lmabda function `comparison`.
 *
 * \tparam Array is the type of array/vector. It can be, for example,  \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *    \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView.
 * \tparam Compare is a lambda function for comparing of two elements. It returns true if the first argument should be ordered before the second - both are given
 *  by indices representing their positions. The lambda function is supposed to be defined as follows:
 *  ```
 *  auto compare = [=] __cuda_callable__ ( const Index& a , const Index& b ) -> bool { return .... };
 *  ```
 * \param arr is an instance of tested array.
 * \param compare is an instance of the lambda function for elements comparison.
 * \param sorter is an instance of sorter.
 *
 * \return true if the array is sorted in ascending order.
 * \return false  if the array is NOT sorted in ascending order.
 */
template< typename Array, typename Compare >
bool isSorted( const Array& arr, const Compare& compare )
{
   using Device = typename Array::DeviceType;
   if (arr.getSize() <= 1)
      return true;

   auto view = arr.getConstView();
   auto fetch = [=] __cuda_callable__(int i) { return ! compare( view[ i ], view[ i - 1 ] ); };
   auto reduction = [] __cuda_callable__(bool a, bool b) { return a && b; };
   return TNL::Algorithms::reduce< Device >( 1, arr.getSize(), fetch, reduction, true );
}

/**
 * \brief Functions returning true if the array elements are sorted in ascending order.
 *
 * \tparam Array is the type of array/vector. It can be, for example,  \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *    \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView.
 *
 * \param arr is an instance of tested array.
 *
 * \return true if the array is sorted in ascending order.
 * \return false  if the array is NOT sorted in ascending order.
 */
template< typename Array >
bool isAscending( const Array& arr )
{
   using Value = typename Array::ValueType;
   return isSorted( arr, [] __cuda_callable__( const Value& a, const Value& b ) { return a < b; });
}

/**
 * \brief Functions returning true if the array elements are sorted in descending order.
 *
 * \tparam Array is the type of array/vector. It can be, for example,  \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *    \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView.
 *
 * \param arr is an instance of tested array.
 *
 * \return true if the array is sorted in descending order.
 * \return false  if the array is NOT sorted in descending order.
 */
template< typename Array >
bool isDescending( const Array& arr)
{
   using Value = typename Array::ValueType;
   return isSorted( arr, [] __cuda_callable__( const Value& a, const Value& b ) { return a > b; });
}

   } // namespace Algorithms
} // namespace TNL
