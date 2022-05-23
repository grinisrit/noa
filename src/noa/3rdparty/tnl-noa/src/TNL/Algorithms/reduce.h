// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::pair, std::forward

#include <noa/3rdparty/tnl-noa/src/TNL/Functional.h>  // extension of STL functionals for reduction
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/detail/Reduction.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/TypeTraits.h>  // RemoveET

namespace noa::TNL {
namespace Algorithms {

/**
 * \brief \e reduce implements [(parallel) reduction](https://en.wikipedia.org/wiki/Reduce_(parallel_pattern))
 * for vectors and arrays.
 *
 * Reduction can be used for operations having one or more vectors (or arrays)
 * elements as input and returning one number (or element) as output. Some
 * examples of such operations can be vectors/arrays comparison, vector norm,
 * scalar product of two vectors or computing minimum or maximum. If one needs
 * to know even the position of the smallest or the largest element, the
 * function \ref reduceWithArgument can be used.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Result is a type of the reduction result.
 * \tparam Fetch is a lambda function for fetching the input data.
 * \tparam Reduction is a lambda function performing the reduction.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential,
 * \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation, i.e. element which does not
 *                 change the result of the reduction.
 * \return result of the reduction
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * The `reduction` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/SumExampleWithLambda.cpp
 *
 * \par Output
 *
 * \include SumExampleWithLambda.out
 */
template< typename Device, typename Index, typename Result, typename Fetch, typename Reduction >
Result
reduce( Index begin, Index end, Fetch&& fetch, Reduction&& reduction, const Result& identity )
{
   return detail::Reduction< Device >::reduce(
      begin, end, std::forward< Fetch >( fetch ), std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Variant of \ref reduce with functional instead of reduction lambda function.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Fetch is a lambda function for fetching the input data.
 * \tparam Reduction is a functional performing the reduction.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential,
 * \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \e Reduction can be one of the following \ref TNL::Plus, \ref TNL::Multiplies,
 * \ref TNL::Min, \ref TNL::Max, \ref TNL::LogicalAnd, \ref TNL::LogicalOr,
 * \ref TNL::BitAnd or \ref TNL::BitOr. \ref TNL::Plus is used by default.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation.
 * \return result of the reduction
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/SumExampleWithFunctional.cpp
 *
 * \par Output
 *
 * \include SumExampleWithFunctional.out
 */
template< typename Device, typename Index, typename Fetch, typename Reduction = TNL::Plus >
auto
reduce( Index begin, Index end, Fetch&& fetch, Reduction&& reduction = TNL::Plus{} )
{
   using Result = Containers::Expressions::RemoveET< decltype( reduction( fetch( 0 ), fetch( 0 ) ) ) >;
   return reduce< Device >( begin,
                            end,
                            std::forward< Fetch >( fetch ),
                            std::forward< Reduction >( reduction ),
                            reduction.template getIdentity< Result >() );
}

/**
 * \brief Variant of \ref reduce for arrays, views and compatible objects.
 *
 * The referenced \ref reduce function is called with:
 *
 * - `Device`, which is `typename Array::DeviceType` by default, as the `Device` type,
 * - `0` as the beginning of the interval for reduction,
 * - `array.getSize()` as the end of the interval for reduction,
 * - `array.getConstView()` as the `fetch` functor,
 * - `reduction` as the reduction operation,
 * - and `identity` as the identity element of the reduction.
 *
 * \par Example
 *
 * \include Algorithms/reduceArrayExample.cpp
 *
 * \par Output
 *
 * \include reduceArrayExample.out
 */
template< typename Array, typename Device = typename Array::DeviceType, typename Reduction, typename Result >
auto
reduce( const Array& array, Reduction&& reduction, Result identity )
{
   return reduce< Device >(
      (typename Array::IndexType) 0, array.getSize(), array.getConstView(), std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Variant of \ref reduce for arrays, views and compatible objects.
 *
 * \e Reduction can be one of the following \ref TNL::Plus, \ref TNL::Multiplies,
 * \ref TNL::Min, \ref TNL::Max, \ref TNL::LogicalAnd, \ref TNL::LogicalOr,
 * \ref TNL::BitAnd or \ref TNL::BitOr. \ref TNL::Plus is used by default.
 *
 * The referenced \ref reduce function is called with:
 *
 * - `Device`, which is `typename Array::DeviceType` by default, as the `Device` type,
 * - `0` as the beginning of the interval for reduction,
 * - `array.getSize()` as the end of the interval for reduction,
 * - `array.getConstView()` as the `fetch` functor,
 * - `reduction` as the reduction operation,
 * - and the identity element obtained from the reduction functional object.
 *
 * \par Example
 *
 * \include Algorithms/reduceArrayExample.cpp
 *
 * \par Output
 *
 * \include reduceArrayExample.out
 */
template< typename Array, typename Device = typename Array::DeviceType, typename Reduction = TNL::Plus >
auto
reduce( const Array& array, Reduction&& reduction = TNL::Plus{} )
{
   using Result = Containers::Expressions::RemoveET< decltype( reduction( array( 0 ), array( 0 ) ) ) >;
   return reduce< Array, Device >( array, std::forward< Reduction >( reduction ), reduction.template getIdentity< Result >() );
}

/**
 * \brief Variant of \ref reduce returning also the position of the element of interest.
 *
 * For example, in case of computing minimal or maximal element in array/vector,
 * the position of the element having given value can be obtained. This method
 * is, however, more flexible.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Result is a type of the reduction result.
 * \tparam Reduction is a lambda function performing the reduction.
 * \tparam Fetch is a lambda function for fetching the input data.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential,
 * \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation and managing the elements positions.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation, i.e. element which does not
 *                 change the result of the reduction.
 * \return result of the reduction in a form of std::pair< Index, Result> structure. `pair.first`
 *         is the element position and `pair.second` is the reduction result.
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * The `reduction` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/ReductionWithArgument.cpp
 *
 * \par Output
 *
 * \include ReductionWithArgument.out
 */
template< typename Device, typename Index, typename Result, typename Fetch, typename Reduction >
std::pair< Result, Index >
reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduction&& reduction, const Result& identity )
{
   return detail::Reduction< Device >::reduceWithArgument(
      begin, end, std::forward< Fetch >( fetch ), std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Variant of \ref reduceWithArgument with functional instead of reduction lambda function.
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Index is a type for indexing.
 * \tparam Result is a type of the reduction result.
 * \tparam Reduction is a functional performing the reduction.
 * \tparam Fetch is a lambda function for fetching the input data.
 *
 * \e Device can be on of the following \ref TNL::Devices::Sequential,
 * \ref TNL::Devices::Host and \ref TNL::Devices::Cuda.
 *
 * \e Reduction can be one of \ref TNL::MinWithArg, \ref TNL::MaxWithArg.
 *
 * \param begin defines range [begin, end) of indexes which will be used for the reduction.
 * \param end defines range [begin, end) of indexes which will be used for the reduction.
 * \param fetch is a lambda function fetching the input data.
 * \param reduction is a lambda function defining the reduction operation and managing the elements positions.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation, i.e. element which does not
 *                 change the result of the reduction.
 * \return result of the reduction in a form of std::pair< Index, Result> structure. `pair.first`
 *         is the element position and `pair.second` is the reduction result.
 *
 * The `fetch` lambda function takes one argument which is index of the element to be fetched:
 *
 * ```
 * auto fetch = [=] __cuda_callable__ ( Index i ) { return ... };
 * ```
 *
 * The `reduction` lambda function takes two variables which are supposed to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/ReductionWithArgumentWithFunctional.cpp
 *
 * \par Output
 *
 * \include ReductionWithArgumentWithFunctional.out
 */
template< typename Device, typename Index, typename Fetch, typename Reduction >
auto
reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduction&& reduction )
{
   using Result = Containers::Expressions::RemoveET< decltype( fetch( 0 ) ) >;
   return reduceWithArgument< Device >( begin,
                                        end,
                                        std::forward< Fetch >( fetch ),
                                        std::forward< Reduction >( reduction ),
                                        reduction.template getIdentity< Result >() );
}

/**
 * \brief Variant of \ref reduceWithArgument for arrays, views and compatible objects.
 *
 * The referenced \ref reduceWithArgument function is called with:
 *
 * - `Device`, which is `typename Array::DeviceType` by default, as the `Device` type,
 * - `0` as the beginning of the interval for reduction,
 * - `array.getSize()` as the end of the interval for reduction,
 * - `array.getConstView()` as the `fetch` functor,
 * - `reduction` as the reduction operation,
 * - and `identity` as the identity element of the reduction.
 *
 * \par Example
 *
 * \include Algorithms/reduceWithArgumentArrayExample.cpp
 *
 * \par Output
 *
 * \include reduceWithArgumentArrayExample.out
 */
template< typename Array, typename Device = typename Array::DeviceType, typename Reduction, typename Result >
auto
reduceWithArgument( const Array& array, Reduction&& reduction, Result identity )
{
   return reduceWithArgument< Device >(
      (typename Array::IndexType) 0, array.getSize(), array.getConstView(), std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Variant of \ref reduceWithArgument for arrays, views and compatible objects.
 *
 * \e Reduction can be one of \ref TNL::MinWithArg, \ref TNL::MaxWithArg.
 *
 * The referenced \ref reduceWithArgument function is called with:
 *
 * - `Device`, which is `typename Array::DeviceType` by default, as the `Device` type,
 * - `0` as the beginning of the interval for reduction,
 * - `array.getSize()` as the end of the interval for reduction,
 * - `array.getConstView()` as the `fetch` functor,
 * - `reduction` as the reduction operation,
 * - and the identity element obtained from the reduction functional object.
 *
 * \par Example
 *
 * \include Algorithms/reduceWithArgumentArrayExample.cpp
 *
 * \par Output
 *
 * \include reduceWithArgumentArrayExample.out
 */
template< typename Array, typename Device = typename Array::DeviceType, typename Reduction >
auto
reduceWithArgument( const Array& array, Reduction&& reduction )
{
   using Result = Containers::Expressions::RemoveET< decltype( array( 0 ) ) >;
   return reduceWithArgument< Array, Device >(
      array, std::forward< Reduction >( reduction ), reduction.template getIdentity< Result >() );
}

}  // namespace Algorithms
}  // namespace noa::TNL
