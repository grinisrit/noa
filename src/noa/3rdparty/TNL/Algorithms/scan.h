// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::forward

#include <TNL/Algorithms/detail/Scan.h>
#include <TNL/Functional.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief Computes an inclusive scan (or prefix sum) of an input array and
 *        stores it in an output array.
 *
 * [Inclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$s_1, \ldots, s_n\f$ defined as
 *
 * \f[
 * s_i = \sum_{j=1}^i a_i.
 * \f]
 *
 * \tparam InputArray type of the array to be scanned
 * \tparam OutputArray type of the output array
 * \tparam Reduction type of the reduction functor
 *
 * \param input the input array to be scanned
 * \param output the array where the result will be stored
 * \param begin the first element in the array to be scanned
 * \param end the last element in the array to be scanned
 * \param outputBegin the first element in the output array to be written. There
 *                    must be at least `end - begin` elements in the output
 *                    array starting at the position given by `outputBegin`.
 * \param reduction functor implementing the reduction operation
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation, i.e. element which does not
 *                 change the result of the reduction.
 *
 * The reduction functor takes two variables to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/inclusiveScanExample.cpp
 *
 * \par Output
 *
 * \include inclusiveScanExample.out
 */
template< typename InputArray,
          typename OutputArray,
          typename Reduction >
void
inclusiveScan( const InputArray& input,
               OutputArray& output,
               typename InputArray::IndexType begin,
               typename InputArray::IndexType end,
               typename OutputArray::IndexType outputBegin,
               Reduction&& reduction,
               typename OutputArray::ValueType identity )
{
   static_assert( std::is_same< typename InputArray::DeviceType, typename OutputArray::DeviceType >::value,
                  "The input and output arrays must have the same device type." );
   TNL_ASSERT_EQ( reduction( identity, identity ), identity,
                  "identity is not an identity element of the reduction operation" );
   // TODO: check if evaluating the input is expensive (e.g. a vector expression), otherwise use WriteInSecondPhase (optimal for array-to-array)
   using Scan = detail::Scan< typename OutputArray::DeviceType, detail::ScanType::Inclusive, detail::ScanPhaseType::WriteInFirstPhase >;
   Scan::perform( input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Overload of \ref inclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The [identity element](https://en.wikipedia.org/wiki/Identity_element) is
 * taken as `reduction.template getIdentity< typename OutputArray::ValueType >()`.
 * See \ref inclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `input.getSize()`.
 */
template< typename InputArray,
          typename OutputArray,
          typename Reduction = TNL::Plus >
void
inclusiveScan( const InputArray& input,
               OutputArray& output,
               typename InputArray::IndexType begin = 0,
               typename InputArray::IndexType end = 0,
               typename OutputArray::IndexType outputBegin = 0,
               Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = input.getSize();
   constexpr typename OutputArray::ValueType identity = Reduction::template getIdentity< typename OutputArray::ValueType >();
   inclusiveScan( input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Computes an exclusive scan (or prefix sum) of an input array and
 *        stores it in an output array.
 *
 * [Exclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$\sigma_1, \ldots, \sigma_n\f$ defined as
 *
 * \f[
 * \sigma_i = \sum_{j=1}^{i-1} a_i.
 * \f]
 *
 * \tparam InputArray type of the array to be scanned
 * \tparam OutputArray type of the output array
 * \tparam Reduction type of the reduction functor
 *
 * \param input the input array to be scanned
 * \param output the array where the result will be stored
 * \param begin the first element in the array to be scanned
 * \param end the last element in the array to be scanned
 * \param outputBegin the first element in the output array to be written. There
 *                    must be at least `end - begin` elements in the output
 *                    array starting at the position given by `outputBegin`.
 * \param reduction functor implementing the reduction operation
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation, i.e. element which does not
 *                 change the result of the reduction.
 *
 * The reduction functor takes two variables to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/exclusiveScanExample.cpp
 *
 * \par Output
 *
 * \include exclusiveScanExample.out
 */
template< typename InputArray,
          typename OutputArray,
          typename Reduction >
void
exclusiveScan( const InputArray& input,
               OutputArray& output,
               typename InputArray::IndexType begin,
               typename InputArray::IndexType end,
               typename OutputArray::IndexType outputBegin,
               Reduction&& reduction,
               typename OutputArray::ValueType identity )
{
   static_assert( std::is_same< typename InputArray::DeviceType, typename OutputArray::DeviceType >::value,
                  "The input and output arrays must have the same device type." );
   TNL_ASSERT_EQ( reduction( identity, identity ), identity,
                  "identity is not an identity element of the reduction operation" );
   // TODO: check if evaluating the input is expensive (e.g. a vector expression), otherwise use WriteInSecondPhase (optimal for array-to-array)
   using Scan = detail::Scan< typename OutputArray::DeviceType, detail::ScanType::Exclusive, detail::ScanPhaseType::WriteInFirstPhase >;
   Scan::perform( input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Overload of \ref exclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The [identity element](https://en.wikipedia.org/wiki/Identity_element) is
 * taken as `reduction.template getIdentity< typename OutputArray::ValueType >()`.
 * See \ref exclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `input.getSize()`.
 */
template< typename InputArray,
          typename OutputArray,
          typename Reduction = TNL::Plus >
void
exclusiveScan( const InputArray& input,
               OutputArray& output,
               typename InputArray::IndexType begin = 0,
               typename InputArray::IndexType end = 0,
               typename OutputArray::IndexType outputBegin = 0,
               Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = input.getSize();
   constexpr typename OutputArray::ValueType identity = Reduction::template getIdentity< typename OutputArray::ValueType >();
   exclusiveScan( input, output, begin, end, outputBegin, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Computes an inclusive scan (or prefix sum) of an array in-place.
 *
 * [Inclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$s_1, \ldots, s_n\f$ defined as
 *
 * \f[
 * s_i = \sum_{j=1}^i a_i.
 * \f]
 *
 * \tparam Array type of the array to be scanned
 * \tparam Reduction type of the reduction functor
 *
 * \param array input array, the result of scan is stored in the same array
 * \param begin the first element in the array to be scanned
 * \param end the last element in the array to be scanned
 * \param reduction functor implementing the reduction operation
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation, i.e. element which does not
 *                 change the result of the reduction.
 *
 * The reduction functor takes two variables to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/inplaceInclusiveScanExample.cpp
 *
 * \par Output
 *
 * \include inplaceInclusiveScanExample.out
 */
template< typename Array,
          typename Reduction >
void
inplaceInclusiveScan( Array& array,
                      typename Array::IndexType begin,
                      typename Array::IndexType end,
                      Reduction&& reduction,
                      typename Array::ValueType identity )
{
   TNL_ASSERT_EQ( reduction( identity, identity ), identity,
                  "identity is not an identity element of the reduction operation" );
   using Scan = detail::Scan< typename Array::DeviceType, detail::ScanType::Inclusive, detail::ScanPhaseType::WriteInSecondPhase >;
   Scan::perform( array, array, begin, end, begin, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Overload of \ref inplaceInclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The [identity element](https://en.wikipedia.org/wiki/Identity_element) is
 * taken as `reduction.template getIdentity< typename Array::ValueType >()`.
 * See \ref inplaceInclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `array.getSize()`.
 */
template< typename Array,
          typename Reduction = TNL::Plus >
void
inplaceInclusiveScan( Array& array,
                      typename Array::IndexType begin = 0,
                      typename Array::IndexType end = 0,
                      Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = array.getSize();
   constexpr typename Array::ValueType identity = Reduction::template getIdentity< typename Array::ValueType >();
   inplaceInclusiveScan( array, begin, end, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Computes an exclusive scan (or prefix sum) of an array in-place.
 *
 * [Exclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$\sigma_1, \ldots, \sigma_n\f$ defined as
 *
 * \f[
 * \sigma_i = \sum_{j=1}^{i-1} a_i.
 * \f]
 *
 * \tparam Array type of the array to be scanned
 * \tparam Reduction type of the reduction functor
 *
 * \param array input array, the result of scan is stored in the same array
 * \param begin the first element in the array to be scanned
 * \param end the last element in the array to be scanned
 * \param reduction functor implementing the reduction operation
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *                 for the reduction operation, i.e. element which does not
 *                 change the result of the reduction.
 *
 * The reduction functor takes two variables to be reduced:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
 * ```
 *
 * \par Example
 *
 * \include ReductionAndScan/inplaceExclusiveScanExample.cpp
 *
 * \par Output
 *
 * \include inplaceExclusiveScanExample.out
 */
template< typename Array,
          typename Reduction >
void
inplaceExclusiveScan( Array& array,
                      typename Array::IndexType begin,
                      typename Array::IndexType end,
                      Reduction&& reduction,
                      typename Array::ValueType identity )
{
   TNL_ASSERT_EQ( reduction( identity, identity ), identity,
                  "identity is not an identity element of the reduction operation" );
   using Scan = detail::Scan< typename Array::DeviceType, detail::ScanType::Exclusive, detail::ScanPhaseType::WriteInSecondPhase >;
   Scan::perform( array, array, begin, end, begin, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Overload of \ref inplaceExclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The [identity element](https://en.wikipedia.org/wiki/Identity_element) is
 * taken as `reduction.template getIdentity< typename Array::ValueType >()`.
 * See \ref inplaceExclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `array.getSize()`.
 */
template< typename Array,
          typename Reduction = TNL::Plus >
void
inplaceExclusiveScan( Array& array,
                      typename Array::IndexType begin = 0,
                      typename Array::IndexType end = 0,
                      Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = array.getSize();
   constexpr typename Array::ValueType identity = Reduction::template getIdentity< typename Array::ValueType >();
   inplaceExclusiveScan( array, begin, end, std::forward< Reduction >( reduction ), identity );
}

} // namespace Algorithms
} // namespace TNL
