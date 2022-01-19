// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <utility>  // std::forward

#include <TNL/Algorithms/detail/DistributedScan.h>
#include <TNL/Functional.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief Computes an inclusive scan (or prefix sum) of a distributed array in-place.
 *
 * [Inclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$s_1, \ldots, s_n\f$ defined as
 *
 * \f[
 * s_i = \sum_{j=1}^i a_i.
 * \f]
 *
 * \tparam DistributedArray type of the distributed array to be scanned
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
 */
template< typename InputDistributedArray,
          typename OutputDistributedArray,
          typename Reduction >
void
distributedInclusiveScan( const InputDistributedArray& input,
                          OutputDistributedArray& output,
                          typename InputDistributedArray::IndexType begin,
                          typename InputDistributedArray::IndexType end,
                          Reduction&& reduction,
                          typename OutputDistributedArray::ValueType identity )
{
   static_assert( std::is_same< typename InputDistributedArray::DeviceType, typename OutputDistributedArray::DeviceType >::value,
                  "The input and output arrays must have the same device type." );
   TNL_ASSERT_EQ( input.getCommunicator(), output.getCommunicator(),
                  "The input and output arrays must have the same MPI communicator." );
   TNL_ASSERT_EQ( input.getLocalRange(), output.getLocalRange(),
                  "The input and output arrays must have the same local range on all ranks." );
   // TODO: check if evaluating the input is expensive (e.g. a vector expression), otherwise use WriteInSecondPhase (optimal for array-to-array)
   using Scan = detail::DistributedScan< detail::ScanType::Inclusive, detail::ScanPhaseType::WriteInFirstPhase >;
   Scan::perform( input, output, begin, end, std::forward< Reduction >( reduction ), identity );
   output.startSynchronization();
}

/**
 * \brief Overload of \ref distributedInclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The identity element is taken as `reduction.template getIdentity< typename OutputDistributedArray::ValueType >()`.
 * See \ref distributedInclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `input.getSize()`.
 */
template< typename InputDistributedArray,
          typename OutputDistributedArray,
          typename Reduction = TNL::Plus >
void
distributedInclusiveScan( const InputDistributedArray& input,
                          OutputDistributedArray& output,
                          typename InputDistributedArray::IndexType begin = 0,
                          typename InputDistributedArray::IndexType end = 0,
                          Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = input.getSize();
   constexpr typename OutputDistributedArray::ValueType identity = Reduction::template getIdentity< typename OutputDistributedArray::ValueType >();
   distributedInclusiveScan( input, output, begin, end, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Computes an exclusive scan (or prefix sum) of a distributed array in-place.
 *
 * [Exclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$\sigma_1, \ldots, \sigma_n\f$ defined as
 *
 * \f[
 * \sigma_i = \sum_{j=1}^{i-1} a_i.
 * \f]
 *
 * \tparam DistributedArray type of the distributed array to be scanned
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
 */
template< typename InputDistributedArray,
          typename OutputDistributedArray,
          typename Reduction >
void
distributedExclusiveScan( const InputDistributedArray& input,
                          OutputDistributedArray& output,
                          typename InputDistributedArray::IndexType begin,
                          typename InputDistributedArray::IndexType end,
                          Reduction&& reduction,
                          typename OutputDistributedArray::ValueType identity )
{
   static_assert( std::is_same< typename InputDistributedArray::DeviceType, typename OutputDistributedArray::DeviceType >::value,
                  "The input and output arrays must have the same device type." );
   TNL_ASSERT_EQ( input.getCommunicator(), output.getCommunicator(),
                  "The input and output arrays must have the same MPI communicator." );
   TNL_ASSERT_EQ( input.getLocalRange(), output.getLocalRange(),
                  "The input and output arrays must have the same local range on all ranks." );
   // TODO: check if evaluating the input is expensive (e.g. a vector expression), otherwise use WriteInSecondPhase (optimal for array-to-array)
   using Scan = detail::DistributedScan< detail::ScanType::Exclusive, detail::ScanPhaseType::WriteInFirstPhase >;
   Scan::perform( input, output, begin, end, std::forward< Reduction >( reduction ), identity );
   output.startSynchronization();
}

/**
 * \brief Overload of \ref distributedExclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The identity element is taken as `reduction.template getIdentity< typename OutputDistributedArray::ValueType >()`.
 * See \ref distributedExclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `input.getSize()`.
 */
template< typename InputDistributedArray,
          typename OutputDistributedArray,
          typename Reduction = TNL::Plus >
void
distributedExclusiveScan( const InputDistributedArray& input,
                          OutputDistributedArray& output,
                          typename InputDistributedArray::IndexType begin = 0,
                          typename InputDistributedArray::IndexType end = 0,
                          Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = input.getSize();
   constexpr typename OutputDistributedArray::ValueType identity = Reduction::template getIdentity< typename OutputDistributedArray::ValueType >();
   distributedExclusiveScan( input, output, begin, end, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Computes an inclusive scan (or prefix sum) of a distributed array in-place.
 *
 * [Inclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$s_1, \ldots, s_n\f$ defined as
 *
 * \f[
 * s_i = \sum_{j=1}^i a_i.
 * \f]
 *
 * \tparam DistributedArray type of the distributed array to be scanned
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
 */
template< typename DistributedArray,
          typename Reduction >
void
distributedInplaceInclusiveScan( DistributedArray& array,
                                 typename DistributedArray::IndexType begin,
                                 typename DistributedArray::IndexType end,
                                 Reduction&& reduction,
                                 typename DistributedArray::ValueType identity )
{
   using Scan = detail::DistributedScan< detail::ScanType::Inclusive, detail::ScanPhaseType::WriteInSecondPhase >;
   Scan::perform( array, array, begin, end, std::forward< Reduction >( reduction ), identity );
   array.startSynchronization();
}

/**
 * \brief Overload of \ref distributedInplaceInclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The identity element is taken as `reduction.template getIdentity< typename DistributedArray::ValueType >()`.
 * See \ref distributedInplaceInclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `array.getSize()`.
 */
template< typename DistributedArray,
          typename Reduction = TNL::Plus >
void
distributedInplaceInclusiveScan( DistributedArray& array,
                                 typename DistributedArray::IndexType begin = 0,
                                 typename DistributedArray::IndexType end = 0,
                                 Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = array.getSize();
   constexpr typename DistributedArray::ValueType identity = Reduction::template getIdentity< typename DistributedArray::ValueType >();
   distributedInplaceInclusiveScan( array, begin, end, std::forward< Reduction >( reduction ), identity );
}

/**
 * \brief Computes an exclusive scan (or prefix sum) of a distributed array in-place.
 *
 * [Exclusive scan (or prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)
 * operation turns a sequence \f$a_1, \ldots, a_n\f$ into a sequence
 * \f$\sigma_1, \ldots, \sigma_n\f$ defined as
 *
 * \f[
 * \sigma_i = \sum_{j=1}^{i-1} a_i.
 * \f]
 *
 * \tparam DistributedArray type of the distributed array to be scanned
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
 */
template< typename DistributedArray,
          typename Reduction >
void
distributedInplaceExclusiveScan( DistributedArray& array,
                                 typename DistributedArray::IndexType begin,
                                 typename DistributedArray::IndexType end,
                                 Reduction&& reduction,
                                 typename DistributedArray::ValueType identity )
{
   using Scan = detail::DistributedScan< detail::ScanType::Exclusive, detail::ScanPhaseType::WriteInSecondPhase >;
   Scan::perform( array, array, begin, end, std::forward< Reduction >( reduction ), identity );
   array.startSynchronization();
}

/**
 * \brief Overload of \ref distributedInplaceExclusiveScan which uses a TNL functional
 *        object for reduction. \ref TNL::Plus is used by default.
 *
 * The identity element is taken as `reduction.template getIdentity< typename DistributedArray::ValueType >()`.
 * See \ref distributedInplaceExclusiveScan for the explanation of other parameters.
 * Note that when `end` equals 0 (the default), it is set to `array.getSize()`.
 */
template< typename DistributedArray,
          typename Reduction = TNL::Plus >
void
distributedInplaceExclusiveScan( DistributedArray& array,
                                 typename DistributedArray::IndexType begin = 0,
                                 typename DistributedArray::IndexType end = 0,
                                 Reduction&& reduction = TNL::Plus{} )
{
   if( end == 0 )
      end = array.getSize();
   constexpr typename DistributedArray::ValueType identity = Reduction::template getIdentity< typename DistributedArray::ValueType >();
   distributedInplaceExclusiveScan( array, begin, end, std::forward< Reduction >( reduction ), identity );
}

} // namespace Algorithms
} // namespace TNL
