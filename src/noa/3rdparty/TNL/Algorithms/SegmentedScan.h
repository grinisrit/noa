// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

// TODO: move this into the detail namespace, create dispatching functions like
// inplaceInclusiveSegmentedScan, inplaceExclusiveSegmentedScan, etc.

#include <noa/3rdparty/TNL/Devices/Sequential.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Algorithms/detail/ScanType.h>

namespace noaTNL {
namespace Algorithms {

/**
 * \brief Computes segmented scan (or prefix sum) on a vector.
 *
 * Segmented scan is a modification of common scan. In this case the sequence of
 * numbers in hand is divided into segments like this, for example
 *
 * ```
 * [1,3,5][2,4,6,9][3,5],[3,6,9,12,15]
 * ```
 *
 * and we want to compute inclusive or exclusive scan of each segment. For inclusive segmented prefix sum we get
 *
 * ```
 * [1,4,9][2,6,12,21][3,8][3,9,18,30,45]
 * ```
 *
 * and for exclusive segmented prefix sum it is
 *
 * ```
 * [0,1,4][0,2,6,12][0,3][0,3,9,18,30]
 * ```
 *
 * In addition to common scan, we need to encode the segments of the input sequence.
 * It is done by auxiliary flags array (it can be array of booleans) having `1` at the
 * beginning of each segment and `0` on all other positions. In our example, it would be like this:
 *
 * ```
 * [1,0,0,1,0,0,0,1,0,1,0,0, 0, 0]
 * [1,3,5,2,4,6,9,3,5,3,6,9,12,15]
 *
 * ```
 *
 * \tparam Device parameter says on what device the reduction is gonna be performed.
 * \tparam Type parameter says if inclusive or exclusive is scan is to be computed.
 *
 * See \ref Scan< Devices::Host, Type > and \ref Scan< Devices::Cuda, Type >.
 *
 * **Note: Segmented scan is not implemented for CUDA yet.**
 */
template< typename Device,
          detail::ScanType Type = detail::ScanType::Inclusive >
struct SegmentedScan;

template< detail::ScanType Type >
struct SegmentedScan< Devices::Sequential, Type >
{
   /**
    * \brief Computes segmented scan (prefix sum) sequentially.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    * \tparam Flags array type containing zeros and ones defining the segments begining
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param flags is an array with zeros and ones defining the segments begining
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation, i.e. element which does not
    *                 change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SegmentedScanExample.cpp
    *
    * \par Output
    *
    * \include SegmentedScanExample.out
    */
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType identity );
};

template< detail::ScanType Type >
struct SegmentedScan< Devices::Host, Type >
{
   /**
    * \brief Computes segmented scan (prefix sum) using OpenMP.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    * \tparam Flags array type containing zeros and ones defining the segments begining
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param flags is an array with zeros and ones defining the segments begining
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation, i.e. element which does not
    *                 change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SegmentedScanExample.cpp
    *
    * \par Output
    *
    * \include SegmentedScanExample.out
    */
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType identity );
};

template< detail::ScanType Type >
struct SegmentedScan< Devices::Cuda, Type >
{
   /**
    * \brief Computes segmented scan (prefix sum) on GPU.
    *
    * \tparam Vector type vector being used for the scan.
    * \tparam Reduction lambda function defining the reduction operation
    * \tparam Flags array type containing zeros and ones defining the segments begining
    *
    * \param v input vector, the result of scan is stored in the same vector
    * \param flags is an array with zeros and ones defining the segments begining
    * \param begin the first element in the array to be scanned
    * \param end the last element in the array to be scanned
    * \param reduction lambda function implementing the reduction operation
    * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *                 for the reduction operation, i.e. element which does not
    *                 change the result of the reduction.
    *
    * The reduction lambda function takes two variables which are supposed to be reduced:
    *
    * ```
    * auto reduction = [] __cuda_callable__ ( const Result& a, const Result& b ) { return ... };
    * ```
    *
    * \par Example
    *
    * \include ReductionAndScan/SegmentedScanExample.cpp
    *
    * \par Output
    *
    * \include SegmentedScanExample.out
    *
    * **Note: Segmented scan is not implemented for CUDA yet.**
    */
   template< typename Vector,
             typename Reduction,
             typename Flags >
   static void
   perform( Vector& v,
            Flags& flags,
            const typename Vector::IndexType begin,
            const typename Vector::IndexType end,
            const Reduction& reduction,
            const typename Vector::ValueType identity );
};

} // namespace Algorithms
} // namespace noaTNL

#include <noa/3rdparty/TNL/Algorithms/SegmentedScan.hpp>
