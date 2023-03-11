// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <functional>  // reduction functions like std::plus, std::logical_and, std::logical_or etc.

#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Sequential.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>

namespace noa::TNL {
namespace Algorithms {

template< typename Device >
struct Multireduction;

template<>
struct Multireduction< Devices::Sequential >
{
   /**
    * Parameters:
    *    identity: the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *              for the reduction operation, i.e. element which does not
    *              change the result of the reduction
    *    dataFetcher: callable object such that `dataFetcher( i, j )` yields
    *                 the i-th value to be reduced from the j-th dataset
    *                 (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *               for example, it can be an instance of std::plus, std::logical_and,
    *               std::logical_or etc.
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    result: output array of size = n
    */
   template< typename Result, typename DataFetcher, typename Reduction, typename Index >
   static constexpr void
   reduce( Result identity, DataFetcher dataFetcher, Reduction reduction, Index size, int n, Result* result );
};

template<>
struct Multireduction< Devices::Host >
{
   /**
    * Parameters:
    *    identity: the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *              for the reduction operation, i.e. element which does not
    *              change the result of the reduction
    *    dataFetcher: callable object such that `dataFetcher( i, j )` yields
    *                 the i-th value to be reduced from the j-th dataset
    *                 (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *               for example, it can be an instance of std::plus, std::logical_and,
    *               std::logical_or etc.
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    result: output array of size = n
    */
   template< typename Result, typename DataFetcher, typename Reduction, typename Index >
   static void
   reduce( Result identity, DataFetcher dataFetcher, Reduction reduction, Index size, int n, Result* result );
};

template<>
struct Multireduction< Devices::Cuda >
{
   /**
    * Parameters:
    *    identity: the [identity element](https://en.wikipedia.org/wiki/Identity_element)
    *              for the reduction operation, i.e. element which does not
    *              change the result of the reduction
    *    dataFetcher: callable object such that `dataFetcher( i, j )` yields
    *                 the i-th value to be reduced from the j-th dataset
    *                 (i = 0,...,size-1; j = 0,...,n-1)
    *    reduction: callable object representing the reduction operation
    *               for example, it can be an instance of std::plus, std::logical_and,
    *               std::logical_or etc.
    *    size: the size of each dataset
    *    n: number of datasets to be reduced
    *    hostResult: output array of size = n
    */
   template< typename Result, typename DataFetcher, typename Reduction, typename Index >
   static void
   reduce( Result identity, DataFetcher dataFetcher, Reduction reduction, Index size, int n, Result* hostResult );
};

}  // namespace Algorithms
}  // namespace noa::TNL

#include "Multireduction.hpp"
