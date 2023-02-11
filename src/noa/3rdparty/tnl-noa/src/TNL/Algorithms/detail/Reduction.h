// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::pair, std::forward

#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Sequential.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>

namespace noa::TNL {
namespace Algorithms {
namespace detail {

template< typename Device >
struct Reduction;

template<>
struct Reduction< Devices::Sequential >
{
   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static constexpr Result
   reduce( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );

   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static constexpr std::pair< Result, Index >
   reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );
};

template<>
struct Reduction< Devices::Host >
{
   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static Result
   reduce( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );

   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );
};

template<>
struct Reduction< Devices::Cuda >
{
   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static Result
   reduce( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );

   template< typename Index, typename Result, typename Fetch, typename Reduce >
   static std::pair< Result, Index >
   reduceWithArgument( Index begin, Index end, Fetch&& fetch, Reduce&& reduce, const Result& identity );
};

}  // namespace detail
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/detail/Reduction.hpp>
