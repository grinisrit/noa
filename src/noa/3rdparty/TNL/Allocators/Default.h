// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Allocators/Host.h>
#include <noa/3rdparty/TNL/Allocators/Cuda.h>
#include <noa/3rdparty/TNL/Devices/Sequential.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>

namespace noa::TNL {
namespace Allocators {

/**
 * \brief A trait-like class used for the selection of a default allocators for
 * given device.
 */
template< typename Device >
struct Default;

//! Sets \ref Allocators::Host as the default allocator for \ref Devices::Sequential.
template<>
struct Default< Devices::Sequential >
{
   template< typename T >
   using Allocator = Allocators::Host< T >;
};

//! Sets \ref Allocators::Host as the default allocator for \ref Devices::Host.
template<>
struct Default< Devices::Host >
{
   template< typename T >
   using Allocator = Allocators::Host< T >;
};

//! Sets \ref Allocators::Cuda as the default allocator for \ref Devices::Cuda.
template<>
struct Default< Devices::Cuda >
{
   template< typename T >
   using Allocator = Allocators::Cuda< T >;
};

} // namespace Allocators
} // namespace noa::TNL
