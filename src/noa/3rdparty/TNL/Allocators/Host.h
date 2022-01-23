// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <memory>

namespace noa::TNL {

/**
 * \brief Namespace for TNL allocators.
 *
 * All TNL allocators must satisfy the requirements imposed by the
 * [Allocator concept](https://en.cppreference.com/w/cpp/named_req/Allocator)
 * from STL.
 */
namespace Allocators {

/**
 * \brief Allocator for the host memory space -- alias for \ref std::allocator.
 */
template< class T >
using Host = std::allocator< T >;

} // namespace Allocators
} // namespace noa::TNL
