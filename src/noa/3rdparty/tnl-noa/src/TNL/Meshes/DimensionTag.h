// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>

namespace noa::TNL {
namespace Meshes {

/***
 * This tag or integer wrapper is necessary for C++ templates specializations.
 * As the C++ standard says:
 *
 *   A partially specialized non-type argument expression shall not involve
 *   a template parameter of the partial specialization except when the argument
 *   expression is a simple identifier.
 *
 * Therefore one cannot specialize the mesh layers just by integers saying the mesh
 * layer dimension but instead this tag must be used. This makes the code more difficult
 * to read and we would like to avoid it if it is possible sometime.
 * On the other hand, DimensionTag is also used for method overloading when
 * asking for different mesh entities. In this case it makes sense and it cannot be
 * replaced.
 */

template< int Dimension >
class DimensionTag
{
   static_assert( Dimension >= 0, "The dimension cannot be negative." );

public:
   static constexpr int value = Dimension;

   using Decrement = DimensionTag< Dimension - 1 >;
   using Increment = DimensionTag< Dimension + 1 >;
};

template<>
class DimensionTag< 0 >
{
public:
   static constexpr int value = 0;

   using Increment = DimensionTag< 1 >;
};

}  // namespace Meshes
}  // namespace noa::TNL
