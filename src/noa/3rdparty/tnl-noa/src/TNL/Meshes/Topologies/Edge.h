// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Vertex.h>

namespace noa::TNL {
namespace Meshes {
namespace Topologies {

struct Edge
{
   static constexpr int dimension = 1;
};

template<>
struct Subtopology< Edge, 0 >
{
   using Topology = Vertex;

   static constexpr int count = 2;
};

}  // namespace Topologies
}  // namespace Meshes
}  // namespace noa::TNL
