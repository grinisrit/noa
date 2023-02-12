// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Edge.h>

namespace noa::TNL {
namespace Meshes {
namespace Topologies {

struct Polygon
{
   static constexpr int dimension = 2;
};

template<>
struct Subtopology< Polygon, 0 >
{
   using Topology = Vertex;
};

template<>
struct Subtopology< Polygon, 1 >
{
   using Topology = Edge;
};

}  // namespace Topologies
}  // namespace Meshes
}  // namespace noa::TNL
