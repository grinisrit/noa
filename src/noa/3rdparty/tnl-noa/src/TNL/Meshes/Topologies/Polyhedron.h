// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polygon.h>

namespace noa::TNL {
namespace Meshes {
namespace Topologies {

struct Polyhedron
{
   static constexpr int dimension = 3;
};

template<>
struct Subtopology< Polyhedron, 0 >
{
   using Topology = Vertex;
};

template<>
struct Subtopology< Polyhedron, 1 >
{
   using Topology = Edge;
};

template<>
struct Subtopology< Polyhedron, 2 >
{
   using Topology = Polygon;
};

}  // namespace Topologies
}  // namespace Meshes
}  // namespace noa::TNL
