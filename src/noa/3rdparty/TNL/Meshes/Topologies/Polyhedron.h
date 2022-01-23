// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Topologies/Polygon.h>

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
   typedef Vertex Topology;
};

template<>
struct Subtopology< Polyhedron, 1 >
{
   typedef Edge Topology;
};

template<>
struct Subtopology< Polyhedron, 2 >
{
   typedef Polygon Topology;
};

} // namespace Topologies
} // namespace Meshes
} // namespace noa::TNL