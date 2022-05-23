// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
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
   typedef Vertex Topology;
};

template<>
struct Subtopology< Polygon, 1 >
{
   typedef Edge Topology;
};

}  // namespace Topologies
}  // namespace Meshes
}  // namespace noa::TNL