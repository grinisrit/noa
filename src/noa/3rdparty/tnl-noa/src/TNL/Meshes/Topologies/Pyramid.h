// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/SubentityVertexCount.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polygon.h>

namespace noa::TNL {
namespace Meshes {
namespace Topologies {

struct Pyramid
{
   static constexpr int dimension = 3;
};

template<>
struct Subtopology< Pyramid, 0 >
{
   typedef Vertex Topology;

   static constexpr int count = 5;
};

template<>
struct Subtopology< Pyramid, 1 >
{
   typedef Edge Topology;

   static constexpr int count = 8;
};

template<>
struct Subtopology< Pyramid, 2 >
{
   typedef Polygon Topology;

   static constexpr int count = 5;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 0, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 0, 1 >
{
   static constexpr int index = 1;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 1, 0 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 1, 1 >
{
   static constexpr int index = 2;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 2, 0 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 2, 1 >
{
   static constexpr int index = 3;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 3, 0 >
{
   static constexpr int index = 3;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 3, 1 >
{
   static constexpr int index = 0;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 4, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 4, 1 >
{
   static constexpr int index = 4;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 5, 0 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 5, 1 >
{
   static constexpr int index = 4;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 6, 0 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 6, 1 >
{
   static constexpr int index = 4;
};

template<>
struct SubentityVertexMap< Pyramid, Edge, 7, 0 >
{
   static constexpr int index = 3;
};
template<>
struct SubentityVertexMap< Pyramid, Edge, 7, 1 >
{
   static constexpr int index = 4;
};

template<>
struct SubentityVertexCount< Pyramid, Polygon, 0 >
{
   static constexpr int count = 4;
};

template<>
struct SubentityVertexMap< Pyramid, Polygon, 0, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 0, 1 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 0, 2 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 0, 3 >
{
   static constexpr int index = 3;
};

template<>
struct SubentityVertexCount< Pyramid, Polygon, 1 >
{
   static constexpr int count = 3;
};

template<>
struct SubentityVertexMap< Pyramid, Polygon, 1, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 1, 1 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 1, 2 >
{
   static constexpr int index = 4;
};

template<>
struct SubentityVertexCount< Pyramid, Polygon, 2 >
{
   static constexpr int count = 3;
};

template<>
struct SubentityVertexMap< Pyramid, Polygon, 2, 0 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 2, 1 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 2, 2 >
{
   static constexpr int index = 4;
};

template<>
struct SubentityVertexCount< Pyramid, Polygon, 3 >
{
   static constexpr int count = 3;
};

template<>
struct SubentityVertexMap< Pyramid, Polygon, 3, 0 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 3, 1 >
{
   static constexpr int index = 3;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 3, 2 >
{
   static constexpr int index = 4;
};

template<>
struct SubentityVertexCount< Pyramid, Polygon, 4 >
{
   static constexpr int count = 3;
};

template<>
struct SubentityVertexMap< Pyramid, Polygon, 4, 0 >
{
   static constexpr int index = 3;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 4, 1 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Pyramid, Polygon, 4, 2 >
{
   static constexpr int index = 4;
};

}  // namespace Topologies
}  // namespace Meshes
}  // namespace noa::TNL
