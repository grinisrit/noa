// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Triangle.h>

namespace noa::TNL {
namespace Meshes {
namespace Topologies {

struct Tetrahedron
{
   static constexpr int dimension = 3;
};

template<>
struct Subtopology< Tetrahedron, 0 >
{
   using Topology = Vertex;

   static constexpr int count = 4;
};

template<>
struct Subtopology< Tetrahedron, 1 >
{
   using Topology = Edge;

   static constexpr int count = 6;
};

template<>
struct Subtopology< Tetrahedron, 2 >
{
   using Topology = Triangle;

   static constexpr int count = 4;
};

template<>
struct SubentityVertexMap< Tetrahedron, Edge, 0, 0 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Tetrahedron, Edge, 0, 1 >
{
   static constexpr int index = 2;
};

template<>
struct SubentityVertexMap< Tetrahedron, Edge, 1, 0 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Tetrahedron, Edge, 1, 1 >
{
   static constexpr int index = 0;
};

template<>
struct SubentityVertexMap< Tetrahedron, Edge, 2, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Tetrahedron, Edge, 2, 1 >
{
   static constexpr int index = 1;
};

template<>
struct SubentityVertexMap< Tetrahedron, Edge, 3, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Tetrahedron, Edge, 3, 1 >
{
   static constexpr int index = 3;
};

template<>
struct SubentityVertexMap< Tetrahedron, Edge, 4, 0 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Tetrahedron, Edge, 4, 1 >
{
   static constexpr int index = 3;
};

template<>
struct SubentityVertexMap< Tetrahedron, Edge, 5, 0 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Tetrahedron, Edge, 5, 1 >
{
   static constexpr int index = 3;
};

// i-th subvertex is the opposite vertex of i-th subface
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 0, 0 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 0, 1 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 0, 2 >
{
   static constexpr int index = 3;
};

template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 1, 0 >
{
   static constexpr int index = 2;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 1, 1 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 1, 2 >
{
   static constexpr int index = 3;
};

template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 2, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 2, 1 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 2, 2 >
{
   static constexpr int index = 3;
};

template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 3, 0 >
{
   static constexpr int index = 0;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 3, 1 >
{
   static constexpr int index = 1;
};
template<>
struct SubentityVertexMap< Tetrahedron, Triangle, 3, 2 >
{
   static constexpr int index = 2;
};

}  // namespace Topologies
}  // namespace Meshes
}  // namespace noa::TNL
