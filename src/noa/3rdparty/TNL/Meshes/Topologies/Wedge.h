// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Topologies/SubentityVertexCount.h>
#include <TNL/Meshes/Topologies/Polygon.h>

namespace TNL {
namespace Meshes {
namespace Topologies {

struct Wedge
{
   static constexpr int dimension = 3;
};


template<>
struct Subtopology< Wedge, 0 >
{
   typedef Vertex Topology;

   static constexpr int count = 6;
};

template<>
struct Subtopology< Wedge, 1 >
{
   typedef Edge Topology;

   static constexpr int count = 9;
};

template<>
struct Subtopology< Wedge, 2 >
{
   typedef Polygon Topology;

   static constexpr int count = 5;
};

template<> struct SubentityVertexMap< Wedge, Edge, 0, 0> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Wedge, Edge, 0, 1> { static constexpr int index = 1; };

template<> struct SubentityVertexMap< Wedge, Edge, 1, 0> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Wedge, Edge, 1, 1> { static constexpr int index = 2; };

template<> struct SubentityVertexMap< Wedge, Edge, 2, 0> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Wedge, Edge, 2, 1> { static constexpr int index = 0; };

template<> struct SubentityVertexMap< Wedge, Edge, 3, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Wedge, Edge, 3, 1> { static constexpr int index = 4; };

template<> struct SubentityVertexMap< Wedge, Edge, 4, 0> { static constexpr int index = 4; };
template<> struct SubentityVertexMap< Wedge, Edge, 4, 1> { static constexpr int index = 5; };

template<> struct SubentityVertexMap< Wedge, Edge, 5, 0> { static constexpr int index = 5; };
template<> struct SubentityVertexMap< Wedge, Edge, 5, 1> { static constexpr int index = 3; };

template<> struct SubentityVertexMap< Wedge, Edge, 6, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Wedge, Edge, 6, 1> { static constexpr int index = 0; };

template<> struct SubentityVertexMap< Wedge, Edge, 7, 0> { static constexpr int index = 5; };
template<> struct SubentityVertexMap< Wedge, Edge, 7, 1> { static constexpr int index = 2; };

template<> struct SubentityVertexMap< Wedge, Edge, 8, 0> { static constexpr int index = 4; };
template<> struct SubentityVertexMap< Wedge, Edge, 8, 1> { static constexpr int index = 1; };


template <>
struct SubentityVertexCount< Wedge, Polygon, 0 >
{
   static constexpr int count = 3;
};

template<> struct SubentityVertexMap< Wedge, Polygon, 0, 0> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Wedge, Polygon, 0, 1> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Wedge, Polygon, 0, 2> { static constexpr int index = 2; };

template <>
struct SubentityVertexCount< Wedge, Polygon, 1 >
{
   static constexpr int count = 3;
};

template<> struct SubentityVertexMap< Wedge, Polygon, 1, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Wedge, Polygon, 1, 1> { static constexpr int index = 4; };
template<> struct SubentityVertexMap< Wedge, Polygon, 1, 2> { static constexpr int index = 5; };

template <>
struct SubentityVertexCount< Wedge, Polygon, 2 >
{
   static constexpr int count = 4;
};

template<> struct SubentityVertexMap< Wedge, Polygon, 2, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Wedge, Polygon, 2, 1> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Wedge, Polygon, 2, 2> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Wedge, Polygon, 2, 3> { static constexpr int index = 5; };

template <>
struct SubentityVertexCount< Wedge, Polygon, 3 >
{
   static constexpr int count = 4;
};

template<> struct SubentityVertexMap< Wedge, Polygon, 3, 0> { static constexpr int index = 4; };
template<> struct SubentityVertexMap< Wedge, Polygon, 3, 1> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Wedge, Polygon, 3, 2> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Wedge, Polygon, 3, 3> { static constexpr int index = 5; };

template <>
struct SubentityVertexCount< Wedge, Polygon, 4 >
{
   static constexpr int count = 4;
};

template<> struct SubentityVertexMap< Wedge, Polygon, 4, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Wedge, Polygon, 4, 1> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Wedge, Polygon, 4, 2> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Wedge, Polygon, 4, 3> { static constexpr int index = 4; };

} // namespace Topologies
} // namespace Meshes
} // namespace TNL
