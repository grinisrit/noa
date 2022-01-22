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

#include <noa/3rdparty/TNL/Meshes/Topologies/Quadrangle.h>

namespace noaTNL {
namespace Meshes {
namespace Topologies {

struct Hexahedron
{
   static constexpr int dimension = 3;
};

template<>
struct Subtopology< Hexahedron, 0 >
{
   typedef Vertex Topology;

   static constexpr int count = 8;
};

template<>
struct Subtopology< Hexahedron, 1 >
{
   typedef Edge Topology;

   static constexpr int count = 12;
};

template<>
struct Subtopology< Hexahedron, 2 >
{
   typedef Quadrangle Topology;

   static constexpr int count = 6;
};

/****
 * Indexing of the vertices follows the VTK file format
 *
 *        7+---------------------------+6
 *        /|                          /|
 *       / |                         / |
 *      /  |                        /  |
 *     /   |                       /   |
 *   4+---------------------------+5   |
 *    |    |                      |    |
 *    |    |                      |    |
 *    |   3+----------------------|----+2
 *    |   /                       |   /
 *    |  /                        |  /
 *    | /                         | /
 *    |/                          |/
 *   0+---------------------------+1
 *
 *
 * The edges are indexed as follows:
 *
 *         +---------------------------+
 *        /|           10             /|
 *     11/ |                         / |
 *      /  |                        /9 |
 *     /  7|                       /   |6
 *    +---------------------------+    |
 *    |    |        8             |    |
 *    |    |                      |    |
 *    |    +----------------------|----+
 *   4|   /           2           |5  /
 *    | 3/                        |  /
 *    | /                         | /1
 *    |/                          |/
 *    +---------------------------+
 *                 0
 *
 * The faces are indexed as follows (the indexed are positioned to
 * the opposite corners of given face):
 *
 *         +---------------------------+
 *        /|5                        3/|
 *       /4|                         /2|
 *      /  |                        /  |
 *     /   |                     5 /   |
 *    +---------------------------+    |
 *    |1   |                      |    |
 *    |    |3                     |    |
 *    |    +----------------------|----+
 *    |   /                       |  0/
 *    |  /                        |  /
 *    |4/                         |2/
 *    |/0                        1|/
 *    +---------------------------+
 *
 */

template<> struct SubentityVertexMap< Hexahedron, Edge,  0, 0> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  0, 1> { static constexpr int index = 1; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  1, 0> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  1, 1> { static constexpr int index = 2; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  2, 0> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  2, 1> { static constexpr int index = 3; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  3, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  3, 1> { static constexpr int index = 0; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  4, 0> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  4, 1> { static constexpr int index = 4; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  5, 0> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  5, 1> { static constexpr int index = 5; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  6, 0> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  6, 1> { static constexpr int index = 6; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  7, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  7, 1> { static constexpr int index = 7; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  8, 0> { static constexpr int index = 4; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  8, 1> { static constexpr int index = 5; };

template<> struct SubentityVertexMap< Hexahedron, Edge,  9, 0> { static constexpr int index = 5; };
template<> struct SubentityVertexMap< Hexahedron, Edge,  9, 1> { static constexpr int index = 6; };

template<> struct SubentityVertexMap< Hexahedron, Edge, 10, 0> { static constexpr int index = 6; };
template<> struct SubentityVertexMap< Hexahedron, Edge, 10, 1> { static constexpr int index = 7; };

template<> struct SubentityVertexMap< Hexahedron, Edge, 11, 0> { static constexpr int index = 7; };
template<> struct SubentityVertexMap< Hexahedron, Edge, 11, 1> { static constexpr int index = 4; };


template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 0, 0> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 0, 1> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 0, 2> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 0, 3> { static constexpr int index = 3; };

template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 1, 0> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 1, 1> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 1, 2> { static constexpr int index = 5; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 1, 3> { static constexpr int index = 4; };

template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 2, 0> { static constexpr int index = 1; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 2, 1> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 2, 2> { static constexpr int index = 6; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 2, 3> { static constexpr int index = 5; };

template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 3, 0> { static constexpr int index = 2; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 3, 1> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 3, 2> { static constexpr int index = 7; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 3, 3> { static constexpr int index = 6; };

template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 4, 0> { static constexpr int index = 3; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 4, 1> { static constexpr int index = 0; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 4, 2> { static constexpr int index = 4; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 4, 3> { static constexpr int index = 7; };

template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 5, 0> { static constexpr int index = 4; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 5, 1> { static constexpr int index = 5; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 5, 2> { static constexpr int index = 6; };
template<> struct SubentityVertexMap< Hexahedron, Quadrangle, 5, 3> { static constexpr int index = 7; };

} // namespace Topologies
} // namespace Meshes
} // namespace noaTNL
