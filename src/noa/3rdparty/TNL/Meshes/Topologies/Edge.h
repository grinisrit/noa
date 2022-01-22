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

#include <noa/3rdparty/TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Vertex.h>

namespace noaTNL {
namespace Meshes {
namespace Topologies {

struct Edge
{
   static constexpr int dimension = 1;
};


template<>
struct Subtopology< Edge, 0 >
{
   typedef Vertex Topology;

   static constexpr int count = 2;
};

} // namespace Topologies
} // namespace Meshes
} // namespace noaTNL
