// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/TypeTraits.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Vertex.h>

namespace noa::TNL {
namespace Meshes {
namespace Topologies {

/**
 * \brief Type trait for checking if Topology has at least one missing Subtopology< Topology, D > >::count for all D from Topology::dimension - 1 to 0
 */
template< typename Topology, int D = Topology::dimension >
struct IsDynamicTopology
{
   static constexpr bool value = ! HasCountMember< Subtopology< Topology, D - 1 > >::value ||
                                   IsDynamicTopology< Topology, D - 1 >::value;
};

/**
 * \brief Specialization for Vertex Topology
 */
template<>
struct IsDynamicTopology< Vertex, 0 > : std::false_type
{};

/**
 * \brief Specialization for D = 1 to end recursion
 */
template< typename Topology >
struct IsDynamicTopology< Topology, 1 >
{
   static constexpr bool value = ! HasCountMember< Subtopology< Topology, 0 > >::value;
};

} // namespace Topologies
} // namespace Meshes
} // namespace noa::TNL
