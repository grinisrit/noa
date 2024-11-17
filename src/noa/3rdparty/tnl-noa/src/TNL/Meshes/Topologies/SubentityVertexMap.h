// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Meshes {
namespace Topologies {

template< typename EntityTopology, int SubentityDimension >
struct Subtopology
{};

template< typename EntityTopology, typename SubentityTopology, int SubentityIndex, int SubentityVertexIndex >
struct SubentityVertexMap
{};

}  // namespace Topologies
}  // namespace Meshes
}  // namespace noa::TNL
