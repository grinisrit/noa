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

namespace TNL {
namespace Meshes{
namespace Topologies {

template< typename EntityTopology,
          int SubentityDimension >
struct Subtopology
{
};

template< typename EntityTopology,
          typename SubentityTopology,
          int SubentityIndex,
          int SubentityVertexIndex >
struct SubentityVertexMap
{
};

} // namespace Topologies
} // namespace Meshes
} // namespace TNL
