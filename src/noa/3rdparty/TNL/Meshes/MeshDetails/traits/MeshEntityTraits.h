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

#include <noa/3rdparty/TNL/Containers/Array.h>
#include <noa/3rdparty/TNL/Containers/UnorderedIndexedSet.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/initializer/EntitySeed.h>

#include <unordered_set>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename EntityTopology > class MeshEntity;

/****
 *       Mesh entity traits with specializations
 *
 *  DYNAMIC TOPOLOGY
 *       FALSE
 */
template< typename MeshConfig,
          typename Device,
          int Dimension >
class MeshEntityTraits< MeshConfig, Device, Dimension, false >
{
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );

   using EntityTopology                = typename EntityTopologyGetter< MeshConfig, DimensionTag< Dimension > >::Topology;
   using EntityType                    = MeshEntity< MeshConfig, Device, EntityTopology >;
   using SeedType                      = EntitySeed< MeshConfig, EntityTopology >;
   
   using SeedIndexedSetType            = Containers::UnorderedIndexedSet< SeedType, GlobalIndexType, typename SeedType::HashType, typename SeedType::KeyEqual >;
   using SeedSetType                   = std::unordered_set< typename SeedIndexedSetType::key_type, typename SeedIndexedSetType::hasher, typename SeedIndexedSetType::key_equal >;
   using SeedMatrixType                = EntitySeedMatrix< MeshConfig, EntityTopology >;

   // container for storing the subentity indices
   using SubentityMatrixType = Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, EllpackSegments >;
};

/****
 *       Mesh entity traits with specializations
 *
 *  DYNAMIC TOPOLOGY
 *       TRUE
 */
template< typename MeshConfig,
          typename Device,
          int Dimension >
class MeshEntityTraits< MeshConfig, Device, Dimension, true >
{
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );

   using EntityTopology                = typename EntityTopologyGetter< MeshConfig, DimensionTag< Dimension > >::Topology;
   using EntityType                    = MeshEntity< MeshConfig, Device, EntityTopology >;
   using SeedType                      = EntitySeed< MeshConfig, EntityTopology >;
   
   using SeedIndexedSetType            = Containers::UnorderedIndexedSet< SeedType, GlobalIndexType, typename SeedType::HashType, typename SeedType::KeyEqual >;
   using SeedSetType                   = std::unordered_set< typename SeedIndexedSetType::key_type, typename SeedIndexedSetType::hasher, typename SeedIndexedSetType::key_equal >;
   using SeedMatrixType                = EntitySeedMatrix< MeshConfig, EntityTopology >;

   // container for storing the subentity indices
   using SubentityMatrixType = Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, SlicedEllpackSegments >;
};

} // namespace Meshes
} // namespace noa::TNL
