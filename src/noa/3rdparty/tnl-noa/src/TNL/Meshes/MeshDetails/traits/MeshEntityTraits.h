// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Array.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/UnorderedIndexedSet.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/SubentityVertexMap.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/EntitySeed.h>

#include <unordered_set>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename EntityTopology >
class MeshEntity;

template< typename MeshConfig, typename Device, int Dimension >
class MeshEntityTraits
{
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   static constexpr bool isDynamicTopology =
      Topologies::IsDynamicTopology< typename EntityTopologyGetter< MeshConfig, DimensionTag< Dimension > >::Topology >::value;

public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );

   using EntityTopology = typename EntityTopologyGetter< MeshConfig, DimensionTag< Dimension > >::Topology;
   using EntityType = MeshEntity< MeshConfig, Device, EntityTopology >;
   using SeedType = EntitySeed< MeshConfig, EntityTopology >;

   using SeedIndexedSetType =
      Containers::UnorderedIndexedSet< SeedType, GlobalIndexType, typename SeedType::HashType, typename SeedType::KeyEqual >;
   using SeedSetType = std::unordered_set< typename SeedIndexedSetType::key_type,
                                           typename SeedIndexedSetType::hasher,
                                           typename SeedIndexedSetType::key_equal >;
   using SeedMatrixType = EntitySeedMatrix< MeshConfig, EntityTopology >;

   // container for storing the subentity indices
   using SubentityMatrixType = std::conditional_t<
      isDynamicTopology,
      Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, SlicedEllpackSegments >,
      Matrices::SparseMatrix< bool, Device, GlobalIndexType, Matrices::GeneralMatrix, EllpackSegments > >;
};

}  // namespace Meshes
}  // namespace noa::TNL
