// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig, typename Device, typename EntityTopology, int Dimension >
class MeshSuperentityTraits
{
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType = typename MeshConfig::LocalIndexType;

public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );
   static_assert( EntityTopology::dimension < Dimension, "Superentity dimension must be higher than the entity dimension." );

   static constexpr bool storageEnabled = MeshConfig::superentityStorage( EntityTopology::dimension, Dimension );

   using SuperentityTopology = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityTopology;
   using SuperentityType = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityType;
};

}  // namespace Meshes
}  // namespace noa::TNL
