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

#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>

namespace noaTNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          int Dimension >
class MeshSuperentityTraits
{
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType  = typename MeshConfig::LocalIndexType;

public:
   static_assert( 0 <= Dimension && Dimension <= MeshConfig::meshDimension, "invalid dimension" );
   static_assert( EntityTopology::dimension < Dimension, "Superentity dimension must be higher than the entity dimension." );

   static constexpr bool storageEnabled = MeshConfig::superentityStorage( EntityTopology::dimension, Dimension );

   using SuperentityTopology = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityTopology;
   using SuperentityType     = typename MeshEntityTraits< MeshConfig, Device, Dimension >::EntityType;
};

} // namespace Meshes
} // namespace noaTNL
