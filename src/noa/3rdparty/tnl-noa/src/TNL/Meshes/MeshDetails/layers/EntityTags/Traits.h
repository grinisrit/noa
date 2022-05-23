// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace noa::TNL {
namespace Meshes {
namespace EntityTags {

template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool sensible = ( DimensionTag::value <= MeshConfig::meshDimension ) >
struct WeakStorageTrait
{
   static constexpr bool entityTagsEnabled = MeshConfig::entityTagsStorage( DimensionTag::value );
};

template< typename MeshConfig, typename Device, typename DimensionTag >
struct WeakStorageTrait< MeshConfig, Device, DimensionTag, false >
{
   static constexpr bool entityTagsEnabled = false;
};

// Entity tags are used in a bitset fashion. Unused bits are available for
// user needs, but these bits should not be changed by users.
enum EntityTags : std::uint8_t
{
   BoundaryEntity = 1,
   GhostEntity = 2,
};

}  // namespace EntityTags
}  // namespace Meshes
}  // namespace noa::TNL
