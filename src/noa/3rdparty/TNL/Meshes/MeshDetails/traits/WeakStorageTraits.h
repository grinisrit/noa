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

#include <TNL/Meshes/MeshDetails/traits/MeshSubentityTraits.h>
#include <TNL/Meshes/MeshDetails/traits/MeshSuperentityTraits.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag,
          bool sensible = (SubdimensionTag::value < EntityTopology::dimension) >
struct WeakSubentityStorageTrait
{
   static constexpr bool storageEnabled = MeshTraits< MeshConfig, Device >::template SubentityTraits< EntityTopology, SubdimensionTag::value >::storageEnabled;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
struct WeakSubentityStorageTrait< MeshConfig, Device, EntityTopology, SubdimensionTag, false >
{
   static constexpr bool storageEnabled = false;
};


template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SuperdimensionTag,
          bool sensible = (SuperdimensionTag::value > EntityTopology::dimension) >
struct WeakSuperentityStorageTrait
{
   static constexpr bool storageEnabled = MeshTraits< MeshConfig, Device >::template SuperentityTraits< EntityTopology, SuperdimensionTag::value >::storageEnabled;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SuperdimensionTag >
struct WeakSuperentityStorageTrait< MeshConfig, Device, EntityTopology, SuperdimensionTag, false >
{
   static constexpr bool storageEnabled = false;
};

} // namespace Meshes
} // namespace TNL
