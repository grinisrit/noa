// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/TypeTraits.h>
#include <noa/3rdparty/TNL/Meshes/MeshEntity.h>

namespace noaTNL {
namespace Meshes {
namespace Writers {
namespace detail {

template< typename T, typename Enable = void >
struct has_entity_topology : std::false_type {};

template< typename T >
struct has_entity_topology< T, typename enable_if_type< typename T::EntityTopology >::type >
: std::true_type
{};

template< typename Entity,
          bool _is_mesh_entity = has_entity_topology< Entity >::value >
struct VerticesPerEntity
{
   static constexpr int count = Topologies::Subtopology< typename Entity::EntityTopology, 0 >::count;
};

template< typename MeshConfig, typename Device >
struct VerticesPerEntity< MeshEntity< MeshConfig, Device, Topologies::Vertex >, true >
{
   static constexpr int count = 1;
};

template< typename GridEntity >
struct VerticesPerEntity< GridEntity, false >
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr int count =
      (dim == 0) ? 1 :
      (dim == 1) ? 2 :
      (dim == 2) ? 4 :
                   8;
};

} // namespace detail
} // namespace Writers
} // namespace Meshes
} // namespace noaTNL
