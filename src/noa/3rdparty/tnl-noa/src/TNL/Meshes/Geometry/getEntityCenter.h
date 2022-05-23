// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridEntity.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshEntity.h>

namespace noa::TNL {
namespace Meshes {

// compatibility wrapper
template< typename Grid, int EntityDimension, typename Config >
__cuda_callable__
typename Grid::PointType
getEntityCenter( const Grid& grid, const GridEntity< Grid, EntityDimension, Config >& entity )
{
   return entity.getCenter();
}

template< typename MeshConfig, typename Device >
__cuda_callable__
typename MeshTraits< MeshConfig >::PointType
getEntityCenter( const Mesh< MeshConfig, Device >& mesh, const MeshEntity< MeshConfig, Device, Topologies::Vertex >& entity )
{
   return entity.getPoint();
}

/*
 * Get an arithmetic mean of the entity's subvertices.
 *
 * For a simplex entity this corresponds to the centroid of the entity, but
 * note that other shapes such as general polygons have different formulas for
 * the centroid: https://en.wikipedia.org/wiki/Centroid#Centroid_of_a_polygon
 */
template< typename MeshConfig, typename Device, typename EntityTopology >
__cuda_callable__
typename MeshTraits< MeshConfig >::PointType
getEntityCenter( const Mesh< MeshConfig, Device >& mesh, const MeshEntity< MeshConfig, Device, EntityTopology >& entity )
{
   using EntityType = MeshEntity< MeshConfig, Device, EntityTopology >;
   const typename MeshConfig::LocalIndexType subvertices = entity.template getSubentitiesCount< 0 >();
   typename MeshTraits< MeshConfig >::PointType c = 0;
   for( typename MeshConfig::LocalIndexType i = 0; i < subvertices; i++ ) {
      c += mesh.getPoint( entity.template getSubentityIndex< 0 >( i ) );
   }
   return ( 1.0 / subvertices ) * c;
}

}  // namespace Meshes
}  // namespace noa::TNL
