// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/detail/VerticesPerEntity.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {
namespace detail {

template< typename Mesh, int EntityDimension, typename EntityType = typename Mesh::template EntityType< EntityDimension > >
struct VTKOffsetsCountGetter
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType
   getOffsetsCount( const Mesh& mesh )
   {
      const IndexType entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const IndexType verticesPerEntity = VerticesPerEntity< EntityType >::count;
      return entitiesCount * verticesPerEntity;
   }
};

template< typename Mesh, int EntityDimension >
struct VTKOffsetsCountGetter< Mesh,
                              EntityDimension,
                              MeshEntity< typename Mesh::Config, typename Mesh::DeviceType, Topologies::Polygon > >
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType
   getOffsetsCount( const Mesh& mesh )
   {
      const IndexType entitiesCount = mesh.template getEntitiesCount< EntityDimension >();
      IndexType offsetsCount = 0;
      for( IndexType index = 0; index < entitiesCount; index++ )
         offsetsCount += mesh.template getSubentitiesCount< EntityDimension, 0 >( index );
      return offsetsCount;
   }
};

template< typename Mesh, int EntityDimension >
struct VTKOffsetsCountGetter< Mesh,
                              EntityDimension,
                              MeshEntity< typename Mesh::Config, typename Mesh::DeviceType, Topologies::Polyhedron > >
{
   using IndexType = typename Mesh::GlobalIndexType;

   static IndexType
   getOffsetsCount( const Mesh& mesh )
   {
      const IndexType entitiesCount = mesh.template getEntitiesCount< EntityDimension >();
      IndexType offsetsCount = 0;
      for( IndexType index = 0; index < entitiesCount; index++ ) {
         const IndexType num_faces = mesh.template getSubentitiesCount< EntityDimension, EntityDimension - 1 >( index );
         // one value (num_faces) for each cell
         offsetsCount++;
         // one value (num_vertices) for each face
         offsetsCount += num_faces;
         // list of vertex indices for each face
         for( IndexType f = 0; f < num_faces; f++ ) {
            const IndexType face = mesh.template getSubentityIndex< EntityDimension, EntityDimension - 1 >( index, f );
            offsetsCount += mesh.template getSubentitiesCount< EntityDimension - 1, 0 >( face );
         }
      }
      return offsetsCount;
   }
};

}  // namespace detail
}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL
