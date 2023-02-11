// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/staticFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/DevicePointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>

#include "Traits.h"

namespace noa::TNL {
namespace Meshes {
namespace EntityTags {

template< typename Mesh >
constexpr bool
entityTagsNeedInitialization()
{
   for( int dim = 0; dim <= Mesh::getMeshDimension(); dim++ )
      if( Mesh::Config::entityTagsStorage( dim ) )
         return true;
   return false;
}

template< typename Mesh >
void
initializeEntityTags( Mesh& mesh )
{
   using DeviceType = typename Mesh::DeviceType;
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using LocalIndexType = typename Mesh::LocalIndexType;

   if constexpr( entityTagsNeedInitialization< Mesh >() ) {
      // set entities count
      Algorithms::staticFor< int, 0, Mesh::getMeshDimension() + 1 >(
         [ &mesh ]( auto dim )
         {
            mesh.template entityTagsSetEntitiesCount< dim >( mesh.template getEntitiesCount< dim >() );
         } );

      // reset entity tags
      Algorithms::staticFor< int, 0, Mesh::getMeshDimension() + 1 >(
         [ &mesh ]( auto dim )
         {
            using WeakTrait = WeakStorageTrait< typename Mesh::Config, DeviceType, DimensionTag< dim > >;
            if constexpr( WeakTrait::entityTagsEnabled ) {
               mesh.template getEntityTagsView< dim >().setValue( 0 );
            }
         } );

      auto kernel = [] __cuda_callable__( GlobalIndexType faceIndex, Mesh * mesh )
      {
         const auto& face = mesh->template getEntity< Mesh::getMeshDimension() - 1 >( faceIndex );
         if( face.template getSuperentitiesCount< Mesh::getMeshDimension() >() == 1 ) {
            // initialize the face
            mesh->template addEntityTag< Mesh::getMeshDimension() - 1 >( faceIndex, EntityTags::BoundaryEntity );
            // initialize the cell superentity
            const GlobalIndexType cellIndex = face.template getSuperentityIndex< Mesh::getMeshDimension() >( 0 );
            mesh->template addEntityTag< Mesh::getMeshDimension() >( cellIndex, EntityTags::BoundaryEntity );
            // initialize all subentities
            Algorithms::staticFor< int, 0, Mesh::getMeshDimension() - 1 >(
               [ &mesh, &face ]( auto dim )
               {
                  if constexpr( Mesh::Config::entityTagsStorage( dim ) ) {
                     const LocalIndexType subentitiesCount = face.template getSubentitiesCount< dim >();
                     for( LocalIndexType i = 0; i < subentitiesCount; i++ ) {
                        const GlobalIndexType subentityIndex = face.template getSubentityIndex< dim >( i );
                        mesh->template addEntityTag< dim >( subentityIndex, EntityTags::BoundaryEntity );
                     }
                  }
               } );
         }
      };

      const GlobalIndexType facesCount = mesh.template getEntitiesCount< Mesh::getMeshDimension() - 1 >();
      Pointers::DevicePointer< Mesh > meshPointer( mesh );
      Algorithms::ParallelFor< DeviceType >::exec(
         (GlobalIndexType) 0, facesCount, kernel, &meshPointer.template modifyData< DeviceType >() );

      // update entity tags
      Algorithms::staticFor< int, 0, Mesh::getMeshDimension() + 1 >(
         [ &mesh ]( auto dim )
         {
            mesh.template updateEntityTagsLayer< dim >();
         } );
   }
}

}  // namespace EntityTags
}  // namespace Meshes
}  // namespace noa::TNL
