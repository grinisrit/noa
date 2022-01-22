// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Traverser.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>

namespace noaTNL {
namespace Meshes {

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processBoundaryEntities( const MeshPointer& meshPointer,
                         UserData userData ) const
{
   const auto boundaryIndices = meshPointer->template getBoundaryIndices< MeshEntity::getEntityDimension() >();
   const GlobalIndexType entitiesCount = boundaryIndices.getSize();
   auto kernel = [boundaryIndices] __cuda_callable__
      ( const GlobalIndexType i,
        const Mesh* mesh,
        UserData userData )
   {
      const GlobalIndexType entityIndex = boundaryIndices[ i ];
      const auto entity = mesh->template getEntity< MeshEntity::getEntityDimension() >( entityIndex );
      EntitiesProcessor::processEntity( *mesh, userData, entity );
   };
   Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
   Algorithms::ParallelFor< DeviceType >::exec(
         (GlobalIndexType) 0, entitiesCount,
         kernel,
         &meshPointer.template getData< DeviceType >(),
         userData );
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processInteriorEntities( const MeshPointer& meshPointer,
                         UserData userData ) const
{
   const auto interiorIndices = meshPointer->template getInteriorIndices< MeshEntity::getEntityDimension() >();
   const GlobalIndexType entitiesCount = interiorIndices.getSize();
   auto kernel = [interiorIndices] __cuda_callable__
      ( const GlobalIndexType i,
        const Mesh* mesh,
        UserData userData )
   {
      const GlobalIndexType entityIndex = interiorIndices[ i ];
      const auto entity = mesh->template getEntity< MeshEntity::getEntityDimension() >( entityIndex );
      EntitiesProcessor::processEntity( *mesh, userData, entity );
   };
   Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
   Algorithms::ParallelFor< DeviceType >::exec(
         (GlobalIndexType) 0, entitiesCount,
         kernel,
         &meshPointer.template getData< DeviceType >(),
         userData );
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processAllEntities( const MeshPointer& meshPointer,
                    UserData userData ) const
{
   const GlobalIndexType entitiesCount = meshPointer->template getEntitiesCount< MeshEntity::getEntityDimension() >();
   auto kernel = [] __cuda_callable__
      ( const GlobalIndexType entityIndex,
        const Mesh* mesh,
        UserData userData )
   {
      const auto entity = mesh->template getEntity< MeshEntity::getEntityDimension() >( entityIndex );
      EntitiesProcessor::processEntity( *mesh, userData, entity );
   };
   Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
   Algorithms::ParallelFor< DeviceType >::exec(
         (GlobalIndexType) 0, entitiesCount,
         kernel,
         &meshPointer.template getData< DeviceType >(),
         userData );
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processGhostEntities( const MeshPointer& meshPointer,
                       UserData userData ) const
{
   const GlobalIndexType ghostsOffset = meshPointer->template getGhostEntitiesOffset< MeshEntity::getEntityDimension() >();
   const GlobalIndexType entitiesCount = meshPointer->template getEntitiesCount< MeshEntity::getEntityDimension() >();
   auto kernel = [] __cuda_callable__
      ( const GlobalIndexType entityIndex,
        const Mesh* mesh,
        UserData userData )
   {
      const auto entity = mesh->template getEntity< MeshEntity::getEntityDimension() >( entityIndex );
      EntitiesProcessor::processEntity( *mesh, userData, entity );
   };
   Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
   Algorithms::ParallelFor< DeviceType >::exec(
         ghostsOffset, entitiesCount,
         kernel,
         &meshPointer.template getData< DeviceType >(),
         userData );
}

template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Mesh, MeshEntity, EntitiesDimension >::
processLocalEntities( const MeshPointer& meshPointer,
                      UserData userData ) const
{
   const GlobalIndexType ghostsOffset = meshPointer->template getGhostEntitiesOffset< MeshEntity::getEntityDimension() >();
   auto kernel = [] __cuda_callable__
      ( const GlobalIndexType entityIndex,
        const Mesh* mesh,
        UserData userData )
   {
      const auto entity = mesh->template getEntity< MeshEntity::getEntityDimension() >( entityIndex );
      EntitiesProcessor::processEntity( *mesh, userData, entity );
   };
   Pointers::synchronizeSmartPointersOnDevice< DeviceType >();
   Algorithms::ParallelFor< DeviceType >::exec(
         (GlobalIndexType) 0, ghostsOffset,
         kernel,
         &meshPointer.template getData< DeviceType >(),
         userData );
}

} // namespace Meshes
} // namespace noaTNL
