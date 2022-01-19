// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Pointers/DevicePointer.h>
#include <TNL/Meshes/DimensionTag.h>
#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>

#include "Traits.h"

namespace TNL {
namespace Meshes {
namespace EntityTags {

template< typename MeshConfig, typename Device, typename Mesh >
class Initializer
{
   using DeviceType      = Device;
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;
   using LocalIndexType  = typename MeshConfig::LocalIndexType;

protected:
   // _T is necessary to force *partial* specialization, since explicit specializations
   // at class scope are forbidden
   template< typename CurrentDimension = DimensionTag< MeshConfig::meshDimension >, typename _T = void >
   struct EntityTagsNeedInitialization
   {
      static constexpr bool value = MeshConfig::entityTagsStorage( CurrentDimension::value ) ||
                                    EntityTagsNeedInitialization< typename CurrentDimension::Decrement >::value;
   };

   template< typename _T >
   struct EntityTagsNeedInitialization< DimensionTag< 0 >, _T >
   {
      static constexpr bool value = MeshConfig::entityTagsStorage( 0 );
   };

   template< int Dimension >
   class ResetEntityTags
   {
      using WeakTrait = WeakStorageTrait< MeshConfig, Device, DimensionTag< Dimension > >;
      static constexpr bool enabled = WeakTrait::entityTagsEnabled;

      // _T is necessary to force *partial* specialization, since explicit specializations
      // at class scope are forbidden
      template< bool enabled = true, typename _T = void >
      struct Worker
      {
         static void exec( Mesh& mesh )
         {
            mesh.template getEntityTagsView< Dimension >().setValue( 0 );
         }
      };

      template< typename _T >
      struct Worker< false, _T >
      {
         static void exec( Mesh& mesh ) {}
      };

   public:
      static void exec( Mesh& mesh )
      {
         Worker< enabled >::exec( mesh );
      }
   };

   template< int Subdimension >
   class InitializeSubentities
   {
      static constexpr bool enabled = MeshConfig::entityTagsStorage( Subdimension );

      // _T is necessary to force *partial* specialization, since explicit specializations
      // at class scope are forbidden
      template< bool enabled = true, typename _T = void >
      struct Worker
      {
         __cuda_callable__
         static void exec( Mesh& mesh, const GlobalIndexType& faceIndex, const typename Mesh::Face& face )
         {
            const LocalIndexType subentitiesCount = face.template getSubentitiesCount< Subdimension >();
            for( LocalIndexType i = 0; i < subentitiesCount; i++ ) {
               const GlobalIndexType subentityIndex = face.template getSubentityIndex< Subdimension >( i );
               mesh.template addEntityTag< Subdimension >( subentityIndex, EntityTags::BoundaryEntity );
            }
         }
      };

      template< typename _T >
      struct Worker< false, _T >
      {
         __cuda_callable__
         static void exec( Mesh& mesh, const GlobalIndexType& faceIndex, const typename Mesh::Face& face ) {}
      };

   public:
      __cuda_callable__
      static void exec( Mesh& mesh, const GlobalIndexType& faceIndex, const typename Mesh::Face& face )
      {
         Worker< enabled >::exec( mesh, faceIndex, face );
      }
   };

// nvcc does not allow __cuda_callable__ lambdas inside private or protected sections
#ifdef __NVCC__
public:
#endif
   // _T is necessary to force *partial* specialization, since explicit specializations
   // at class scope are forbidden
   template< bool AnyEntityTags = EntityTagsNeedInitialization<>::value, typename _T = void >
   class Worker
   {
   public:
      static void exec( Mesh& mesh )
      {
         // set entities count
         Algorithms::staticFor< int, 0, Mesh::getMeshDimension() + 1 >(
            [&mesh] ( auto dim ) {
               mesh.template entityTagsSetEntitiesCount< dim >( mesh.template getEntitiesCount< dim >() );
            }
         );

         // reset entity tags
         Algorithms::staticFor< int, 0, Mesh::getMeshDimension() + 1 >(
            [&mesh] ( auto dim ) {
               ResetEntityTags< dim >::exec( mesh );
            }
         );

         auto kernel = [] __cuda_callable__
            ( GlobalIndexType faceIndex,
              Mesh* mesh )
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
                  [&mesh, faceIndex, &face] ( auto dim ) {
                     InitializeSubentities< dim >::exec( *mesh, faceIndex, face );
                  }
               );
            }
         };

         const GlobalIndexType facesCount = mesh.template getEntitiesCount< Mesh::getMeshDimension() - 1 >();
         Pointers::DevicePointer< Mesh > meshPointer( mesh );
         Algorithms::ParallelFor< DeviceType >::exec( (GlobalIndexType) 0, facesCount,
                                                      kernel,
                                                      &meshPointer.template modifyData< DeviceType >() );

         // update entity tags
         Algorithms::staticFor< int, 0, Mesh::getMeshDimension() + 1 >(
            [&mesh] ( auto dim ) {
               mesh.template updateEntityTagsLayer< dim >();
            }
         );
      }
   };

   template< typename _T >
   struct Worker< false, _T >
   {
      static void exec( Mesh& mesh ) {}
   };

public:
   void initLayer()
   {
      Worker<>::exec( *static_cast<Mesh*>(this) );
   }
};

} // namespace EntityTags
} // namespace Meshes
} // namespace TNL
