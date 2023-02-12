// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/SubentitySeedsCreator.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/EntitySeed.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Atomic.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/AtomicOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/staticFor.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig >
class Initializer;

template< int subdimension, int superdimension, typename MeshConfig, typename SeedIndexGetter >
void
initializeSuperentities( Initializer< MeshConfig >& meshInitializer,
                         typename Initializer< MeshConfig >::MeshType& mesh,
                         SeedIndexGetter&& getEntitySeedIndex )
{
   using MeshType = typename Initializer< MeshConfig >::MeshType;
   using GlobalIndexType = typename MeshType::GlobalIndexType;
   using LocalIndexType = typename MeshType::LocalIndexType;
   using SuperentityMatrixType = typename MeshType::MeshTraitsType::SuperentityMatrixType;
   using NeighborCountsArray = typename MeshType::MeshTraitsType::NeighborCountsArray;
   using SuperentityTopology = typename MeshType::MeshTraitsType::template EntityTraits< superdimension >::EntityTopology;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshType, SuperentityTopology, DimensionTag< subdimension > >;
   using SeedType = typename SubentitySeedsCreatorType::SubentitySeed;

   static constexpr bool subentityStorage = MeshConfig::subentityStorage( superdimension, subdimension );
   static constexpr bool superentityStorage = MeshConfig::superentityStorage( subdimension, superdimension );

   // std::cout << "   Initiating superentities with dimension " << superdimension << " for subentities with
   // dimension " << subdimension << " ... " << std::endl;

   const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< subdimension >();
   const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< superdimension >();

   if constexpr( subentityStorage && (subdimension > 0 || std::is_same_v< SuperentityTopology, Topologies::Polyhedron >) ) {
      NeighborCountsArray capacities( superentitiesCount );

      Algorithms::ParallelFor< Devices::Host >::exec(
         GlobalIndexType{ 0 },
         superentitiesCount,
         [ & ]( GlobalIndexType superentityIndex )
         {
            capacities[ superentityIndex ] = SubentitySeedsCreatorType::getSubentitiesCount( mesh, superentityIndex );
         } );

      meshInitializer.template initSubentityMatrix< superdimension, subdimension >( capacities, subentitiesCount );
   }

   typename NeighborCountsArray::ViewType superentitiesCountsView;
   if constexpr( superentityStorage ) {
      // counter for superentities of each subentity
      NeighborCountsArray& superentitiesCounts =
         meshInitializer.template getSuperentitiesCountsArray< subdimension, superdimension >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );
      superentitiesCountsView.bind( superentitiesCounts );
   }

   if constexpr( subentityStorage || superentityStorage ) {
      Algorithms::ParallelFor< Devices::Host >::exec(
         GlobalIndexType{ 0 },
         superentitiesCount,
         [ & ]( GlobalIndexType superentityIndex )
         {
            LocalIndexType i = 0;
            SubentitySeedsCreatorType::iterate(
               mesh,
               superentityIndex,
               [ & ]( SeedType& seed )
               {
                  const GlobalIndexType subentityIndex = getEntitySeedIndex( seed );

                  // Subentity indices for subdimension == 0 of non-polyhedral meshes were already set up from seeds
                  if constexpr( subentityStorage
                                && (subdimension > 0 || std::is_same_v< SuperentityTopology, Topologies::Polyhedron >) )
                     meshInitializer.template setSubentityIndex< superdimension, subdimension >(
                        superentityIndex, i++, subentityIndex );

                  if constexpr( superentityStorage ) {
                     Algorithms::AtomicOperations< Devices::Host >::add( superentitiesCountsView[ subentityIndex ],
                                                                         LocalIndexType{ 1 } );
                  }
               } );
         } );
   }

   if constexpr( superentityStorage ) {
      // allocate superentities storage
      SuperentityMatrixType& matrix = meshInitializer.template getSuperentitiesMatrix< subdimension, superdimension >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCountsView );
      superentitiesCountsView.setValue( 0 );

      // initialize superentities storage
      if constexpr( subentityStorage ) {
         for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
            for( LocalIndexType i = 0;
                 i < mesh.template getSubentitiesCount< superdimension, subdimension >( superentityIndex );
                 i++ ) {
               const GlobalIndexType subentityIndex =
                  mesh.template getSubentityIndex< superdimension, subdimension >( superentityIndex, i );
               auto row = matrix.getRow( subentityIndex );
               row.setElement( superentitiesCountsView[ subentityIndex ]++, superentityIndex, true );
            }
         }
      }
      else {
         for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
            SubentitySeedsCreatorType::iterate( mesh,
                                                superentityIndex,
                                                [ & ]( SeedType& seed )
                                                {
                                                   const GlobalIndexType subentityIndex = getEntitySeedIndex( seed );
                                                   auto row = matrix.getRow( subentityIndex );
                                                   row.setElement(
                                                      superentitiesCountsView[ subentityIndex ]++, superentityIndex, true );
                                                } );
         }
      }
   }
}

template< typename MeshConfig >
void
initializeFacesOfPolyhedrons( Initializer< MeshConfig >& meshInitializer,
                              typename Initializer< MeshConfig >::MeshType::MeshTraitsType::CellSeedMatrixType& cellSeeds,
                              typename Initializer< MeshConfig >::MeshType& mesh )
{
   using MeshType = typename Initializer< MeshConfig >::MeshType;
   using GlobalIndexType = typename MeshType::GlobalIndexType;
   using LocalIndexType = typename MeshType::LocalIndexType;
   using SuperentityMatrixType = typename MeshType::MeshTraitsType::SuperentityMatrixType;
   using NeighborCountsArray = typename MeshType::MeshTraitsType::NeighborCountsArray;

   static constexpr int subdimension = 2;
   static constexpr int superdimension = 3;
   static constexpr bool subentityStorage = MeshConfig::subentityStorage( superdimension, subdimension );
   static constexpr bool superentityStorage = MeshConfig::superentityStorage( subdimension, superdimension );
   static_assert( subentityStorage );

   // std::cout << "   Initiating superentities with dimension " << superdimension << " for subentities with
   // dimension " << subdimension << " ... " << std::endl;

   meshInitializer.template setEntitiesCount< superdimension >( cellSeeds.getEntitiesCount() );

   const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< subdimension >();
   const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< superdimension >();

   typename NeighborCountsArray::ViewType superentitiesCountsView;
   if constexpr( superentityStorage ) {
      // counter for superentities of each subentity
      NeighborCountsArray& superentitiesCounts =
         meshInitializer.template getSuperentitiesCountsArray< subdimension, superdimension >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );
      superentitiesCountsView.bind( superentitiesCounts );

      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         const auto cellSeed = cellSeeds.getSeed( superentityIndex );
         for( LocalIndexType i = 0; i < cellSeed.getCornersCount(); i++ ) {
            const GlobalIndexType subentityIndex = cellSeed.getCornerId( i );
            superentitiesCountsView[ subentityIndex ]++;
         }
      }
   }

   auto& subvertexMatrix = meshInitializer.template getSubentitiesMatrix< superdimension, subdimension >();
   subvertexMatrix = std::move( cellSeeds.getMatrix() );
   meshInitializer.template initSubentitiesCounts< superdimension, subdimension >( cellSeeds.getEntityCornerCounts() );

   if constexpr( superentityStorage ) {
      // allocate superentities storage
      SuperentityMatrixType& matrix = meshInitializer.template getSuperentitiesMatrix< subdimension, superdimension >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCountsView );
      superentitiesCountsView.setValue( 0 );

      // initialize superentities storage
      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         for( LocalIndexType i = 0; i < mesh.template getSubentitiesCount< superdimension, subdimension >( superentityIndex );
              i++ ) {
            const GlobalIndexType subentityIndex =
               mesh.template getSubentityIndex< superdimension, subdimension >( superentityIndex, i );
            auto row = matrix.getRow( subentityIndex );
            row.setElement( superentitiesCountsView[ subentityIndex ]++, superentityIndex, true );
         }
      }
   }
}

template< typename MeshConfig, int EntityDimension >
class EntityInitializer
{
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;
   using MeshTraitsType = typename MeshType::MeshTraitsType;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< EntityDimension >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;

   using SeedType = EntitySeed< MeshConfig, typename EntityTraitsType::EntityTopology >;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedIndexedSet = typename MeshTraits< MeshConfig >::template EntityTraits< EntityDimension >::SeedIndexedSetType;

   static constexpr bool subvertexStorage = MeshConfig::subentityStorage( EntityDimension, 0 );

public:
   template< typename SeedIndexGetter >
   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh, SeedIndexGetter&& getEntitySeedIndex )
   {
      Algorithms::staticFor< int, EntityDimension + 1, MeshType::getMeshDimension() + 1 >(
         [ & ]( auto dim )
         {
            // transform dim to ensure decrementing steps in the loop
            constexpr int superdimension = MeshType::getMeshDimension() + EntityDimension + 1 - dim;

            initializeSuperentities< EntityDimension, superdimension >( meshInitializer, mesh, getEntitySeedIndex );
         } );
   }

   static void
   createEntities( InitializerType& meshInitializer, MeshType& mesh )
   {
      // std::cout << " Creating entities with dimension " << EntityDimension << " ... " << std::endl;

      // create seeds
      SeedIndexedSet seedsIndexedSet;
      seedsIndexedSet.reserve( mesh.template getEntitiesCount< MeshTraitsType::meshDimension >() );
      using SubentitySeedsCreator =
         SubentitySeedsCreator< MeshType, typename MeshTraitsType::CellTopology, DimensionTag< EntityDimension > >;
      for( GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshType::getMeshDimension() >(); i++ ) {
         SubentitySeedsCreator::iterate( mesh,
                                         i,
                                         [ &seedsIndexedSet ]( SeedType& seed )
                                         {
                                            seedsIndexedSet.insert( std::move( seed ) );
                                         } );
      }

      // set entities count
      const GlobalIndexType numberOfEntities = seedsIndexedSet.size();
      meshInitializer.template setEntitiesCount< EntityDimension >( numberOfEntities );

      auto getEntitySeedIndex = [ &seedsIndexedSet ]( const SeedType& seed )
      {
         GlobalIndexType index = -1;
         if( ! seedsIndexedSet.find( seed, index ) )
            throw std::domain_error( "given seed was not found in the indexed set" );
         return index;
      };

      // allocate the subvertex matrix
      NeighborCountsArray capacities( numberOfEntities );
      for( auto& pair : seedsIndexedSet ) {
         const auto& seed = pair.first;
         const auto& entityIndex = pair.second;
         capacities.setElement( entityIndex, seed.getCornersCount() );
      }
      meshInitializer.template initSubentityMatrix< EntityDimension, 0 >( capacities );

      // initialize the entities (this allows us to create subentity seeds from existing entities instead of intermediate seeds)
      for( auto& pair : seedsIndexedSet ) {
         const auto& seed = pair.first;
         const auto& entityIndex = pair.second;
         for( LocalIndexType i = 0; i < seed.getCornerIds().getSize(); i++ )
            meshInitializer.template setSubentityIndex< EntityDimension, 0 >( entityIndex, i, seed.getCornerIds()[ i ] );
      }

      // initialize links between the entities and all superentities
      initSuperentities( meshInitializer, mesh, getEntitySeedIndex );
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
