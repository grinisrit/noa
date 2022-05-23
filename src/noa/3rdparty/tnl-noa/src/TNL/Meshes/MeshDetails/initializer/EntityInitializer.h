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

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/EntitySeed.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/SubentitySeedsCreator.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Atomic.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/AtomicOperations.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig >
class Initializer;

template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperdimensionTag,
          typename SuperentityTopology =
             typename MeshTraits< MeshConfig >::template EntityTraits< SuperdimensionTag::value >::EntityTopology,
          // storage in the superentity
          bool SubentityStorage = MeshConfig::subentityStorage( SuperdimensionTag::value, SubdimensionTag::value ),
          // storage in the subentity
          bool SuperentityStorage = MeshConfig::superentityStorage( SubdimensionTag::value, SuperdimensionTag::value ),
          // necessary to disambiguate the stop condition for specializations
          bool valid_dimension = ! std::is_same< SubdimensionTag, SuperdimensionTag >::value >
class EntityInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology,
          bool SubvertexStorage = MeshConfig::subentityStorage( EntityTopology::dimension, 0 ) >
class EntityInitializer : public EntityInitializerLayer< MeshConfig,
                                                         DimensionTag< EntityTopology::dimension >,
                                                         DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
   using BaseType = EntityInitializerLayer< MeshConfig,
                                            DimensionTag< EntityTopology::dimension >,
                                            DimensionTag< MeshTraits< MeshConfig >::meshDimension > >;

   using MeshTraitsType = MeshTraits< MeshConfig >;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;

   using SeedType = EntitySeed< MeshConfig, EntityTopology >;
   using InitializerType = Initializer< MeshConfig >;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedMatrixType = typename EntityTraitsType::SeedMatrixType;

public:
   static void
   initSubvertexMatrix( NeighborCountsArray& capacities, InitializerType& initializer )
   {
      initializer.template initSubentityMatrix< EntityTopology::dimension, 0 >( capacities );
   }

   static void
   initSubvertexMatrix( SeedMatrixType& seeds, InitializerType& initializer )
   {
      auto& subvertexMatrix = initializer.template getSubentitiesMatrix< EntityTopology::dimension, 0 >();
      subvertexMatrix = std::move( seeds.getMatrix() );
      initializer.template initSubentitiesCounts< EntityTopology::dimension, 0 >( seeds.getEntityCornerCounts() );
   }

   static void
   initEntity( const GlobalIndexType entityIndex, const SeedType& entitySeed, InitializerType& initializer )
   {
      // this is necessary if we want to use existing entities instead of intermediate seeds to create subentity seeds
      for( LocalIndexType i = 0; i < entitySeed.getCornerIds().getSize(); i++ )
         initializer.template setSubentityIndex< EntityTopology::dimension, 0 >(
            entityIndex, i, entitySeed.getCornerIds()[ i ] );
   }
};

template< typename MeshConfig, typename EntityTopology >
class EntityInitializer< MeshConfig, EntityTopology, false >
: public EntityInitializerLayer< MeshConfig,
                                 DimensionTag< EntityTopology::dimension >,
                                 DimensionTag< MeshTraits< MeshConfig >::meshDimension > >
{
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< EntityTopology::dimension >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;

   using SeedType = EntitySeed< MeshConfig, EntityTopology >;
   using InitializerType = Initializer< MeshConfig >;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedMatrixType = typename EntityTraitsType::SeedMatrixType;

public:
   static void
   initSubvertexMatrix( const NeighborCountsArray& capacities, InitializerType& initializer )
   {}
   static void
   initSubvertexMatrix( SeedMatrixType& seeds, InitializerType& initializer )
   {}
   static void
   initEntity( const GlobalIndexType entityIndex, const SeedType& entitySeed, InitializerType& initializer )
   {}
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE
 *      TRUE                    TRUE
 */
template< typename MeshConfig, typename SubdimensionTag, typename SuperdimensionTag, typename SuperentityTopology >
class EntityInitializerLayer< MeshConfig, SubdimensionTag, SuperdimensionTag, SuperentityTopology, true, true, true >
: public EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >;
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;
   using MeshTraitsType = MeshTraits< MeshConfig >;

   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityTraitsType = typename MeshTraitsType::template EntityTraits< SubdimensionTag::value >;
   using SubentityTopology = typename SubentityTraitsType::EntityTopology;
   using SuperentityTraitsType = typename MeshTraitsType::template EntityTraits< SuperdimensionTag::value >;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshConfig, SuperentityTopology, SubdimensionTag >;
   using SuperentityMatrixType = typename MeshTraitsType::SuperentityMatrixType;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedType = EntitySeed< MeshConfig, SubentityTopology >;

public:
   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      // std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with
      // dimension " << SubdimensionTag::value << " ... " << std::endl;

      const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< SubdimensionTag::value >();
      const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();

      if( SubdimensionTag::value > 0 || std::is_same< SuperentityTopology, Topologies::Polyhedron >::value ) {
         NeighborCountsArray capacities( superentitiesCount );

         Algorithms::ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 },
                                                         superentitiesCount,
                                                         [ & ]( GlobalIndexType superentityIndex )
                                                         {
                                                            capacities[ superentityIndex ] =
                                                               SubentitySeedsCreatorType::getSubentitiesCount(
                                                                  meshInitializer, mesh, superentityIndex );
                                                         } );

         meshInitializer.template initSubentityMatrix< SuperdimensionTag::value, SubdimensionTag::value >( capacities,
                                                                                                           subentitiesCount );
      }

      // counter for superentities of each subentity
      auto& superentitiesCounts =
         meshInitializer.template getSuperentitiesCountsArray< SubdimensionTag::value, SuperdimensionTag::value >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );

      Algorithms::ParallelFor< Devices::Host >::exec(
         GlobalIndexType{ 0 },
         superentitiesCount,
         [ & ]( GlobalIndexType superentityIndex )
         {
            LocalIndexType i = 0;
            SubentitySeedsCreatorType::iterate(
               meshInitializer,
               mesh,
               superentityIndex,
               [ & ]( SeedType& seed )
               {
                  const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( seed );

                  // Subentity indices for SubdimensionTag::value == 0 of non-polyhedral meshes were already set up from seeds
                  if( SubdimensionTag::value > 0 || std::is_same< SuperentityTopology, Topologies::Polyhedron >::value )
                     meshInitializer.template setSubentityIndex< SuperdimensionTag::value, SubdimensionTag::value >(
                        superentityIndex, i++, subentityIndex );

                  Algorithms::AtomicOperations< Devices::Host >::add( superentitiesCounts[ subentityIndex ],
                                                                      LocalIndexType{ 1 } );
               } );
         } );

      // allocate superentities storage
      SuperentityMatrixType& matrix =
         meshInitializer.template getSuperentitiesMatrix< SubdimensionTag::value, SuperdimensionTag::value >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCounts );
      superentitiesCounts.setValue( 0 );

      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         for( LocalIndexType i = 0;
              i < mesh.template getSubentitiesCount< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex );
              i++ )
         {
            const GlobalIndexType subentityIndex =
               mesh.template getSubentityIndex< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex, i );
            auto row = matrix.getRow( subentityIndex );
            row.setElement( superentitiesCounts[ subentityIndex ]++, superentityIndex, true );
         }
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE     Subdimension     Superdimension     SUPERENTITY TOPOLOGY
 *      TRUE                    TRUE                  2                 3                POLYHEDRON
 */
template< typename MeshConfig >
class EntityInitializerLayer< MeshConfig, DimensionTag< 2 >, DimensionTag< 3 >, Topologies::Polyhedron, true, true, true >
: public EntityInitializerLayer< MeshConfig, DimensionTag< 2 >, typename DimensionTag< 3 >::Decrement >
{
   using SubdimensionTag = DimensionTag< 2 >;
   using SuperdimensionTag = DimensionTag< 3 >;

   using BaseType = EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >;
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;
   using MeshTraitsType = MeshTraits< MeshConfig >;

   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityTraitsType = typename MeshTraitsType::template EntityTraits< SubdimensionTag::value >;
   using SubentityTopology = typename SubentityTraitsType::EntityTopology;
   using SuperentityTraitsType = typename MeshTraitsType::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology = typename SuperentityTraitsType::EntityTopology;
   using SuperentityMatrixType = typename MeshTraitsType::SuperentityMatrixType;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedType = EntitySeed< MeshConfig, SubentityTopology >;

public:
   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      // std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with
      // dimension " << SubdimensionTag::value << " ... " << std::endl;

      const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< SubdimensionTag::value >();
      const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();

      // counter for superentities of each subentity
      auto& superentitiesCounts =
         meshInitializer.template getSuperentitiesCountsArray< SubdimensionTag::value, SuperdimensionTag::value >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );

      auto& cellSeeds = meshInitializer.getCellSeeds();
      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         const auto cellSeed = cellSeeds.getSeed( superentityIndex );
         for( LocalIndexType i = 0; i < cellSeed.getCornersCount(); i++ ) {
            const GlobalIndexType subentityIndex = cellSeed.getCornerId( i );
            superentitiesCounts[ subentityIndex ]++;
         }
      }

      auto& subvertexMatrix =
         meshInitializer.template getSubentitiesMatrix< SuperdimensionTag::value, SubdimensionTag::value >();
      subvertexMatrix = std::move( cellSeeds.getMatrix() );
      meshInitializer.template initSubentitiesCounts< SuperdimensionTag::value, SubdimensionTag::value >(
         cellSeeds.getEntityCornerCounts() );

      // allocate superentities storage
      SuperentityMatrixType& matrix =
         meshInitializer.template getSuperentitiesMatrix< SubdimensionTag::value, SuperdimensionTag::value >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCounts );
      superentitiesCounts.setValue( 0 );

      // initialize superentities storage
      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         for( LocalIndexType i = 0;
              i < mesh.template getSubentitiesCount< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex );
              i++ )
         {
            const GlobalIndexType subentityIndex =
               mesh.template getSubentityIndex< SuperdimensionTag::value, SubdimensionTag::value >( superentityIndex, i );
            auto row = matrix.getRow( subentityIndex );
            row.setElement( superentitiesCounts[ subentityIndex ]++, superentityIndex, true );
         }
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE
 *      TRUE                   FALSE
 */
template< typename MeshConfig, typename SubdimensionTag, typename SuperdimensionTag, typename SuperentityTopology >
class EntityInitializerLayer< MeshConfig, SubdimensionTag, SuperdimensionTag, SuperentityTopology, true, false, true >
: public EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >;
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;
   using MeshTraitsType = MeshTraits< MeshConfig >;

   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityTraitsType = typename MeshTraitsType::template EntityTraits< SubdimensionTag::value >;
   using SubentityTopology = typename SubentityTraitsType::EntityTopology;
   using SuperentityTraitsType = typename MeshTraitsType::template EntityTraits< SuperdimensionTag::value >;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshConfig, SuperentityTopology, SubdimensionTag >;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedType = EntitySeed< MeshConfig, SubentityTopology >;

public:
   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      // std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with
      // dimension " << SubdimensionTag::value << " ... " << std::endl;

      const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< SubdimensionTag::value >();
      const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();
      if( SubdimensionTag::value > 0 || std::is_same< SuperentityTopology, Topologies::Polyhedron >::value ) {
         NeighborCountsArray capacities( superentitiesCount );

         Algorithms::ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 },
                                                         superentitiesCount,
                                                         [ & ]( GlobalIndexType superentityIndex )
                                                         {
                                                            capacities[ superentityIndex ] =
                                                               SubentitySeedsCreatorType::getSubentitiesCount(
                                                                  meshInitializer, mesh, superentityIndex );
                                                         } );

         meshInitializer.template initSubentityMatrix< SuperdimensionTag::value, SubdimensionTag::value >( capacities,
                                                                                                           subentitiesCount );

         Algorithms::ParallelFor< Devices::Host >::exec(
            GlobalIndexType{ 0 },
            superentitiesCount,
            [ & ]( GlobalIndexType superentityIndex )
            {
               LocalIndexType i = 0;
               SubentitySeedsCreatorType::iterate(
                  meshInitializer,
                  mesh,
                  superentityIndex,
                  [ & ]( SeedType& seed )
                  {
                     const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( seed );
                     meshInitializer.template setSubentityIndex< SuperdimensionTag::value, SubdimensionTag::value >(
                        superentityIndex, i++, subentityIndex );
                  } );
            } );
      }

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE     Subdimension     Superdimension     SUPERENTITY TOPOLOGY
 *      TRUE                   FALSE                   2                 3                POLYHEDRON
 */
template< typename MeshConfig >
class EntityInitializerLayer< MeshConfig, DimensionTag< 2 >, DimensionTag< 3 >, Topologies::Polyhedron, true, false, true >
: public EntityInitializerLayer< MeshConfig, DimensionTag< 2 >, typename DimensionTag< 3 >::Decrement >
{
   using SubdimensionTag = DimensionTag< 2 >;
   using SuperdimensionTag = DimensionTag< 3 >;

   using BaseType = EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >;
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;
   using MeshTraitsType = MeshTraits< MeshConfig >;

   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityTraitsType = typename MeshTraitsType::template EntityTraits< SubdimensionTag::value >;
   using SubentityTopology = typename SubentityTraitsType::EntityTopology;
   using SuperentityTraitsType = typename MeshTraitsType::template EntityTraits< SuperdimensionTag::value >;
   using SuperentityTopology = typename SuperentityTraitsType::EntityTopology;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SeedType = EntitySeed< MeshConfig, SubentityTopology >;

public:
   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      // std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with
      // dimension " << SubdimensionTag::value << " ... " << std::endl;
      auto& cellSeeds = meshInitializer.getCellSeeds();
      auto& subvertexMatrix =
         meshInitializer.template getSubentitiesMatrix< SuperdimensionTag::value, SubdimensionTag::value >();
      subvertexMatrix = std::move( cellSeeds.getMatrix() );
      meshInitializer.template initSubentitiesCounts< SuperdimensionTag::value, SubdimensionTag::value >(
         cellSeeds.getEntityCornerCounts() );
      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

/****
 *       Mesh entity initializer layer with specializations
 *
 *  SUBENTITY STORAGE     SUPERENTITY STORAGE
 *      FALSE                  TRUE
 */
template< typename MeshConfig, typename SubdimensionTag, typename SuperdimensionTag, typename SuperentityTopology >
class EntityInitializerLayer< MeshConfig, SubdimensionTag, SuperdimensionTag, SuperentityTopology, false, true, true >
: public EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >
{
   using BaseType = EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >;
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;
   using MeshTraitsType = MeshTraits< MeshConfig >;

   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using SubentityTraitsType = typename MeshTraitsType::template EntityTraits< SubdimensionTag::value >;
   using SubentityTopology = typename SubentityTraitsType::EntityTopology;
   using SuperentityTraitsType = typename MeshTraitsType::template EntityTraits< SuperdimensionTag::value >;
   using SubentitySeedsCreatorType = SubentitySeedsCreator< MeshConfig, SuperentityTopology, SubdimensionTag >;
   using SuperentityMatrixType = typename MeshTraitsType::SuperentityMatrixType;
   using SeedType = EntitySeed< MeshConfig, SubentityTopology >;

public:
   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {
      // std::cout << "   Initiating superentities with dimension " << SuperdimensionTag::value << " for subentities with
      // dimension " << SubdimensionTag::value << " ... " << std::endl;

      const GlobalIndexType subentitiesCount = mesh.template getEntitiesCount< SubdimensionTag::value >();
      const GlobalIndexType superentitiesCount = mesh.template getEntitiesCount< SuperdimensionTag::value >();

      // counter for superentities of each subentity
      auto& superentitiesCounts =
         meshInitializer.template getSuperentitiesCountsArray< SubdimensionTag::value, SuperdimensionTag::value >();
      superentitiesCounts.setSize( subentitiesCount );
      superentitiesCounts.setValue( 0 );

      Algorithms::ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 },
                                                      superentitiesCount,
                                                      [ & ]( GlobalIndexType superentityIndex )
                                                      {
                                                         SubentitySeedsCreatorType::iterate(
                                                            meshInitializer,
                                                            mesh,
                                                            superentityIndex,
                                                            [ & ]( SeedType& seed )
                                                            {
                                                               const GlobalIndexType subentityIndex =
                                                                  meshInitializer.findEntitySeedIndex( seed );
                                                               Algorithms::AtomicOperations< Devices::Host >::add(
                                                                  superentitiesCounts[ subentityIndex ], LocalIndexType{ 1 } );
                                                            } );
                                                      } );

      // allocate superentities storage
      SuperentityMatrixType& matrix =
         meshInitializer.template getSuperentitiesMatrix< SubdimensionTag::value, SuperdimensionTag::value >();
      matrix.setDimensions( subentitiesCount, superentitiesCount );
      matrix.setRowCapacities( superentitiesCounts );
      superentitiesCounts.setValue( 0 );

      // initialize superentities storage
      for( GlobalIndexType superentityIndex = 0; superentityIndex < superentitiesCount; superentityIndex++ ) {
         SubentitySeedsCreatorType::iterate(
            meshInitializer,
            mesh,
            superentityIndex,
            [ & ]( SeedType& seed )
            {
               const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( seed );
               auto row = matrix.getRow( subentityIndex );
               row.setElement( superentitiesCounts[ subentityIndex ]++, superentityIndex, true );
            } );
      }

      // Here is an attempt of parallelization of previous for cycle, that seemingly causes some kind of race condition
      /*Algorithms::ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, superentitiesCount, [&] ( GlobalIndexType
      superentityIndex )
      {
         SubentitySeedsCreatorType::iterate( meshInitializer, mesh, superentityIndex, [&] ( SeedType& seed ) {
            const GlobalIndexType subentityIndex = meshInitializer.findEntitySeedIndex( seed );
            auto row = matrix.getRow( subentityIndex );

            LocalIndexType superentityCount;
            #pragma omp atomic capture
            {
               superentityCount = superentitiesCounts[ subentityIndex ];
               superentitiesCounts[ subentityIndex ]++;
            }

            row.setElement( superentityCount, superentityIndex, true );
         });
      });*/

      BaseType::initSuperentities( meshInitializer, mesh );
   }
};

template< typename MeshConfig, typename SubdimensionTag, typename SuperdimensionTag, typename SuperentityTopology >
class EntityInitializerLayer< MeshConfig, SubdimensionTag, SuperdimensionTag, SuperentityTopology, false, false, true >
: public EntityInitializerLayer< MeshConfig, SubdimensionTag, typename SuperdimensionTag::Decrement >
{};

template< typename MeshConfig,
          typename SubdimensionTag,
          typename SuperentityTopology,
          bool SubentityStorage,
          bool SuperentityStorage >
class EntityInitializerLayer< MeshConfig,
                              SubdimensionTag,
                              SubdimensionTag,
                              SuperentityTopology,
                              SubentityStorage,
                              SuperentityStorage,
                              false >
{
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = typename InitializerType::MeshType;

public:
   static void
   initSuperentities( InitializerType& meshInitializer, MeshType& mesh )
   {}
};

}  // namespace Meshes
}  // namespace noa::TNL
