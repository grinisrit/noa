// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshDetails/initializer/EntityInitializer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>

/*
 * How this beast works:
 *
 * The algorithm is optimized for memory requirements. Therefore, the mesh is
 * not allocated at once, but by parts (by dimensions). The flow is roughly the
 * following:
 *
 *  - Initialize vertices by copying their physical coordinates from the input
 *    array, deallocate the input array of points.
 *  - Initialize cells by copying their subvertex indices from cell seeds (but
 *    other subentity indices are left uninitialized), deallocate cell seeds
 *    (the cells will be used later).
 *  - For all dimensions D from (cell dimension - 1) to 1:
 *     - Create intermediate entity seeds, count the number of entities with
 *       current dimension.
 *     - Set their subvertex indices. Create an indexed set of entity seeds.
 *     - For all superdimensions S > D:
 *        - Iterate over entities with dimension S and initialize their
 *          subentity indices with dimension D. Inverse mapping (D->S) is
 *          recorded in the process.
 *        - For entities with dimension D, initialize their superentity indices
 *          with dimension S.
 *     - Deallocate all intermediate data structures.
 *
 * Optimization notes:
 *   - Recomputing the seed key involves sorting all subvertex indices, but the
 *     cost is negligible compared to memory consumption incurred by storing
 *     both the key and original seed in the indexed set.
 *   - Since std::set and std::map don't provide a shrink_to_fit method like
 *     std::vector, these dynamic structures should be kept as local variables
 *     if possible. This is probably the only way to be sure that the unused
 *     space is not wasted.
 */

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig, typename DimensionTag >
class InitializerLayer;

template< typename MeshConfig >
class Initializer : public InitializerLayer< MeshConfig, typename MeshTraits< MeshConfig >::DimensionTag::Decrement >
{
protected:
   // must be declared before its use in expression with decltype()
   Mesh< MeshConfig >* mesh = nullptr;

public:
   using MeshType = Mesh< MeshConfig >;
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using PointArrayType = typename MeshTraitsType::PointArrayType;
   using CellSeedMatrixType = typename MeshTraitsType::CellSeedMatrixType;
   using FaceSeedMatrixType = typename MeshTraitsType::FaceSeedMatrixType;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

   template< int Dimension, int Subdimension >
   using SubentityMatrixRowsCapacitiesType =
      typename MeshTraitsType::template SubentityMatrixType< Dimension >::RowsCapacitiesType;

   // The points and cellSeeds arrays will be reset when not needed to save memory.
   void
   createMesh( PointArrayType& points, FaceSeedMatrixType& faceSeeds, CellSeedMatrixType& cellSeeds, MeshType& mesh )
   {
      // move points
      mesh.template setEntitiesCount< 0 >( points.getSize() );
      mesh.getPoints() = std::move( points );

      this->mesh = &mesh;
      constexpr int meshDimension = MeshType::getMeshDimension();

      if constexpr( std::is_same_v< typename MeshType::Config::CellTopology, Topologies::Polyhedron > ) {
         // initialize faces
         setEntitiesCount< meshDimension - 1 >( faceSeeds.getEntitiesCount() );
         initSubvertexMatrix< meshDimension - 1 >( faceSeeds );

         // initialize links between faces and cells
         initializeFacesOfPolyhedrons( *this, cellSeeds, mesh );

         // initialize other entities
         using BaseType = InitializerLayer< MeshConfig, DimensionTag< meshDimension - 2 > >;
         BaseType::initEntities( *this, mesh );
      }
      else {
         // initialize cells
         setEntitiesCount< meshDimension >( cellSeeds.getEntitiesCount() );
         initSubvertexMatrix< meshDimension >( cellSeeds );

         // initialize other entities
         using BaseType = InitializerLayer< MeshConfig, DimensionTag< meshDimension - 1 > >;
         BaseType::initEntities( *this, mesh );
      }
   }

   template< int EntityDimension, typename SeedMatrixType >
   void
   initSubvertexMatrix( SeedMatrixType& seeds )
   {
      auto& subvertexMatrix = getSubentitiesMatrix< EntityDimension, 0 >();
      subvertexMatrix = std::move( seeds.getMatrix() );
      initSubentitiesCounts< EntityDimension, 0 >( seeds.getEntityCornerCounts() );
   }

   template< int Dimension, int Subdimension >
   void
   initSubentityMatrix( NeighborCountsArray& capacities, GlobalIndexType subentitiesCount = 0 )
   {
      if( Subdimension == 0 )
         subentitiesCount = mesh->template getEntitiesCount< 0 >();
      auto& matrix = mesh->template getSubentitiesMatrix< Dimension, Subdimension >();
      matrix.setDimensions( capacities.getSize(), subentitiesCount );
      matrix.setRowCapacities( capacities );
      initSubentitiesCounts< Dimension, Subdimension >( capacities );
   }

   template< int Dimension, int Subdimension >
   void
   initSubentitiesCounts( const NeighborCountsArray& capacities )
   {
      mesh->template setSubentitiesCounts< Dimension, Subdimension >( std::move( capacities ) );
   }

   template< int Dimension >
   void
   setEntitiesCount( GlobalIndexType entitiesCount )
   {
      mesh->template setEntitiesCount< Dimension >( entitiesCount );
   }

   template< int Dimension, int Subdimension >
   void
   setSubentityIndex( GlobalIndexType entityIndex, LocalIndexType localIndex, GlobalIndexType globalIndex )
   {
      mesh->template getSubentitiesMatrix< Dimension, Subdimension >()
         .getRow( entityIndex )
         .setElement( localIndex, globalIndex, true );
   }

   template< int Dimension >
   auto
   getSubvertices( GlobalIndexType entityIndex )
      -> decltype( this->mesh->template getSubentitiesMatrix< Dimension, 0 >().getRow( 0 ) )
   {
      return mesh->template getSubentitiesMatrix< Dimension, 0 >().getRow( entityIndex );
   }

   template< int Dimension >
   auto
   getSubverticesCount( GlobalIndexType entityIndex )
      -> decltype( this->mesh->template getSubentitiesCount< Dimension, 0 >( 0 ) )
   {
      return mesh->template getSubentitiesCount< Dimension, 0 >( entityIndex );
   }

   template< int Dimension, int Subdimension >
   auto
   getSubentitiesMatrix() -> decltype( this->mesh->template getSubentitiesMatrix< Dimension, Subdimension >() )
   {
      return mesh->template getSubentitiesMatrix< Dimension, Subdimension >();
   }

   template< int Dimension, int Superdimension >
   auto
   getSuperentitiesCountsArray() -> decltype( this->mesh->template getSuperentitiesCountsArray< Dimension, Superdimension >() )
   {
      return mesh->template getSuperentitiesCountsArray< Dimension, Superdimension >();
   }

   template< int Dimension, int Superdimension >
   auto
   getSuperentitiesMatrix() -> decltype( this->mesh->template getSuperentitiesMatrix< Dimension, Superdimension >() )
   {
      return mesh->template getSuperentitiesMatrix< Dimension, Superdimension >();
   }
};

/**
 * \brief Mesh initializer layer for mesh entities other than vertices, cells and faces of polyhedral meshes.
 */
template< typename MeshConfig, typename DimensionTag >
class InitializerLayer : public InitializerLayer< MeshConfig, typename DimensionTag::Decrement >
{
protected:
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = Mesh< MeshConfig >;
   using BaseType = InitializerLayer< MeshConfig, typename DimensionTag::Decrement >;
   using EntityInitializerType = EntityInitializer< MeshConfig, DimensionTag::value >;

public:
   void
   initEntities( InitializerType& initializer, MeshType& mesh )
   {
      // skip initialization of entities which do not store their subvertices
      // (and hence do not participate in any other incidence matrix)
      if constexpr( MeshConfig::subentityStorage( DimensionTag::value, 0 ) ) {
         EntityInitializerType::createEntities( initializer, mesh );
      }

      // continue with the next dimension
      BaseType::initEntities( initializer, mesh );
   }
};

/**
 * \brief Mesh initializer layer for vertices
 */
template< typename MeshConfig >
class InitializerLayer< MeshConfig, DimensionTag< 0 > >
{
   using InitializerType = Initializer< MeshConfig >;
   using MeshType = Mesh< MeshConfig >;
   using DimensionTag = Meshes::DimensionTag< 0 >;
   using EntityInitializerType = EntityInitializer< MeshConfig, DimensionTag::value >;

   using EntityTraitsType = typename MeshType::MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using EntityTopology = typename EntityTraitsType::EntityTopology;
   using SeedType = EntitySeed< MeshConfig, EntityTopology >;

public:
   void
   initEntities( InitializerType& initializer, MeshType& mesh )
   {
      // std::cout << " Initiating entities with dimension " << DimensionTag::value << " ... " << std::endl;
      auto getEntitySeedIndex = []( const SeedType& seed )
      {
         return seed.getCornerIds()[ 0 ];
      };
      EntityInitializerType::initSuperentities( initializer, mesh, getEntitySeedIndex );
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
