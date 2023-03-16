// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Array.h>

namespace noa::TNL {
namespace Meshes {

template< bool reverse = true >
struct CuthillMcKeeOrdering
{
   template< typename MeshEntity, typename Mesh, typename PermutationArray >
   static void
   getPermutations( const Mesh& mesh, PermutationArray& perm, PermutationArray& iperm )
   {
      static_assert( std::is_same< typename Mesh::DeviceType, TNL::Devices::Host >::value, "" );
      static_assert( std::is_same< typename PermutationArray::DeviceType, TNL::Devices::Host >::value, "" );

      // The reverse Cuthill-McKee ordering is implemented only for cells,
      // other entities are ordered from the current order of cells exactly
      // as if the mesh was re-initialized from the cell seeds.
      Wrapper< MeshEntity, Mesh >::getPermutations( mesh, perm, iperm );
   }

private:
   template< typename MeshEntity, typename Mesh, bool is_cell = MeshEntity::getEntityDimension() == Mesh::getMeshDimension() >
   struct Wrapper
   {
      template< typename PermutationArray >
      static void
      getPermutations( const Mesh& mesh, PermutationArray& perm, PermutationArray& iperm )
      {
         reorderCells( mesh, perm, iperm );
      }
   };

   template< typename MeshEntity, typename Mesh >
   struct Wrapper< MeshEntity, Mesh, false >
   {
      template< typename PermutationArray >
      static void
      getPermutations( const Mesh& mesh, PermutationArray& perm, PermutationArray& iperm )
      {
         reorderEntities< MeshEntity >( mesh, perm, iperm );
      }
   };

   template< typename Mesh, typename PermutationArray >
   static void
   reorderCells( const Mesh& mesh, PermutationArray& perm, PermutationArray& iperm )
   {
      using IndexType = typename Mesh::GlobalIndexType;
      const IndexType numberOfCells = mesh.template getEntitiesCount< typename Mesh::Cell >();

      // allocate permutation vectors
      perm.setSize( numberOfCells );
      iperm.setSize( numberOfCells );

      // vector view for the neighbor counts
      const auto neighborCounts = mesh.getNeighborCounts().getConstView();

      // worker array - marker for inserted elements
      TNL::Containers::Array< bool, TNL::Devices::Host, IndexType > marker( numberOfCells );
      marker.setValue( false );
      // worker vector for collecting neighbors
      std::vector< IndexType > neighbors;

      // comparator functor
      auto comparator = [ & ]( IndexType a, IndexType b )
      {
         return neighborCounts[ a ] < neighborCounts[ b ];
      };

      // counter for assigning indices
      IndexType permIndex = 0;

      // modifier for the reversed variant
      auto mod = [ numberOfCells ]( IndexType i )
      {
         if( reverse )
            return numberOfCells - 1 - i;
         else
            return i;
      };

      // start with a peripheral node
      const IndexType peripheral = argMin( neighborCounts ).first;
      perm[ mod( permIndex ) ] = peripheral;
      iperm[ peripheral ] = mod( permIndex );
      permIndex++;
      marker[ peripheral ] = true;

      // Cuthill--McKee
      IndexType i = 0;
      while( permIndex < numberOfCells ) {
         const IndexType k = perm[ mod( i ) ];
         // collect all neighbors which were not marked yet
         const IndexType count = neighborCounts[ k ];
         for( IndexType n = 0; n < count; n++ ) {
            const IndexType nk = mesh.getCellNeighborIndex( k, n );
            if( ! marker[ nk ] ) {
               neighbors.push_back( nk );
               marker[ nk ] = true;
            }
         }
         // sort collected neighbors with ascending neighbors count
         std::sort( neighbors.begin(), neighbors.end(), comparator );
         // assign an index to the neighbors in this order
         for( auto nk : neighbors ) {
            perm[ mod( permIndex ) ] = nk;
            iperm[ nk ] = mod( permIndex );
            permIndex++;
         }
         // next iteration
         i++;
         neighbors.clear();
      }
   }

   template< typename MeshEntity, typename Mesh, typename PermutationArray >
   static void
   reorderEntities( const Mesh& mesh, PermutationArray& perm, PermutationArray& iperm )
   {
      using IndexType = typename Mesh::GlobalIndexType;
      const IndexType numberOfEntities = mesh.template getEntitiesCount< MeshEntity >();
      const IndexType numberOfCells = mesh.template getEntitiesCount< typename Mesh::Cell >();

      // allocate permutation vectors
      perm.setSize( numberOfEntities );
      iperm.setSize( numberOfEntities );

      // worker array - marker for numbered entities
      TNL::Containers::Array< bool, TNL::Devices::Host, IndexType > marker( numberOfEntities );
      marker.setValue( false );

      IndexType permIndex = 0;
      for( IndexType K = 0; K < numberOfCells; K++ ) {
         const auto& cell = mesh.template getEntity< Mesh::getMeshDimension() >( K );
         for( typename Mesh::LocalIndexType e = 0; e < cell.template getSubentitiesCount< MeshEntity::getEntityDimension() >();
              e++ ) {
            const auto E = cell.template getSubentityIndex< MeshEntity::getEntityDimension() >( e );
            if( ! marker[ E ] ) {
               marker[ E ] = true;
               perm[ permIndex ] = E;
               iperm[ E ] = permIndex;
               permIndex++;
            }
         }
      }
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
