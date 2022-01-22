// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/layers/EntityTags/Traits.h>
#include <noa/3rdparty/TNL/Algorithms/scan.h>
#include <noa/3rdparty/TNL/Algorithms/contains.h>

namespace noaTNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename GlobalIndexType >
auto
exchangeGhostEntitySeeds( MPI_Comm communicator,
                          const std::vector< std::vector< GlobalIndexType > >& seeds_vertex_indices,
                          const std::vector< std::vector< GlobalIndexType > >& seeds_entity_offsets )
{
   const int rank = MPI::GetRank( communicator );
   const int nproc = MPI::GetSize( communicator );

   // exchange sizes of the arrays
   Containers::Array< GlobalIndexType, Devices::Host, int > sizes_vertex_indices( nproc ), sizes_entity_offsets( nproc );
   {
      Containers::Array< GlobalIndexType, Devices::Host, int > sendbuf_indices( nproc ), sendbuf_offsets( nproc );
      for( int i = 0; i < nproc; i++ ) {
         sendbuf_indices[ i ] = seeds_vertex_indices[ i ].size();
         sendbuf_offsets[ i ] = seeds_entity_offsets[ i ].size();
      }
      MPI::Alltoall( sendbuf_indices.getData(), 1,
                     sizes_vertex_indices.getData(), 1,
                     communicator );
      MPI::Alltoall( sendbuf_offsets.getData(), 1,
                     sizes_entity_offsets.getData(), 1,
                     communicator );
   }

   // allocate arrays for the results
   std::vector< std::vector< GlobalIndexType > > foreign_seeds_vertex_indices, foreign_seeds_entity_offsets;
   foreign_seeds_vertex_indices.resize( nproc );
   foreign_seeds_entity_offsets.resize( nproc );
   for( int i = 0; i < nproc; i++ ) {
      foreign_seeds_vertex_indices[ i ].resize( sizes_vertex_indices[ i ] );
      foreign_seeds_entity_offsets[ i ].resize( sizes_entity_offsets[ i ] );
   }

   // buffer for asynchronous communication requests
   std::vector< MPI_Request > requests;

   // issue all async receive operations
   for( int j = 0; j < nproc; j++ ) {
      if( j == rank )
          continue;
      requests.push_back( MPI::Irecv(
               foreign_seeds_vertex_indices[ j ].data(),
               foreign_seeds_vertex_indices[ j ].size(),
               j, 0, communicator ) );
      requests.push_back( MPI::Irecv(
               foreign_seeds_entity_offsets[ j ].data(),
               foreign_seeds_entity_offsets[ j ].size(),
               j, 1, communicator ) );
   }

   // issue all async send operations
   for( int i = 0; i < nproc; i++ ) {
      if( i == rank )
          continue;
      requests.push_back( MPI::Isend(
               seeds_vertex_indices[ i ].data(),
               seeds_vertex_indices[ i ].size(),
               i, 0, communicator ) );
      requests.push_back( MPI::Isend(
               seeds_entity_offsets[ i ].data(),
               seeds_entity_offsets[ i ].size(),
               i, 1, communicator ) );
   }

   // wait for all communications to finish
   MPI::Waitall( requests.data(), requests.size() );

   return std::make_tuple( foreign_seeds_vertex_indices, foreign_seeds_entity_offsets );
}

template< typename GlobalIndexType >
auto
exchangeGhostIndices( MPI_Comm communicator,
                      const std::vector< std::vector< GlobalIndexType > >& foreign_ghost_indices,
                      const std::vector< std::vector< GlobalIndexType > >& seeds_local_indices )
{
   const int rank = MPI::GetRank( communicator );
   const int nproc = MPI::GetSize( communicator );

   // allocate arrays for the results
   std::vector< std::vector< GlobalIndexType > > ghost_indices;
   ghost_indices.resize( nproc );
   for( int i = 0; i < nproc; i++ )
      ghost_indices[ i ].resize( seeds_local_indices[ i ].size() );

   // buffer for asynchronous communication requests
   std::vector< MPI_Request > requests;

   // issue all async receive operations
   for( int j = 0; j < nproc; j++ ) {
      if( j == rank )
          continue;
      requests.push_back( MPI::Irecv(
               ghost_indices[ j ].data(),
               ghost_indices[ j ].size(),
               j, 0, communicator ) );
   }

   // issue all async send operations
   for( int i = 0; i < nproc; i++ ) {
      if( i == rank )
          continue;
      requests.push_back( MPI::Isend(
               foreign_ghost_indices[ i ].data(),
               foreign_ghost_indices[ i ].size(),
               i, 0, communicator ) );
   }

   // wait for all communications to finish
   MPI::Waitall( requests.data(), requests.size() );

   return ghost_indices;
}

// \param preferHighRanks  When true, faces on the interface between two subdomains will be owned
//                         by the rank with the higher number. Otherwise, they will be owned by the
//                         rank with the lower number.
template< int Dimension, typename DistributedMesh >
void
distributeSubentities( DistributedMesh& mesh, bool preferHighRanks = true )
{
   using DeviceType = typename DistributedMesh::DeviceType;
   using GlobalIndexType = typename DistributedMesh::GlobalIndexType;
   using LocalIndexType = typename DistributedMesh::LocalIndexType;
   using LocalMesh = typename DistributedMesh::MeshType;

   static_assert( ! std::is_same< DeviceType, Devices::Cuda >::value,
                  "this method can be called only for host meshes" );
   static_assert( 0 < Dimension && Dimension < DistributedMesh::getMeshDimension(),
                  "Vertices and cells cannot be distributed using this method." );
   if( mesh.getGhostLevels() <= 0 )
      throw std::logic_error( "There are no ghost levels on the distributed mesh." );

   const int rank = MPI::GetRank( mesh.getCommunicator() );
   const int nproc = MPI::GetSize( mesh.getCommunicator() );

   // 0. exchange cell data to prepare getCellOwner for use in getEntityOwner
   DistributedMeshSynchronizer< DistributedMesh, DistributedMesh::getMeshDimension() > cell_synchronizer;
   cell_synchronizer.initialize( mesh );

   auto getCellOwner = [&] ( GlobalIndexType local_idx ) -> int
   {
      const GlobalIndexType global_idx = mesh.template getGlobalIndices< DistributedMesh::getMeshDimension() >()[ local_idx ];
      return cell_synchronizer.getEntityOwner( global_idx );
   };

   // algorithm for getEntityOwner:
   // 0. if all neighbor cells are local, we are the owner
   //    -> this is still necessary, e.g. for situations like this:
   //             __/\__
   //       if the upper rank owns the vertices on the interface, it would get the face below the /\ cape
   // (1.) all vertices internal -> local entity
   //      (this should be covered by 0.)
   // (2.) some internal vertices, other vertices on the interface -> local entity
   //      (this should be covered by 0.)
   // 3. all vertices on the interface (i.e. some neighbor cells are local, some are ghosts)
   //    -> faces: there are exactly 2 ranks which share the face on their interface, pick the one with min/max number
   //    -> other subentities: the face owner should own all subentities of the face
   // 4. any number of ghost non-interface vertices (i.e. the entity has only ghost neighbor cells)
   //    -> the true owner cannot be determined, so we return just the owner of the first neighbor cell
   //       and let the iterative algorithm with padding indices to take care of the rest
   //       (the neighbor will return a padding index until it gets the global index from the true owner)
   // TODO: implement getEntityOwner for other subentities
   static_assert( Dimension == DistributedMesh::getMeshDimension() - 1, "the getEntityOwner function is implemented only for faces" );
   auto getEntityOwner = [&] ( GlobalIndexType local_idx ) -> int
   {
      auto entity = mesh.getLocalMesh().template getEntity< Dimension >( local_idx );

      int local_neighbor_cells_count = 0;
      for( LocalIndexType k = 0; k < entity.template getSuperentitiesCount< DistributedMesh::getMeshDimension() >(); k++ ) {
         const GlobalIndexType gk = entity.template getSuperentityIndex< DistributedMesh::getMeshDimension() >( k );
         if( ! mesh.getLocalMesh().template isGhostEntity< DistributedMesh::getMeshDimension() >( gk ) )
            local_neighbor_cells_count++;
      }

      // if all neighbor cells are local, we are the owner
      if( local_neighbor_cells_count == 2 )
         return rank;

      // face is on the interface
      if( local_neighbor_cells_count == 1 ) {
         int owner = (preferHighRanks) ? 0 : nproc;
         for( LocalIndexType k = 0; k < entity.template getSuperentitiesCount< DistributedMesh::getMeshDimension() >(); k++ ) {
            const GlobalIndexType gk = entity.template getSuperentityIndex< DistributedMesh::getMeshDimension() >( k );
            if( preferHighRanks )
               owner = noaTNL::max( owner, getCellOwner( gk ) );
            else
               owner = noaTNL::min( owner, getCellOwner( gk ) );
         }
         return owner;
      }

      // only ghost cell neighbors - owner cannot be determined, return the first cell neighbor
      const GlobalIndexType gk = entity.template getSuperentityIndex< DistributedMesh::getMeshDimension() >( 0 );
      return getCellOwner( gk );
   };

   // 1. identify local entities, set the GhostEntity tag on others
   LocalMesh& localMesh = mesh.getLocalMesh();
   localMesh.template forAll< Dimension >( [&] ( GlobalIndexType i ) mutable {
      if( getEntityOwner( i ) != rank )
         localMesh.template addEntityTag< Dimension >( i, EntityTags::GhostEntity );
   });

   // 2. exchange the counts of local entities between ranks, compute offsets for global indices
   Containers::Vector< GlobalIndexType, Devices::Host, int > globalOffsets( nproc );
   {
      GlobalIndexType localEntitiesCount = 0;
      for( GlobalIndexType i = 0; i < localMesh.template getEntitiesCount< Dimension >(); i++ )
         if( ! localMesh.template isGhostEntity< Dimension >( i ) )
            ++localEntitiesCount;

      Containers::Array< GlobalIndexType, Devices::Host, int > sendbuf( nproc );
      sendbuf.setValue( localEntitiesCount );
      MPI::Alltoall( sendbuf.getData(), 1,
                     globalOffsets.getData(), 1,
                     mesh.getCommunicator() );
   }
   Algorithms::inplaceExclusiveScan( globalOffsets );

   // 3. assign global indices to the local entities and a padding index to ghost entities
   //    (later we can check the padding index to know if an index was set or not)
   const GlobalIndexType padding_index = (std::is_signed<GlobalIndexType>::value) ? -1 : std::numeric_limits<GlobalIndexType>::max();
   mesh.template getGlobalIndices< Dimension >().setSize( localMesh.template getEntitiesCount< Dimension >() );
   // also create mapping for ghost entities so that we can efficiently iterate over ghost entities
   // while the entities were not sorted yet on the local mesh
   std::vector< GlobalIndexType > localGhostIndices;
   GlobalIndexType entityIndex = globalOffsets[ rank ];
   for( GlobalIndexType i = 0; i < localMesh.template getEntitiesCount< Dimension >(); i++ ) {
      if( localMesh.template isGhostEntity< Dimension >( i ) ) {
         mesh.template getGlobalIndices< Dimension >()[ i ] = padding_index;
         localGhostIndices.push_back( i );
      }
      else
         mesh.template getGlobalIndices< Dimension >()[ i ] = entityIndex++;
   }

   // Now for each ghost entity, we will take the global indices of its subvertices and
   // send them to another rank (does not necessarily have to be the owner of the entity).
   // The other rank will scan its vertex-entity superentity matrix, find the entity which
   // has the received vertex indices and send the global entity index (which can be a
   // padding index if it does not know it yet) back to the inquirer. The communication
   // process is repeated until there are no padding indices.
   // Note that we have to synchronize based on the vertex-entity superentity matrix,
   // because synchronization based on the cell-entity subentity matrix would not be
   // general. For example, two subdomains can have a common face, but no common cell,
   // even when ghost_levels > 0. On the other hand, if two subdomains have a common face,
   // they have common all its subvertices.

   // 4. build seeds for ghost entities
   std::vector< std::vector< GlobalIndexType > > seeds_vertex_indices, seeds_entity_offsets, seeds_local_indices;
   seeds_vertex_indices.resize( nproc );
   seeds_entity_offsets.resize( nproc );
   seeds_local_indices.resize( nproc );
   for( auto entity_index : localGhostIndices ) {
      const int owner = getEntityOwner( entity_index );
      for( LocalIndexType v = 0; v < localMesh.template getSubentitiesCount< Dimension, 0 >( entity_index ); v++ ) {
         const GlobalIndexType local_index = localMesh.template getSubentityIndex< Dimension, 0 >( entity_index, v );
         const GlobalIndexType global_index = mesh.template getGlobalIndices< 0 >()[ local_index ];
         seeds_vertex_indices[ owner ].push_back( global_index );
      }
      seeds_entity_offsets[ owner ].push_back( seeds_vertex_indices[ owner ].size() );
      // record the corresponding local index for later use
      seeds_local_indices[ owner ].push_back( entity_index );
   }

   // 5. exchange seeds for ghost entities
   const auto foreign_seeds = exchangeGhostEntitySeeds( mesh.getCommunicator(), seeds_vertex_indices, seeds_entity_offsets );
   const auto& foreign_seeds_vertex_indices = std::get< 0 >( foreign_seeds );
   const auto& foreign_seeds_entity_offsets = std::get< 1 >( foreign_seeds );

   std::vector< std::vector< GlobalIndexType > > foreign_ghost_indices;
   foreign_ghost_indices.resize( nproc );
   for( int i = 0; i < nproc; i++ )
      foreign_ghost_indices[ i ].resize( foreign_seeds_entity_offsets[ i ].size() );

   // 6. iterate until there are no padding indices
   //    (maybe 2 iterations are always enough, but we should be extra careful)
   for( int iter = 0; iter < DistributedMesh::getMeshDimension(); iter++ )
   {
      // 6a. determine global indices for the received seeds
      Algorithms::ParallelFor< Devices::Host >::exec( 0, nproc, [&] ( int i ) {
         GlobalIndexType vertexOffset = 0;
         // loop over all foreign ghost entities
         for( std::size_t entityIndex = 0; entityIndex < foreign_seeds_entity_offsets[ i ].size(); entityIndex++ ) {
            // data structure for common indices
            std::set< GlobalIndexType > common_indices;

            // global vertex index offset (for global-to-local index conversion)
            const GlobalIndexType globalOffset = mesh.template getGlobalIndices< 0 >().getElement( 0 );

            // loop over all subvertices of the entity
            while( vertexOffset < foreign_seeds_entity_offsets[ i ][ entityIndex ] ) {
               const GlobalIndexType vertex = foreign_seeds_vertex_indices[ i ][ vertexOffset++ ];
               GlobalIndexType localIndex = 0;
               if( vertex >= globalOffset && vertex < globalOffset + localMesh.template getGhostEntitiesOffset< 0 >() )
                  // subtract offset to get local index
                  localIndex = vertex - globalOffset;
               else {
                  // we must go through the ghost vertices
                  for( GlobalIndexType g = localMesh.template getGhostEntitiesOffset< 0 >();
                       g < localMesh.template getEntitiesCount< 0 >();
                       g++ )
                     if( vertex == mesh.template getGlobalIndices< 0 >()[ g ] ) {
                        localIndex = g;
                        break;
                     }
                  if( localIndex == 0 )
                     throw std::runtime_error( "vertex with gid=" + std::to_string(vertex) + " received from rank "
                                             + std::to_string(i) + " was not found on the local mesh for rank " + std::to_string(rank)
                                             + " (global offset = " + std::to_string(globalOffset) + ")" );
               }

               // collect superentities of this vertex
               std::set< GlobalIndexType > superentities;
               for( LocalIndexType e = 0; e < localMesh.template getSuperentitiesCount< 0, Dimension >( localIndex ); e++ ) {
                  const GlobalIndexType entity = localMesh.template getSuperentityIndex< 0, Dimension >( localIndex, e );
                  superentities.insert( entity );
               }

               // initialize or intersect
               if( common_indices.empty() )
                  common_indices = superentities;
               else
                  // remove indices which are not in the current superentities set
                  for( auto it = common_indices.begin(); it != common_indices.end(); ) {
                     if( superentities.count( *it ) == 0 )
                        it = common_indices.erase(it);
                     else
                        ++it;
                  }
            }

            if( common_indices.size() != 1 ) {
               std::stringstream msg;
               msg << "expected exactly 1 common index, but the algorithm found these common indices: ";
               for( auto i : common_indices )
                  msg << i << " ";
               msg << "\nDebug info: rank " << rank << ", entityIndex = " << entityIndex << ", received from rank " << i;
               throw std::runtime_error( msg.str() );
            }

            // assign global index
            // (note that we do not have to check if we are the actual owner of the entity,
            // we can just forward global indices that we received from somebody else, or
            // send the padding index if the global index was not received yet)
            const GlobalIndexType local_index = *common_indices.begin();
            foreign_ghost_indices[ i ][ entityIndex ] = mesh.template getGlobalIndices< Dimension >()[ local_index ];
         }
      });

      // 6b. exchange global ghost indices
      const auto ghost_indices = exchangeGhostIndices( mesh.getCommunicator(), foreign_ghost_indices, seeds_local_indices );

      // 6c. set the global indices of our ghost entities
      bool done = true;
      for( int i = 0; i < nproc; i++ ) {
         for( std::size_t g = 0; g < ghost_indices[ i ].size(); g++ ) {
            mesh.template getGlobalIndices< Dimension >()[ seeds_local_indices[ i ][ g ] ] = ghost_indices[ i ][ g ];
            if( ghost_indices[ i ][ g ] == padding_index )
               done = false;
         }
      }

      // 6d. check if finished
      bool all_done = false;
      MPI::Allreduce( &done, &all_done, 1, MPI_LAND, mesh.getCommunicator() );
      if( all_done )
         break;
   }
   if( Algorithms::contains( mesh.template getGlobalIndices< Dimension >(), padding_index ) )
      throw std::runtime_error( "some global indices were left unset" );

   // 7. reorder the entities to make sure that global indices are sorted
   {
      // prepare arrays for the permutation and inverse permutation
      typename LocalMesh::GlobalIndexArray perm, iperm;
      perm.setSize( localMesh.template getEntitiesCount< Dimension >() );
      iperm.setSize( localMesh.template getEntitiesCount< Dimension >() );

      // create the permutation - sort the identity array by the global index, but put local entities first
      localMesh.template forAll< Dimension >( [&] ( GlobalIndexType i ) mutable { perm[ i ] = i; });
      std::stable_sort( perm.getData(), // + mesh.getLocalMesh().template getGhostEntitiesOffset< Dimension >(),
                        perm.getData() + perm.getSize(),
                        [&mesh, &localMesh](auto& left, auto& right) {
         auto is_local = [&](auto& i) {
            return ! localMesh.template isGhostEntity< Dimension >( i );
         };
         if( is_local(left) && ! is_local(right) )
            return true;
         if( ! is_local(left) && is_local(right) )
            return false;
         return mesh.template getGlobalIndices< Dimension >()[ left ] < mesh.template getGlobalIndices< Dimension >()[ right ];
      });

      // invert the permutation
      localMesh.template forAll< Dimension >( [&] ( GlobalIndexType i ) mutable {
         iperm[ perm[ i ] ] = i;
      });

      // reorder the mesh
      mesh.template reorderEntities< Dimension >( perm, iperm );
   }
}

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace noaTNL
