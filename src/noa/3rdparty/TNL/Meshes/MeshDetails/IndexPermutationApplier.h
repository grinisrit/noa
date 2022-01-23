// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/TNL/Matrices/MatrixPermutationApplier.h>

namespace noa::TNL {
namespace Meshes {

template< typename Mesh, int Dimension >
struct IndexPermutationApplier
{
private:
   using GlobalIndexArray = typename Mesh::GlobalIndexArray;

   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                Subdimension >::storageEnabled
             >
   struct SubentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& perm )
      {
         auto& subentitiesStorage = mesh.template getSubentitiesMatrix< Dimension, Subdimension >();
         Matrices::permuteMatrixRows( subentitiesStorage, perm );
      }
   };

   template< int Subdimension >
   struct SubentitiesStorageWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Dimension >::EntityTopology,
                                                                  Superdimension >::storageEnabled
             >
   struct SuperentitiesStorageWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& perm )
      {
         permuteArray( mesh.template getSuperentitiesCountsArray< Dimension, Superdimension >(), perm );
         auto& superentitiesStorage = mesh.template getSuperentitiesMatrix< Dimension, Superdimension >();
         Matrices::permuteMatrixRows( superentitiesStorage, perm );
      }
   };

   template< int Superdimension >
   struct SuperentitiesStorageWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
   };


   template< int Subdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SuperentityTraits< typename Mesh::template EntityType< Subdimension >::EntityTopology,
                                                                  Dimension >::storageEnabled
             >
   struct SubentitiesWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm )
      {
         auto& superentitiesStorage = mesh.template getSuperentitiesMatrix< Subdimension, Dimension >();
         Matrices::permuteMatrixColumns( superentitiesStorage, iperm );
      }
   };

   template< int Subdimension >
   struct SubentitiesWorker< Subdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
   };


   template< int Superdimension,
             bool Enabled =
                Mesh::MeshTraitsType::template SubentityTraits< typename Mesh::template EntityType< Superdimension >::EntityTopology,
                                                                Dimension >::storageEnabled
             >
   struct SuperentitiesWorker
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm )
      {
         auto& subentitiesStorage = mesh.template getSubentitiesMatrix< Superdimension, Dimension >();
         Matrices::permuteMatrixColumns( subentitiesStorage, iperm );
      }
   };

   template< int Superdimension >
   struct SuperentitiesWorker< Superdimension, false >
   {
      static void exec( Mesh& mesh, const GlobalIndexArray& iperm ) {}
   };


   template< typename Mesh_, std::enable_if_t< Mesh_::Config::dualGraphStorage(), bool > = true >
   static void permuteDualGraph( Mesh_& mesh, const GlobalIndexArray& perm, const GlobalIndexArray& iperm )
   {
      permuteArray( mesh.getNeighborCounts(), perm );
      auto& graph = mesh.getDualGraph();
      Matrices::permuteMatrixRows( graph, perm );
      Matrices::permuteMatrixColumns( graph, iperm );
   }

   template< typename Mesh_, std::enable_if_t< ! Mesh_::Config::dualGraphStorage(), bool > = true >
   static void permuteDualGraph( Mesh_& mesh, const GlobalIndexArray& perm, const GlobalIndexArray& iperm ) {}

public:
   template< typename ArrayOrView >
   static void permuteArray( ArrayOrView& array, const GlobalIndexArray& perm )
   {
      using ValueType = typename ArrayOrView::ValueType;
      using IndexType = typename ArrayOrView::IndexType;
      using DeviceType = typename ArrayOrView::DeviceType;

      Containers::Array< ValueType, DeviceType, IndexType > buffer( array.getSize() );

      // kernel to copy values to new array, applying the permutation
      auto kernel1 = [] __cuda_callable__
         ( IndexType i,
           const ValueType* array,
           ValueType* buffer,
           const IndexType* perm )
      {
         buffer[ i ] = array[ perm[ i ] ];
      };

      // kernel to copy permuted values back to the mesh
      auto kernel2 = [] __cuda_callable__
         ( IndexType i,
           ValueType* array,
           const ValueType* buffer )
      {
         array[ i ] = buffer[ i ];
      };

      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, array.getSize(),
                                                   kernel1,
                                                   array.getData(),
                                                   buffer.getData(),
                                                   perm.getData() );
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, array.getSize(),
                                                   kernel2,
                                                   array.getData(),
                                                   buffer.getData() );
   }

   static void exec( Mesh& mesh,
                     const GlobalIndexArray& perm,
                     const GlobalIndexArray& iperm )
   {
      using IndexType = typename Mesh::GlobalIndexType;
      using DeviceType = typename Mesh::DeviceType;

      if( Dimension == 0 )
         permuteArray( mesh.getPoints(), perm );

      // permute subentities storage
      Algorithms::staticFor< int, 0, Dimension >(
         [&] ( auto dim ) {
            SubentitiesStorageWorker< dim >::exec( mesh, perm );
         }
      );

      // permute superentities storage
      Algorithms::staticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1 >(
         [&] ( auto dim ) {
            SuperentitiesStorageWorker< dim >::exec( mesh, perm );
         }
      );

      // update superentity indices from the subentities
      Algorithms::staticFor< int, 0, Dimension >(
         [&] ( auto dim ) {
            SubentitiesWorker< dim >::exec( mesh, iperm );
         }
      );

      // update subentity indices from the superentities
      Algorithms::staticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1 >(
         [&] ( auto dim ) {
            SuperentitiesWorker< dim >::exec( mesh, iperm );
         }
      );

      if( Dimension == Mesh::getMeshDimension() ) {
         // permute dual graph
         permuteDualGraph( mesh, perm, iperm );
      }
   }
};

} // namespace Meshes
} // namespace noa::TNL
