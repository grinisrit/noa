// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixPermutationApplier.h>

namespace noa::TNL {
namespace Meshes {

template< typename Mesh, int Dimension >
struct IndexPermutationApplier
{
private:
   using GlobalIndexArray = typename Mesh::GlobalIndexArray;

   template< int Subdimension >
   static void
   permuteSubentitiesStorage( Mesh& mesh, const GlobalIndexArray& perm )
   {
      if constexpr( Mesh::Config::subentityStorage( Dimension, Subdimension ) ) {
         using EntityTopology = typename Mesh::template EntityType< Dimension >::EntityTopology;
         if constexpr( Topologies::IsDynamicTopology< EntityTopology >::value ) {
            // copy the subentity counts into an array
            typename Mesh::MeshTraitsType::NeighborCountsArray counts( perm.getSize() );
            for( typename GlobalIndexArray::ValueType i = 0; i < perm.getSize(); i++ )
               counts[ i ] = mesh.template getSubentitiesCount< Dimension, Subdimension >( i );
            // permute the array
            permuteArray( counts, perm );
            // set the permuted counts
            mesh.template setSubentitiesCounts< Dimension, Subdimension >( std::move( counts ) );
         }

         auto& subentitiesStorage = mesh.template getSubentitiesMatrix< Dimension, Subdimension >();
         Matrices::permuteMatrixRows( subentitiesStorage, perm );
      }
   }

   template< int Superdimension >
   static void
   permuteSuperentitiesStorage( Mesh& mesh, const GlobalIndexArray& perm )
   {
      if constexpr( Mesh::Config::superentityStorage( Dimension, Superdimension ) ) {
         permuteArray( mesh.template getSuperentitiesCountsArray< Dimension, Superdimension >(), perm );
         auto& superentitiesStorage = mesh.template getSuperentitiesMatrix< Dimension, Superdimension >();
         Matrices::permuteMatrixRows( superentitiesStorage, perm );
      }
   }

   template< int Subdimension >
   static void
   permuteSuperentitiesOfSubentities( Mesh& mesh, const GlobalIndexArray& iperm )
   {
      if constexpr( Mesh::Config::superentityStorage( Subdimension, Dimension ) ) {
         auto& superentitiesStorage = mesh.template getSuperentitiesMatrix< Subdimension, Dimension >();
         Matrices::permuteMatrixColumns( superentitiesStorage, iperm );
      }
   }

   template< int Superdimension >
   static void
   permuteSubentitiesOfSuperentities( Mesh& mesh, const GlobalIndexArray& iperm )
   {
      if constexpr( Mesh::Config::subentityStorage( Superdimension, Dimension ) ) {
         auto& subentitiesStorage = mesh.template getSubentitiesMatrix< Superdimension, Dimension >();
         Matrices::permuteMatrixColumns( subentitiesStorage, iperm );
      }
   }

   static void
   permuteDualGraph( Mesh& mesh, const GlobalIndexArray& perm, const GlobalIndexArray& iperm )
   {
      if constexpr( Mesh::Config::dualGraphStorage() ) {
         permuteArray( mesh.getNeighborCounts(), perm );
         auto& graph = mesh.getDualGraph();
         Matrices::permuteMatrixRows( graph, perm );
         Matrices::permuteMatrixColumns( graph, iperm );
      }
   }

public:
   template< typename ArrayOrView >
   static void
   permuteArray( ArrayOrView& array, const GlobalIndexArray& perm )
   {
      using ValueType = typename ArrayOrView::ValueType;
      using IndexType = typename ArrayOrView::IndexType;
      using DeviceType = typename ArrayOrView::DeviceType;

      Containers::Array< ValueType, DeviceType, IndexType > buffer( array.getSize() );

      // kernel to copy values to new array, applying the permutation
      auto kernel1 = [] __cuda_callable__( IndexType i, const ValueType* array, ValueType* buffer, const IndexType* perm )
      {
         buffer[ i ] = array[ perm[ i ] ];
      };

      // kernel to copy permuted values back to the mesh
      auto kernel2 = [] __cuda_callable__( IndexType i, ValueType * array, const ValueType* buffer )
      {
         array[ i ] = buffer[ i ];
      };

      Algorithms::ParallelFor< DeviceType >::exec(
         (IndexType) 0, array.getSize(), kernel1, array.getData(), buffer.getData(), perm.getData() );
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, array.getSize(), kernel2, array.getData(), buffer.getData() );
   }

   static void
   exec( Mesh& mesh, const GlobalIndexArray& perm, const GlobalIndexArray& iperm )
   {
      if( Dimension == 0 )
         permuteArray( mesh.getPoints(), perm );

      // permute subentities storage
      Algorithms::staticFor< int, 0, Dimension >(
         [ & ]( auto dim )
         {
            permuteSubentitiesStorage< dim >( mesh, perm );
         } );

      // permute superentities storage
      Algorithms::staticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1 >(
         [ & ]( auto dim )
         {
            permuteSuperentitiesStorage< dim >( mesh, perm );
         } );

      // update superentity indices from the subentities
      Algorithms::staticFor< int, 0, Dimension >(
         [ & ]( auto dim )
         {
            permuteSuperentitiesOfSubentities< dim >( mesh, iperm );
         } );

      // update subentity indices from the superentities
      Algorithms::staticFor< int, Dimension + 1, Mesh::getMeshDimension() + 1 >(
         [ & ]( auto dim )
         {
            permuteSubentitiesOfSuperentities< dim >( mesh, iperm );
         } );

      if constexpr( Dimension == Mesh::getMeshDimension() ) {
         // permute dual graph
         permuteDualGraph( mesh, perm, iperm );
      }
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
