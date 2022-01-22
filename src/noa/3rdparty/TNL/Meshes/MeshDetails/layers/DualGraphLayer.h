// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/MeshTraits.h>

namespace noaTNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          bool enabled = MeshConfig::dualGraphStorage() >
class DualGraphLayer
{
public:
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using DualGraph = typename MeshTraitsType::DualGraph;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;

   DualGraphLayer() = default;

   explicit DualGraphLayer( const DualGraphLayer& ) = default;

   DualGraphLayer( DualGraphLayer&& ) = default;

   template< typename Device_ >
   DualGraphLayer( const DualGraphLayer< MeshConfig, Device_ >& other )
   {
      operator=( other );
   }

   DualGraphLayer& operator=( const DualGraphLayer& ) = default;

   DualGraphLayer& operator=( DualGraphLayer&& ) = default;

   template< typename Device_ >
   DualGraphLayer& operator=( const DualGraphLayer< MeshConfig, Device_ >& other )
   {
      neighborCounts = other.getNeighborCounts();
      graph = other.getDualGraph();
      return *this;
   }

   bool operator==( const DualGraphLayer& other ) const
   {
      return neighborCounts == other.getNeighborCounts() &&
             graph == other.getDualGraph();
   }

   __cuda_callable__
   const NeighborCountsArray& getNeighborCounts() const
   {
      return neighborCounts;
   }

   __cuda_callable__
   NeighborCountsArray& getNeighborCounts()
   {
      return neighborCounts;
   }

   __cuda_callable__
   const DualGraph& getDualGraph() const
   {
      return graph;
   }

   __cuda_callable__
   DualGraph& getDualGraph()
   {
      return graph;
   }

   // algorithm inspired by the CreateGraphDual function in METIS
   template< typename Mesh >
   void initializeDualGraph( const Mesh& mesh,
                             // when this parameter is <= 0, it will be replaced with MeshConfig::dualGraphMinCommonVertices
                             LocalIndexType minCommon = 0 )
   {
      static_assert( std::is_same< MeshConfig, typename Mesh::Config >::value,
                     "mismatched MeshConfig type" );
      static_assert( MeshConfig::superentityStorage( 0, Mesh::getMeshDimension() ),
                     "The dual graph cannot be initialized when links from vertices to cells are not stored in the mesh." );
      static_assert( MeshConfig::dualGraphMinCommonVertices >= 1,
                     "MeshConfig error: dualGraphMinCommonVertices must be at least 1." );
      if( minCommon <= 0 )
         minCommon = MeshConfig::dualGraphMinCommonVertices;

      const GlobalIndexType cellsCount = mesh.template getEntitiesCount< Mesh::getMeshDimension() >();

      // allocate row lengths vector
      neighborCounts.setSize( cellsCount );

      // allocate working arrays
      using GlobalIndexArray = Containers::Array< GlobalIndexType, Devices::Sequential, GlobalIndexType >;
      using LocalIndexArray = Containers::Array< LocalIndexType, Devices::Sequential, GlobalIndexType >;
      GlobalIndexArray neighbors( cellsCount );
      LocalIndexArray marker( cellsCount );
      marker.setValue( 0 );

      auto findNeighbors = [&] ( const GlobalIndexType k )
      {
         const LocalIndexType subvertices = mesh.template getSubentitiesCount< Mesh::getMeshDimension(), 0 >( k );

         // find all elements that share at least one vertex with k
         LocalIndexType counter = 0;
         for( LocalIndexType v = 0; v < subvertices; v++ ) {
            const GlobalIndexType gv = mesh.template getSubentityIndex< Mesh::getMeshDimension(), 0 >( k, v );
            const LocalIndexType supercells = mesh.template getSuperentitiesCount< 0, Mesh::getMeshDimension() >( gv );
            for( LocalIndexType sc = 0; sc < supercells; sc++ ) {
               const GlobalIndexType nk = mesh.template getSuperentityIndex< 0, Mesh::getMeshDimension() >( gv, sc );
               if( marker[ nk ] == 0 )
                  neighbors[ counter++ ] = nk;
               marker[ nk ]++;
            }
         }

         // mark k to be removed from the neighbor list in the next step
         marker[ k ] = 0;

         // compact the list to contain only those with at least minCommon vertices
         LocalIndexType compacted = 0;
         for( LocalIndexType i = 0; i < counter; i++ ) {
            const GlobalIndexType nk = neighbors[ i ];
            const LocalIndexType neighborSubvertices = mesh.template getSubentitiesCount< Mesh::getMeshDimension(), 0 >( nk );
            const LocalIndexType overlap = marker[ nk ];
            if( overlap >= minCommon || overlap >= subvertices - 1 || overlap >= neighborSubvertices - 1 )
              neighbors[ compacted++ ] = nk;
            marker[ nk ] = 0;
         }

         return compacted;
      };

      // count neighbors of each cell
      for( GlobalIndexType k = 0; k < cellsCount; k++ )
         neighborCounts[ k ] = findNeighbors( k );

      // allocate adjacency matrix
      graph.setDimensions( cellsCount, cellsCount );
      graph.setRowCapacities( neighborCounts );

      // fill in neighbor indices
      for( GlobalIndexType k = 0; k < cellsCount; k++) {
         auto row = graph.getRow( k );
         const LocalIndexType nnbrs = findNeighbors( k );
         for( LocalIndexType j = 0; j < nnbrs; j++)
            row.setElement( j, neighbors[ j ], true );
      }
   }

protected:
   NeighborCountsArray neighborCounts;
   DualGraph graph;
};

template< typename MeshConfig,
          typename Device >
class DualGraphLayer< MeshConfig, Device, false >
{
public:
   template< typename Device_ >
   DualGraphLayer& operator=( const DualGraphLayer< MeshConfig, Device_ >& other )
   {
      return *this;
   }

   template< typename Mesh >
   void initializeDualGraph( const Mesh& mesh,
                             int minCommon = 0 )
   {}
};

} // namespace Meshes
} // namespace noaTNL
