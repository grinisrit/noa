// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshEntity.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshBuilder.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/EntityRefiner.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/scan.h>

namespace noa::TNL {
namespace Meshes {

// TODO: refactor to avoid duplicate points altogether - first split edges, then faces, then cells
template< EntityRefinerVersion RefinerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Triangle >::value
                               || std::is_same< typename MeshConfig::CellTopology, Topologies::Quadrangle >::value
                               || std::is_same< typename MeshConfig::CellTopology, Topologies::Tetrahedron >::value
                               || std::is_same< typename MeshConfig::CellTopology, Topologies::Hexahedron >::value,
                            bool > = true >
auto  // returns MeshBuilder
refineMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using namespace noa::TNL;
   using namespace noa::TNL::Containers;
   using namespace noa::TNL::Algorithms;

   using Mesh = Mesh< MeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< Mesh >;
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using PointType = typename Mesh::PointType;
   using EntityRefiner = EntityRefiner< MeshConfig, typename MeshConfig::CellTopology, RefinerVersion >;
   constexpr int CellDimension = Mesh::getMeshDimension();

   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< CellDimension >();

   // Find the number of output points and cells as well as
   // starting indices at which every cell will start writing new refined points and cells
   using IndexPair = std::pair< GlobalIndexType, GlobalIndexType >;
   Array< IndexPair, Devices::Host > indices( inCellsCount + 1 );
   auto setCounts = [ & ]( GlobalIndexType i )
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      indices[ i ] = EntityRefiner::getExtraPointsAndEntitiesCount( cell );
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, setCounts );
   indices[ inCellsCount ] = { 0,
                               0 };  // extend exclusive prefix sum by one element to also get result of reduce at the same time
   auto reduction = []( const IndexPair& a, const IndexPair& b ) -> IndexPair
   {
      return { a.first + b.first, a.second + b.second };
   };
   inplaceExclusiveScan( indices, 0, indices.getSize(), reduction, std::make_pair( 0, 0 ) );
   const auto& reduceResult = indices[ inCellsCount ];
   const GlobalIndexType outPointsCount = inPointsCount + reduceResult.first;
   const GlobalIndexType outCellsCount = reduceResult.second;
   meshBuilder.setEntitiesCount( outPointsCount, outCellsCount );

   // Copy the points from inMesh to outMesh
   auto copyPoint = [ & ]( GlobalIndexType i ) mutable
   {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inPointsCount, copyPoint );

   // Refine each cell
   auto refineCell = [ & ]( GlobalIndexType i ) mutable
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      const auto& indexPair = indices[ i ];

      // Lambda for adding new points
      GlobalIndexType setPointIndex = inPointsCount + indexPair.first;
      auto addPoint = [ & ]( const PointType& point )
      {
         const auto pointIdx = setPointIndex++;
         meshBuilder.setPoint( pointIdx, point );
         return pointIdx;
      };

      // Lambda for adding new cells
      GlobalIndexType setCellIndex = indexPair.second;
      auto addCell = [ & ]( auto... vertexIndices )
      {
         auto entitySeed = meshBuilder.getCellSeed( setCellIndex++ );
         entitySeed.setCornerIds( vertexIndices... );
      };

      EntityRefiner::decompose( cell, addPoint, addCell );
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, refineCell );

   return meshBuilder;
}

template< EntityRefinerVersion RefinerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Triangle >::value
                               || std::is_same< typename MeshConfig::CellTopology, Topologies::Quadrangle >::value
                               || std::is_same< typename MeshConfig::CellTopology, Topologies::Tetrahedron >::value
                               || std::is_same< typename MeshConfig::CellTopology, Topologies::Hexahedron >::value,
                            bool > = true >
auto  // returns Mesh
getRefinedMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using Mesh = Mesh< MeshConfig, Devices::Host >;

   Mesh outMesh;
   auto meshBuilder = refineMesh< RefinerVersion >( inMesh );
   meshBuilder.deduplicatePoints();
   meshBuilder.build( outMesh );
   return outMesh;
}

}  // namespace Meshes
}  // namespace noa::TNL
