// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshEntity.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshBuilder.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Triangle.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polygon.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polyhedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/EntityDecomposer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/scan.h>

namespace noa::TNL {
namespace Meshes {

// Polygon Mesh
template< typename ParentConfig >
struct TriangleConfig : public ParentConfig
{
   using CellTopology = Topologies::Triangle;
};

template< EntityDecomposerVersion DecomposerVersion,
          EntityDecomposerVersion SubdecomposerVersion = EntityDecomposerVersion::ConnectEdgesToPoint,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto  // returns MeshBuilder
decomposeMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using namespace noa::TNL;
   using namespace noa::TNL::Containers;
   using namespace noa::TNL::Algorithms;

   using TriangleMeshConfig = TriangleConfig< MeshConfig >;
   using TriangleMesh = Mesh< TriangleMeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< TriangleMesh >;
   using GlobalIndexType = typename TriangleMesh::GlobalIndexType;
   using PointType = typename TriangleMesh::PointType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polygon, DecomposerVersion >;
   constexpr int CellDimension = TriangleMesh::getMeshDimension();

   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< CellDimension >();

   // Find the number of output points and cells as well as
   // starting indices at which every cell will start writing new decomposed points and cells
   using IndexPair = std::pair< GlobalIndexType, GlobalIndexType >;
   Array< IndexPair, Devices::Host > indices( inCellsCount + 1 );
   auto setCounts = [ & ]( GlobalIndexType i )
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      indices[ i ] = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
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

   // Decompose each cell
   auto decomposeCell = [ & ]( GlobalIndexType i ) mutable
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
      auto addCell = [ & ]( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 )
      {
         auto entitySeed = meshBuilder.getCellSeed( setCellIndex++ );
         entitySeed.setCornerId( 0, v0 );
         entitySeed.setCornerId( 1, v1 );
         entitySeed.setCornerId( 2, v2 );
      };

      EntityDecomposer::decompose( cell, addPoint, addCell );
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, decomposeCell );

   return meshBuilder;
}

template< EntityDecomposerVersion DecomposerVersion,
          EntityDecomposerVersion SubdecomposerVersion = EntityDecomposerVersion::ConnectEdgesToPoint,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true >
auto  // returns Mesh
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using TriangleMeshConfig = TriangleConfig< MeshConfig >;
   using TriangleMesh = Mesh< TriangleMeshConfig, Devices::Host >;

   TriangleMesh outMesh;
   auto meshBuilder = decomposeMesh< DecomposerVersion >( inMesh );
   meshBuilder.build( outMesh );
   return outMesh;
}

// Polyhedral Mesh
template< typename ParentConfig >
struct TetrahedronConfig : public ParentConfig
{
   using CellTopology = Topologies::Tetrahedron;
};

template< EntityDecomposerVersion DecomposerVersion,
          EntityDecomposerVersion SubdecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto  // returns MeshBuilder
decomposeMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using namespace noa::TNL;
   using namespace noa::TNL::Containers;
   using namespace noa::TNL::Algorithms;

   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< TetrahedronMesh >;
   using GlobalIndexType = typename TetrahedronMesh::GlobalIndexType;
   using PointType = typename TetrahedronMesh::PointType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polyhedron, DecomposerVersion, SubdecomposerVersion >;
   constexpr int CellDimension = TetrahedronMesh::getMeshDimension();

   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< CellDimension >();

   // Find the number of output points and cells as well as
   // starting indices at which every cell will start writing new decomposed points and cells
   using IndexPair = std::pair< GlobalIndexType, GlobalIndexType >;
   Array< IndexPair, Devices::Host > indices( inCellsCount + 1 );
   auto setCounts = [ & ]( GlobalIndexType i )
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      indices[ i ] = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
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

   // Decompose each cell
   auto decomposeCell = [ & ]( GlobalIndexType i ) mutable
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
      auto addCell = [ & ]( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2, GlobalIndexType v3 )
      {
         auto entitySeed = meshBuilder.getCellSeed( setCellIndex++ );
         entitySeed.setCornerId( 0, v0 );
         entitySeed.setCornerId( 1, v1 );
         entitySeed.setCornerId( 2, v2 );
         entitySeed.setCornerId( 3, v3 );
      };

      EntityDecomposer::decompose( cell, addPoint, addCell );
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, decomposeCell );

   return meshBuilder;
}

template< EntityDecomposerVersion DecomposerVersion,
          EntityDecomposerVersion SubDecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto  // returns Mesh
getDecomposedMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using TetrahedronMeshConfig = TetrahedronConfig< MeshConfig >;
   using TetrahedronMesh = Mesh< TetrahedronMeshConfig, Devices::Host >;

   TetrahedronMesh outMesh;
   auto meshBuilder = decomposeMesh< DecomposerVersion, SubDecomposerVersion >( inMesh );
   meshBuilder.build( outMesh );
   return outMesh;
}

}  // namespace Meshes
}  // namespace noa::TNL
