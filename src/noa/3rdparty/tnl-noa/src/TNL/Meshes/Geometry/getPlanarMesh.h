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
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/isPlanar.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/EntityDecomposer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/scan.h>

namespace noa::TNL {
namespace Meshes {

// 3D Polygon Mesh
template< EntityDecomposerVersion DecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value, bool > = true,
          std::enable_if_t< MeshConfig::spaceDimension == 3, bool > = true >
auto  // returns MeshBuilder
planarCorrection( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using namespace noa::TNL;
   using namespace noa::TNL::Containers;
   using namespace noa::TNL::Algorithms;

   using PolygonMesh = Mesh< MeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< PolygonMesh >;
   using NeighborCountsArray = typename MeshBuilder::NeighborCountsArray;
   using GlobalIndexType = typename PolygonMesh::GlobalIndexType;
   using LocalIndexType = typename PolygonMesh::LocalIndexType;
   using PointType = typename PolygonMesh::PointType;
   using RealType = typename PolygonMesh::RealType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polygon, DecomposerVersion >;
   constexpr int CellDimension = PolygonMesh::getMeshDimension();

   constexpr RealType precision{ 1e-6 };

   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< CellDimension >();

   // Find the number of output points and cells as well as
   // starting indices at which every cell will start writing new points and cells
   using IndexPair = std::pair< GlobalIndexType, GlobalIndexType >;
   Array< IndexPair, Devices::Host > indices( inCellsCount + 1 );
   auto setCounts = [ & ]( GlobalIndexType i )
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      if( isPlanar( inMesh, cell, precision ) ) {
         indices[ i ] = { 0, 1 };  // cell is not decomposed (0 extra points, 1 cell)
      }
      else {
         indices[ i ] = EntityDecomposer::getExtraPointsAndEntitiesCount( cell );
      }
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

   // set corner counts for cells
   NeighborCountsArray cellCornersCounts( outCellsCount );
   auto setCornersCount = [ & ]( GlobalIndexType i ) mutable
   {
      GlobalIndexType cellIndex = indices[ i ].second;
      const GlobalIndexType nextCellIndex = indices[ i + 1 ].second;
      const GlobalIndexType cellsCount = nextCellIndex - cellIndex;

      if( cellsCount == 1 ) {  // cell is already planar (cell is copied)
         const auto cell = inMesh.template getEntity< CellDimension >( i );
         const auto verticesCount = cell.template getSubentitiesCount< 0 >();
         cellCornersCounts[ cellIndex ] = verticesCount;
      }
      else {  // cell is not planar (cell is decomposed)
         for( ; cellIndex < nextCellIndex; cellIndex++ ) {
            cellCornersCounts[ cellIndex ] = 3;
         }
      }
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, setCornersCount );
   meshBuilder.setCellCornersCounts( std::move( cellCornersCounts ) );

   // Decompose non-planar cells and copy the rest
   auto decomposeCell = [ & ]( GlobalIndexType i ) mutable
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      const auto& indexPair = indices[ i ];
      const auto& nextIndexPair = indices[ i + 1 ];
      const GlobalIndexType cellsCount = nextIndexPair.second - indexPair.second;

      if( cellsCount == 1 ) {  // cell is already planar (cell is copied)
         auto seed = meshBuilder.getCellSeed( indexPair.second );
         const auto verticesCount = cell.template getSubentitiesCount< 0 >();
         for( LocalIndexType j = 0; j < verticesCount; j++ ) {
            seed.setCornerId( j, cell.template getSubentityIndex< 0 >( j ) );
         }
      }
      else {  // cell is not planar (cell is decomposed)
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
      }
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, decomposeCell );

   return meshBuilder;
}

// Polyhedral Mesh
template< EntityDecomposerVersion DecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value, bool > = true >
auto  // returns MeshBuilder
planarCorrection( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using namespace noa::TNL;
   using namespace noa::TNL::Containers;
   using namespace noa::TNL::Algorithms;

   using PolyhedronMesh = Mesh< MeshConfig, Devices::Host >;
   using MeshBuilder = MeshBuilder< PolyhedronMesh >;
   using NeighborCountsArray = typename MeshBuilder::NeighborCountsArray;
   using GlobalIndexType = typename PolyhedronMesh::GlobalIndexType;
   using LocalIndexType = typename PolyhedronMesh::LocalIndexType;
   using PointType = typename PolyhedronMesh::PointType;
   using RealType = typename PolyhedronMesh::RealType;
   using EntityDecomposer = EntityDecomposer< MeshConfig, Topologies::Polygon, DecomposerVersion >;
   constexpr int CellDimension = PolyhedronMesh::getMeshDimension();
   constexpr int FaceDimension = CellDimension - 1;

   constexpr RealType precision{ 1e-6 };

   MeshBuilder meshBuilder;

   const GlobalIndexType inPointsCount = inMesh.template getEntitiesCount< 0 >();
   const GlobalIndexType inFacesCount = inMesh.template getEntitiesCount< FaceDimension >();
   const GlobalIndexType inCellsCount = inMesh.template getEntitiesCount< CellDimension >();

   // Find the number of output points and faces as well as
   // starting indices at which every face will start writing new points and faces
   using IndexPair = std::pair< GlobalIndexType, GlobalIndexType >;
   Array< IndexPair, Devices::Host > indices( inFacesCount + 1 );
   auto setCounts = [ & ]( GlobalIndexType i )
   {
      const auto face = inMesh.template getEntity< FaceDimension >( i );
      if( isPlanar( inMesh, face, precision ) ) {
         indices[ i ] = { 0, 1 };  // face is not decomposed (0 extra points, 1 face)
      }
      else {
         indices[ i ] = EntityDecomposer::getExtraPointsAndEntitiesCount( face );
      }
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inFacesCount, setCounts );
   indices[ inFacesCount ] = { 0,
                               0 };  // extend exclusive prefix sum by one element to also get result of reduce at the same time
   auto reduction = []( const IndexPair& a, const IndexPair& b ) -> IndexPair
   {
      return { a.first + b.first, a.second + b.second };
   };
   inplaceExclusiveScan( indices, 0, indices.getSize(), reduction, std::make_pair( 0, 0 ) );
   const auto& reduceResult = indices[ inFacesCount ];
   const GlobalIndexType outPointsCount = inPointsCount + reduceResult.first;
   const GlobalIndexType outFacesCount = reduceResult.second;
   const GlobalIndexType outCellsCount = inCellsCount;  // The number of cells stays the same
   meshBuilder.setEntitiesCount( outPointsCount, outCellsCount, outFacesCount );

   // Copy the points from inMesh to outMesh
   auto copyPoint = [ & ]( GlobalIndexType i ) mutable
   {
      meshBuilder.setPoint( i, inMesh.getPoint( i ) );
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inPointsCount, copyPoint );

   // set corner counts for cells
   NeighborCountsArray cellCornersCounts( outCellsCount );
   auto setCellCornersCount = [ & ]( GlobalIndexType i ) mutable
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      const LocalIndexType cellFacesCount = cell.template getSubentitiesCount< FaceDimension >();

      // Count the number of corner ids for the cell
      LocalIndexType cornersCount = 0;
      for( LocalIndexType j = 0; j < cellFacesCount; j++ ) {
         const GlobalIndexType faceIdx = cell.template getSubentityIndex< FaceDimension >( j );
         cornersCount += indices[ faceIdx + 1 ].second - indices[ faceIdx ].second;
      }

      cellCornersCounts[ i ] = cornersCount;
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, setCellCornersCount );
   meshBuilder.setCellCornersCounts( std::move( cellCornersCounts ) );

   // Set corner ids for cells
   auto setCellCornersIds = [ & ]( GlobalIndexType i ) mutable
   {
      const auto cell = inMesh.template getEntity< CellDimension >( i );
      const LocalIndexType cellFacesCount = cell.template getSubentitiesCount< FaceDimension >();
      auto cellSeed = meshBuilder.getCellSeed( i );
      for( LocalIndexType j = 0, o = 0; j < cellFacesCount; j++ ) {
         const GlobalIndexType faceIdx = cell.template getSubentityIndex< FaceDimension >( j );
         const GlobalIndexType endFaceIdx = indices[ faceIdx + 1 ].second;
         for( GlobalIndexType k = indices[ faceIdx ].second; k < endFaceIdx; k++ ) {
            cellSeed.setCornerId( o++, k );
         }
      }
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inCellsCount, setCellCornersIds );

   // set corner counts for faces
   NeighborCountsArray faceCornersCounts( outFacesCount );
   auto setFaceCornersCount = [ & ]( GlobalIndexType i ) mutable
   {
      GlobalIndexType faceIndex = indices[ i ].second;
      const GlobalIndexType nextFaceIndex = indices[ i + 1 ].second;
      const GlobalIndexType facesCount = nextFaceIndex - faceIndex;
      if( facesCount == 1 ) {  // face is already planar (it is copied)
         const auto face = inMesh.template getEntity< FaceDimension >( i );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         faceCornersCounts[ faceIndex ] = verticesCount;
      }
      else {  // face is not planar (it is decomposed)
         for( ; faceIndex < nextFaceIndex; faceIndex++ ) {
            faceCornersCounts[ faceIndex ] = 3;
         }
      }
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inFacesCount, setFaceCornersCount );
   meshBuilder.setFaceCornersCounts( std::move( faceCornersCounts ) );

   // Decompose non-planar faces and copy the rest
   auto decomposeFace = [ & ]( GlobalIndexType i ) mutable
   {
      const auto face = inMesh.template getEntity< FaceDimension >( i );
      const auto& indexPair = indices[ i ];
      const auto& nextIndexPair = indices[ i + 1 ];
      const GlobalIndexType facesCount = nextIndexPair.second - indexPair.second;

      if( facesCount == 1 ) {  // face is already planar (it is copied)
         auto seed = meshBuilder.getFaceSeed( indexPair.second );
         const auto verticesCount = face.template getSubentitiesCount< 0 >();
         for( LocalIndexType j = 0; j < verticesCount; j++ ) {
            seed.setCornerId( j, face.template getSubentityIndex< 0 >( j ) );
         }
      }
      else {  // face is not planar (it is decomposed)
         // Lambda for adding new points
         GlobalIndexType setPointIndex = inPointsCount + indexPair.first;
         auto addPoint = [ & ]( const PointType& point )
         {
            const auto pointIdx = setPointIndex++;
            meshBuilder.setPoint( pointIdx, point );
            return pointIdx;
         };

         // Lambda for adding new faces
         GlobalIndexType setFaceIndex = indexPair.second;
         auto addFace = [ & ]( GlobalIndexType v0, GlobalIndexType v1, GlobalIndexType v2 )
         {
            auto entitySeed = meshBuilder.getFaceSeed( setFaceIndex++ );
            entitySeed.setCornerId( 0, v0 );
            entitySeed.setCornerId( 1, v1 );
            entitySeed.setCornerId( 2, v2 );
         };

         EntityDecomposer::decompose( face, addPoint, addFace );
      }
   };
   ParallelFor< Devices::Host >::exec( GlobalIndexType{ 0 }, inFacesCount, decomposeFace );

   return meshBuilder;
}

template< EntityDecomposerVersion DecomposerVersion,
          typename MeshConfig,
          std::enable_if_t< MeshConfig::spaceDimension == 3
                               && ( std::is_same< typename MeshConfig::CellTopology, Topologies::Polygon >::value
                                    || std::is_same< typename MeshConfig::CellTopology, Topologies::Polyhedron >::value ),
                            bool > = true >
auto  // returns Mesh
getPlanarMesh( const Mesh< MeshConfig, Devices::Host >& inMesh )
{
   using Mesh = Mesh< MeshConfig, Devices::Host >;

   Mesh outMesh;
   auto meshBuilder = planarCorrection< DecomposerVersion >( inMesh );
   meshBuilder.build( outMesh );
   return outMesh;
}

}  // namespace Meshes
}  // namespace noa::TNL
