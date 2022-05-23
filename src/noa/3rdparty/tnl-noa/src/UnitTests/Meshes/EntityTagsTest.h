#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <vector>

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/MeshBuilder.h>

namespace EntityTagsTest {

using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

class TestQuadrangleMeshConfig : public DefaultConfig< Topologies::Quadrangle >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
   static constexpr bool boundaryTagsStorage( int entityDimension ) { return true; }
};

TEST( MeshTest, RegularMeshOfQuadranglesTest )
{
   using QuadrangleMeshEntityType = MeshEntity< TestQuadrangleMeshConfig, Devices::Host, Topologies::Quadrangle >;
   using EdgeMeshEntityType = typename QuadrangleMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename QuadrangleMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );

   const IndexType xSize( 3 ), ySize( 4 );
   const RealType width( 1.0 ), height( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize );
   const IndexType numberOfCells = xSize * ySize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );

   typedef Mesh< TestQuadrangleMeshConfig > TestQuadrangleMesh;
   TestQuadrangleMesh mesh, mesh2;
   MeshBuilder< TestQuadrangleMesh > meshBuilder;
   meshBuilder.setEntitiesCount( numberOfVertices, numberOfCells );

   /****
    * Setup vertices
    */
   for( IndexType j = 0; j <= ySize; j++ )
   for( IndexType i = 0; i <= xSize; i++ )
      meshBuilder.setPoint( j * ( xSize + 1 ) + i, PointType( i * hx, j * hy ) );

   /****
    * Setup cells
    */
   IndexType cellIdx( 0 );
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      const IndexType vertex0 = j * ( xSize + 1 ) + i;
      const IndexType vertex1 = j * ( xSize + 1 ) + i + 1;
      const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i + 1;
      const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i;

      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex0 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex1 );
      meshBuilder.getCellSeed( cellIdx   ).setCornerId( 2, vertex2 );
      meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 3, vertex3 );
   }

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   std::vector< IndexType > boundaryCells = {0, 1, 2, 3, 5, 6, 8, 9, 10, 11};
   std::vector< IndexType > interiorCells = {4, 7};

   // Test boundary cells
   EXPECT_EQ( mesh.template getBoundaryIndices< 2 >().getSize(), (int) boundaryCells.size() );
   for( size_t i = 0; i < boundaryCells.size(); i++ ) {
      EXPECT_TRUE( mesh.template isBoundaryEntity< 2 >( boundaryCells[ i ] ) );
      EXPECT_EQ( mesh.template getBoundaryIndices< 2 >()[ i ], boundaryCells[ i ] );
   }
   // Test interior cells
   EXPECT_EQ( mesh.template getInteriorIndices< 2 >().getSize(), (int) interiorCells.size() );
   for( size_t i = 0; i < interiorCells.size(); i++ ) {
      EXPECT_FALSE( mesh.template isBoundaryEntity< 2 >( interiorCells[ i ] ) );
      EXPECT_EQ( mesh.template getInteriorIndices< 2 >()[ i ], interiorCells[ i ] );
   }

   // Test setting other tags
   for( size_t i = 0; i < boundaryCells.size(); i++ ) {
      mesh.template addEntityTag< 2 >( boundaryCells[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 2 >( boundaryCells[ i ] ), Meshes::EntityTags::BoundaryEntity | Meshes::EntityTags::GhostEntity );
      EXPECT_TRUE( mesh.template isBoundaryEntity< 2 >( boundaryCells[ i ] ) );
      mesh.template removeEntityTag< 2 >( boundaryCells[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 2 >( boundaryCells[ i ] ), Meshes::EntityTags::BoundaryEntity );
   }
   for( size_t i = 0; i < interiorCells.size(); i++ ) {
      mesh.template addEntityTag< 2 >( interiorCells[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 2 >( interiorCells[ i ] ), Meshes::EntityTags::GhostEntity );
      EXPECT_FALSE( mesh.template isBoundaryEntity< 2 >( interiorCells[ i ] ) );
      mesh.template removeEntityTag< 2 >( interiorCells[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 2 >( interiorCells[ i ] ), 0 );
   }

   std::vector< IndexType > boundaryFaces = {0, 3, 4, 7, 8, 12, 15, 19, 22, 25, 26, 28, 29, 30};
   std::vector< IndexType > interiorFaces = {1, 2, 5, 6, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21, 23, 24, 27};

   // Test boundary faces
   EXPECT_EQ( mesh.template getBoundaryIndices< 1 >().getSize(), (int) boundaryFaces.size() );
   for( size_t i = 0; i < boundaryFaces.size(); i++ ) {
      EXPECT_TRUE( mesh.template isBoundaryEntity< 1 >( boundaryFaces[ i ] ) );
      EXPECT_EQ( mesh.template getBoundaryIndices< 1 >()[ i ], boundaryFaces[ i ] );
   }
   // Test interior faces
   EXPECT_EQ( mesh.template getInteriorIndices< 1 >().getSize(), (int) interiorFaces.size() );
   for( size_t i = 0; i < interiorFaces.size(); i++ ) {
      EXPECT_FALSE( mesh.template isBoundaryEntity< 1 >( interiorFaces[ i ] ) );
      EXPECT_EQ( mesh.template getInteriorIndices< 1 >()[ i ], interiorFaces[ i ] );
   }

   // Test setting other tags
   for( size_t i = 0; i < boundaryFaces.size(); i++ ) {
      mesh.template addEntityTag< 1 >( boundaryFaces[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 1 >( boundaryFaces[ i ] ), Meshes::EntityTags::BoundaryEntity | Meshes::EntityTags::GhostEntity );
      EXPECT_TRUE( mesh.template isBoundaryEntity< 1 >( boundaryFaces[ i ] ) );
      mesh.template removeEntityTag< 1 >( boundaryFaces[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 1 >( boundaryFaces[ i ] ), Meshes::EntityTags::BoundaryEntity );
   }
   for( size_t i = 0; i < interiorFaces.size(); i++ ) {
      mesh.template addEntityTag< 1 >( interiorFaces[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 1 >( interiorFaces[ i ] ), Meshes::EntityTags::GhostEntity );
      EXPECT_FALSE( mesh.template isBoundaryEntity< 1 >( interiorFaces[ i ] ) );
      mesh.template removeEntityTag< 1 >( interiorFaces[ i ], Meshes::EntityTags::GhostEntity );
      EXPECT_EQ( mesh.template getEntityTag< 1 >( interiorFaces[ i ] ), 0 );
   }
}

} // namespace EntityTagsTest

#endif
