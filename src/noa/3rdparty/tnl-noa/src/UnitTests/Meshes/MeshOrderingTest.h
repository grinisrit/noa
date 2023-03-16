#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <array>

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/MeshBuilder.h>

namespace MeshOrderingTest {

using namespace TNL;
using namespace TNL::Meshes;

class TestTriangleMeshConfig
   : public DefaultConfig< Topologies::Triangle, 2, double, int, short int >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

template< typename Device >
bool buildTriangleMesh( Mesh< TestTriangleMeshConfig, Device >& mesh )
{
   using TriangleMesh = Mesh< TestTriangleMeshConfig, Device >;
   using TriangleMeshEntityType = typename TriangleMesh::template EntityType< 2 >;
   using EdgeMeshEntityType = typename TriangleMesh::template EntityType< 1 >;
   using VertexMeshEntityType = typename TriangleMesh::template EntityType< 0 >;

   static_assert( TriangleMeshEntityType::template SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::template SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::template SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::template SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::template SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::template SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, double > >::value, "" );

   /****
    * We set-up the following situation
            point2   edge3       point3
               |\-------------------|
               | \                  |
               |  \   triangle1     |
               |   \                |

                      ....
            edge1     edge0        edge4
                      ....


               |   triangle0     \  |
               |                  \ |
               ---------------------|
            point0   edge2        point1
    */

   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 1.0 ),
             point3( 1.0, 1.0 );

   MeshBuilder< TriangleMesh > meshBuilder;

   meshBuilder.setEntitiesCount( 4, 2 );

   meshBuilder.setPoint( 0, point0 );
   meshBuilder.setPoint( 1, point1 );
   meshBuilder.setPoint( 2, point2 );
   meshBuilder.setPoint( 3, point3 );

   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 1 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 2 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 3 );
   return meshBuilder.build( mesh );
}

template< typename PermutationArray >
void testMesh( const Mesh< TestTriangleMeshConfig, Devices::Host >& mesh,
               const PermutationArray& vertexPermutation,
               const PermutationArray& edgePermutation,
               const PermutationArray& cellPermutation )
{
   using MeshType = Mesh< TestTriangleMeshConfig, Devices::Host >;
   using PointType = typename MeshType::PointType;

   ASSERT_EQ( vertexPermutation.getSize(), 4 );
   ASSERT_EQ( edgePermutation.getSize(),   5 );
   ASSERT_EQ( cellPermutation.getSize(),   2 );

   EXPECT_EQ( mesh.getEntitiesCount< 0 >(),  4 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(),  5 );
   EXPECT_EQ( mesh.getEntitiesCount< 2 >(),  2 );

   // test points
   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 1.0 ),
             point3( 1.0, 1.0 );

   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).getPoint(),  point0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).getPoint(),  point1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).getPoint(),  point2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).getPoint(),  point3 );


   // test getIndex
   for( int i = 0; i < 4; i++ )
      EXPECT_EQ( mesh.template getEntity< 0 >( i ).getIndex(), i );
   for( int i = 0; i < 5; i++ )
      EXPECT_EQ( mesh.template getEntity< 1 >( i ).getIndex(), i );
   for( int i = 0; i < 2; i++ )
      EXPECT_EQ( mesh.template getEntity< 2 >( i ).getIndex(), i );


   // test subentities
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 1 ] );

   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 0 >( 2 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 1 >( 0 ),  edgePermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 1 >( 1 ),  edgePermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 0 ] ).template getSubentityIndex< 1 >( 2 ),  edgePermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 0 >( 0 ),  vertexPermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 0 >( 1 ),  vertexPermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 0 >( 2 ),  vertexPermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 1 >( 0 ),  edgePermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 1 >( 1 ),  edgePermutation[ 4 ] );
   EXPECT_EQ( mesh.template getEntity< 2 >( cellPermutation[ 1 ] ).template getSubentityIndex< 1 >( 2 ),  edgePermutation[ 0 ] );


   // test superentities
   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 2 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 1 >( 2 ),  edgePermutation[ 4 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 1 >( 2 ),  edgePermutation[ 3 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentityIndex< 1 >( 0 ),  edgePermutation[ 3 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentityIndex< 1 >( 1 ),  edgePermutation[ 4 ] );


   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 0 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 1 ] ).template getSuperentityIndex< 2 >( 1 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 2 ] ).template getSuperentityIndex< 2 >( 1 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( vertexPermutation[ 3 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 1 ] );


   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 0 ] ).template getSuperentityIndex< 2 >( 1 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 1 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 2 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 0 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 3 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 1 ] );

   ASSERT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( edgePermutation[ 4 ] ).template getSuperentityIndex< 2 >( 0 ),  cellPermutation[ 1 ] );


   // test boundary tags
   const std::vector< int > boundaryFaces = {1, 2, 3, 4};
   const std::vector< int > interiorFaces = {0};
   EXPECT_EQ( mesh.template getBoundaryIndices< 1 >().getSize(), (int) boundaryFaces.size() );
   for( size_t i = 0; i < boundaryFaces.size(); i++ ) {
      EXPECT_TRUE( mesh.template isBoundaryEntity< 1 >( edgePermutation[ boundaryFaces[ i ] ] ) );
      // boundary indices are always sorted so we can't test this
//      EXPECT_EQ( mesh.template getBoundaryIndices< 1 >()[ i ], edgePermutation[ boundaryFaces[ i ] ] );
   }
   // Test interior faces
   EXPECT_EQ( mesh.template getInteriorIndices< 1 >().getSize(), (int) interiorFaces.size() );
   for( size_t i = 0; i < interiorFaces.size(); i++ ) {
      EXPECT_FALSE( mesh.template isBoundaryEntity< 1 >( edgePermutation[ interiorFaces[ i ] ] ) );
      // boundary indices are always sorted so we can't test this
//      EXPECT_EQ( mesh.template getInteriorIndices< 1 >()[ i ], edgePermutation[ interiorFaces[ i ] ] );
   }

   // tests for the dual graph layer
   ASSERT_EQ( mesh.getCellNeighborsCount( cellPermutation[ 0 ] ), 1 );
   ASSERT_EQ( mesh.getCellNeighborsCount( cellPermutation[ 1 ] ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex( cellPermutation[ 0 ], 0 ), cellPermutation[ 1 ] );
   EXPECT_EQ( mesh.getCellNeighborIndex( cellPermutation[ 1 ], 0 ), cellPermutation[ 0 ] );
}

// hack due to TNL::Containers::Vector not supporting initilizer lists
const std::array< int, 4 > _vertexIdentity { { 0, 1, 2, 3 } };
const std::array< int, 5 > _edgeIdentity   { { 0, 1, 2, 3, 4 } };
const std::array< int, 2 > _cellIdentity   { { 0, 1 } };

const std::array< int, 4 > _vertexPermutation { { 3, 2, 0, 1 } };
const std::array< int, 5 > _edgePermutation   { { 2, 0, 4, 1, 3 } };
const std::array< int, 2 > _cellPermutation   { { 1, 0 } };

const std::array< int, 4 > _vertexInversePermutation { { 2, 3, 1, 0 } };
const std::array< int, 5 > _edgeInversePermutation   { { 1, 3, 0, 4, 2 } };
const std::array< int, 2 > _cellInversePermutation   { { 1, 0 } };

template< typename TNLVector, typename STDArray >
void setPermutation( TNLVector& perm, const STDArray& stdperm )
{
   perm.setSize( stdperm.size() );
   for( int i = 0; i < perm.getSize(); i++ )
      perm.setElement( i, stdperm[ i ] );
}

TEST( MeshOrderingTest, OrderingOnHost )
{
   using MeshHost = Mesh< TestTriangleMeshConfig, Devices::Host >;

   MeshHost mesh;
   ASSERT_TRUE( buildTriangleMesh( mesh ) );

   using PermutationArray = typename MeshHost::GlobalIndexArray;
   PermutationArray vertexIdentity, edgeIdentity, cellIdentity,
                    vertexPermutation, edgePermutation, cellPermutation,
                    vertexInversePermutation, edgeInversePermutation, cellInversePermutation;
   setPermutation( vertexIdentity, _vertexIdentity );
   setPermutation( edgeIdentity, _edgeIdentity );
   setPermutation( cellIdentity, _cellIdentity );
   setPermutation( vertexPermutation, _vertexPermutation );
   setPermutation( edgePermutation, _edgePermutation );
   setPermutation( cellPermutation, _cellPermutation );
   setPermutation( vertexInversePermutation, _vertexInversePermutation );
   setPermutation( edgeInversePermutation, _edgeInversePermutation );
   setPermutation( cellInversePermutation, _cellInversePermutation );

   mesh.template reorderEntities< 0 >( vertexPermutation, vertexInversePermutation );
   testMesh( mesh, vertexInversePermutation, edgeIdentity, cellIdentity );

   mesh.template reorderEntities< 2 >( cellPermutation, cellInversePermutation );
   testMesh( mesh, vertexInversePermutation, edgeIdentity, cellInversePermutation );

   mesh.template reorderEntities< 1 >( edgePermutation, edgeInversePermutation );
   testMesh( mesh, vertexInversePermutation, edgeInversePermutation, cellInversePermutation );
}

#ifdef __CUDACC__
TEST( MeshOrderingTest, OrderingOnCuda )
{
   using MeshHost = Mesh< TestTriangleMeshConfig, Devices::Host >;
   using MeshCuda = Mesh< TestTriangleMeshConfig, Devices::Cuda >;

   MeshHost meshHost;
   MeshCuda mesh;
   ASSERT_TRUE( buildTriangleMesh( meshHost ) );
   mesh = meshHost;

   using PermutationCuda = typename MeshCuda::GlobalIndexArray;
   PermutationCuda vertexIdentity, edgeIdentity, cellIdentity,
                   vertexPermutation, edgePermutation, cellPermutation,
                   vertexInversePermutation, edgeInversePermutation, cellInversePermutation;
   setPermutation( vertexIdentity, _vertexIdentity );
   setPermutation( edgeIdentity, _edgeIdentity );
   setPermutation( cellIdentity, _cellIdentity );
   setPermutation( vertexPermutation, _vertexPermutation );
   setPermutation( edgePermutation, _edgePermutation );
   setPermutation( cellPermutation, _cellPermutation );
   setPermutation( vertexInversePermutation, _vertexInversePermutation );
   setPermutation( edgeInversePermutation, _edgeInversePermutation );
   setPermutation( cellInversePermutation, _cellInversePermutation );

   mesh.template reorderEntities< 0 >( vertexPermutation, vertexInversePermutation );
   mesh.template reorderEntities< 1 >( edgePermutation, edgeInversePermutation );
   mesh.template reorderEntities< 2 >( cellPermutation, cellInversePermutation );

   // test is on host
   {
      // local scope so we can use the same names
      using PermutationArray = typename MeshHost::GlobalIndexArray;
      PermutationArray vertexIdentity, edgeIdentity, cellIdentity,
                       vertexPermutation, edgePermutation, cellPermutation,
                       vertexInversePermutation, edgeInversePermutation, cellInversePermutation;
      setPermutation( vertexIdentity, _vertexIdentity );
      setPermutation( edgeIdentity, _edgeIdentity );
      setPermutation( cellIdentity, _cellIdentity );
      setPermutation( vertexPermutation, _vertexPermutation );
      setPermutation( edgePermutation, _edgePermutation );
      setPermutation( cellPermutation, _cellPermutation );
      setPermutation( vertexInversePermutation, _vertexInversePermutation );
      setPermutation( edgeInversePermutation, _edgeInversePermutation );
      setPermutation( cellInversePermutation, _cellInversePermutation );

      meshHost = mesh;
      testMesh( meshHost, vertexInversePermutation, edgeInversePermutation, cellInversePermutation );
   }
};
#endif

} // namespace MeshOrderingTest

#endif
