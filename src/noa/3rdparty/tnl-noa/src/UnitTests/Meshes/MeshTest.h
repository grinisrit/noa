#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <sstream>

#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshEntity.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Vertex.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Topologies/Wedge.h>
#include <TNL/Meshes/Topologies/Pyramid.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>
#include <TNL/Meshes/MeshBuilder.h>

#include "EntityTests.h"

namespace MeshTest {

using namespace TNL;
using namespace TNL::Meshes;

using RealType = double;
using Device = Devices::Host;
using IndexType = int;

class TestTriangleMeshConfig : public DefaultConfig< Topologies::Triangle >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestQuadrangleMeshConfig : public DefaultConfig< Topologies::Quadrangle >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestTetrahedronMeshConfig : public DefaultConfig< Topologies::Tetrahedron >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestHexahedronMeshConfig : public DefaultConfig< Topologies::Hexahedron >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestTwoPolygonsMeshConfig : public DefaultConfig< Topologies::Polygon >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestSevenPolygonsMeshConfig : public DefaultConfig< Topologies::Polygon >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestTwoWedgesMeshConfig : public DefaultConfig< Topologies::Wedge >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestTwoPyramidsMeshConfig : public DefaultConfig< Topologies::Pyramid >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

class TestTwoPolyhedronsMeshConfig : public DefaultConfig< Topologies::Polyhedron >
{
public:
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension ) { return true; }
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension ) { return true; }
};

template< typename Object1, typename Object2 >
void compareStringRepresentation( const Object1& obj1, const Object2& obj2 )
{
   std::stringstream str1, str2;
   str1 << obj1;
   str2 << obj2;
   EXPECT_EQ( str1.str(), str2.str() );
}

template< typename Object >
void testCopyAssignment( const Object& obj )
{
   static_assert( std::is_copy_constructible< Object >::value, "" );
   static_assert( std::is_copy_assignable< Object >::value, "" );

   Object new_obj_1( obj );
   EXPECT_EQ( new_obj_1, obj );
   Object new_obj_2;
   new_obj_2 = obj;
   EXPECT_EQ( new_obj_2, obj );

   compareStringRepresentation( obj, new_obj_1 );
}

template< typename Mesh >
void testMeshOnCuda( const Mesh& mesh )
{
#ifdef __CUDACC__
   using DeviceMesh = Meshes::Mesh< typename Mesh::Config, Devices::Cuda >;

   // test host->CUDA copy
   DeviceMesh dmesh1( mesh );
   EXPECT_EQ( dmesh1, mesh );
   DeviceMesh dmesh2;
   dmesh2 = mesh;
   EXPECT_EQ( dmesh2, mesh );
   compareStringRepresentation( dmesh2, mesh );

   // test CUDA->CUDA copy
   testCopyAssignment( dmesh1 );

   // copy CUDA->host copy
   Mesh mesh2( dmesh1 );
   EXPECT_EQ( mesh2, mesh );
   Mesh mesh3;
   mesh3 = dmesh1;
   EXPECT_EQ( mesh2, mesh );
#endif
}

template< typename Mesh >
void testFinishedMesh( const Mesh& mesh )
{
   testCopyAssignment( mesh );
   testMeshOnCuda( mesh );
   testEntities( mesh );
}

Mesh< TestTriangleMeshConfig >
createMeshWithTwoTriangles()
{
   using MeshType = Mesh< TestTriangleMeshConfig >;
   using PointType = typename MeshType::PointType;

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

   MeshType mesh;
   MeshBuilder< MeshType > meshBuilder;

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
   if( ! meshBuilder.build( mesh ) )
      throw std::runtime_error("mesh builder failed");

   return mesh;
}

TEST( MeshTest, TwoTrianglesTest )
{
   using TriangleMeshEntityType = MeshEntity< TestTriangleMeshConfig, Devices::Host, Topologies::Triangle >;
   using EdgeMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( TriangleMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing triangle entity does not store edges as required." );
   static_assert( TriangleMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing triangle entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );

   using MeshType = Mesh< TestTriangleMeshConfig >;
   const MeshType& mesh = createMeshWithTwoTriangles();

   EXPECT_EQ( mesh.getEntitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(),  5 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(),  4 );

   const PointType point0( 0.0, 0.0 ),
                   point1( 1.0, 0.0 ),
                   point2( 0.0, 1.0 ),
                   point3( 1.0, 1.0 );

   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).getPoint(),  point0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).getPoint(),  point1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).getPoint(),  point2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).getPoint(),  point3 );

   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).template getSubentityIndex< 0 >( 1 ),  1 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 2 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 2 ),  0 );

   // tests for the superentities layer
   ASSERT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 0 ),    1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 1 ),    2 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 1 ),    2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 2 ),    4 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 1 ),    1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 1 ),    1 );

   // tests for the dual graph layer
   ASSERT_EQ( mesh.getCellNeighborsCount( 0 ), 1 );
   ASSERT_EQ( mesh.getCellNeighborsCount( 1 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex( 0, 0 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex( 1, 0 ), 0 );

   testFinishedMesh( mesh );
}

TEST( MeshTest, TwoTrianglesTest_ReturnedPair )
{
   using TriangleTestMesh = Mesh< TestTriangleMeshConfig >;
   using PointsArray = typename TriangleTestMesh::MeshTraitsType::PointArrayType;

   auto create_mesh_with_two_triangles = [] ()
      -> std::pair< Mesh< TestTriangleMeshConfig >, typename Mesh< TestTriangleMeshConfig >::MeshTraitsType::PointArrayType >
   {
      using MeshType = Mesh< TestTriangleMeshConfig >;
      const MeshType mesh = createMeshWithTwoTriangles();
      return std::make_pair( mesh, mesh.getPoints() );
   };

   std::vector< std::pair< TriangleTestMesh, PointsArray > > pairs;
   // this invokes the Mesh copy-constructor which is the thing that we originally wanted to test here
   pairs.emplace_back( create_mesh_with_two_triangles() );

   const auto& mesh = pairs.front().first;
   const auto& points = pairs.front().second;

   EXPECT_EQ( mesh.getEntitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(),  5 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(),  4 );

   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).getPoint(),  points[ 0 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).getPoint(),  points[ 1 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).getPoint(),  points[ 2 ] );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).getPoint(),  points[ 3 ] );

   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).template getSubentityIndex< 0 >( 1 ),  1 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 2 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 2 ),  0 );

   // tests for the superentities layer
   ASSERT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 0 ),    1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 1 ),    2 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 1 ),    2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 2 ),    4 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 1 ),    1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 0 ),    0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 1 ),    1 );

   // tests for the dual graph layer
   ASSERT_EQ( mesh.getCellNeighborsCount( 0 ), 1 );
   ASSERT_EQ( mesh.getCellNeighborsCount( 1 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex( 0, 0 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex( 1, 0 ), 0 );

   testFinishedMesh( mesh );
}

TEST( MeshTest, TetrahedronsTest )
{
   using TetrahedronMeshEntityType = MeshEntity< TestTetrahedronMeshConfig, Devices::Host, Topologies::Tetrahedron >;
   using VertexMeshEntityType = typename TetrahedronMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 3, RealType > >::value,
                  "unexpected PointType" );

   typedef Mesh< TestTetrahedronMeshConfig > TestTetrahedronMesh;
   TestTetrahedronMesh mesh;
   MeshBuilder< TestTetrahedronMesh > meshBuilder;

   meshBuilder.setEntitiesCount( 13, 18 );

   meshBuilder.setPoint(  0, PointType(  0.000000, 0.000000, 0.000000 ) );
   meshBuilder.setPoint(  1, PointType(  0.000000, 0.000000, 8.000000 ) );
   meshBuilder.setPoint(  2, PointType(  0.000000, 8.000000, 0.000000 ) );
   meshBuilder.setPoint(  3, PointType( 15.000000, 0.000000, 0.000000 ) );
   meshBuilder.setPoint(  4, PointType(  0.000000, 8.000000, 8.000000 ) );
   meshBuilder.setPoint(  5, PointType( 15.000000, 0.000000, 8.000000 ) );
   meshBuilder.setPoint(  6, PointType( 15.000000, 8.000000, 0.000000 ) );
   meshBuilder.setPoint(  7, PointType( 15.000000, 8.000000, 8.000000 ) );
   meshBuilder.setPoint(  8, PointType(  7.470740, 8.000000, 8.000000 ) );
   meshBuilder.setPoint(  9, PointType(  7.470740, 0.000000, 8.000000 ) );
   meshBuilder.setPoint( 10, PointType(  7.504125, 8.000000, 0.000000 ) );
   meshBuilder.setPoint( 11, PointType(  7.212720, 0.000000, 0.000000 ) );
   meshBuilder.setPoint( 12, PointType( 11.184629, 3.987667, 3.985835 ) );

   /****
    * Setup the following tetrahedrons:
    * ( Generated by Netgen )
    *
    *  12        8        7        5
    *  12        7        8       10
    *  12       11        8        9
    *  10       11        2        8
    *  12        7        6        5
    *   9       12        5        8
    *  12       11        9        3
    *   9        4       11        8
    *  12        9        5        3
    *   1        2        0       11
    *   8       11        2        4
    *   1        2       11        4
    *   9        4        1       11
    *  10       11        8       12
    *  12        6        7       10
    *  10       11       12        3
    *  12        6        3        5
    *  12        3        6       10
    */

    //  12        8        7        5
   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 8 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 7 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 3, 5 );

    //  12        7        8       10
   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 7 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 8 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 3, 10 );

    //  12       11        8        9
   meshBuilder.getCellSeed( 2 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 2, 8 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 3, 9 );

    //  10       11        2        8
   meshBuilder.getCellSeed( 3 ).setCornerId( 0, 10 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 3, 8 );

    //  12        7        6        5
   meshBuilder.getCellSeed( 4 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 1, 7 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 2, 6 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 3, 5 );

    //   9       12        5        8
   meshBuilder.getCellSeed( 5 ).setCornerId( 0, 9 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 1, 12 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 2, 5 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 3, 8 );

    //  12       11        9        3
   meshBuilder.getCellSeed( 6 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 2, 9 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 3, 3 );

    //   9        4       11        8
   meshBuilder.getCellSeed( 7 ).setCornerId( 0, 9 );
   meshBuilder.getCellSeed( 7 ).setCornerId( 1, 4 );
   meshBuilder.getCellSeed( 7 ).setCornerId( 2, 11 );
   meshBuilder.getCellSeed( 7 ).setCornerId( 3, 8 );

    //  12        9        5        3
   meshBuilder.getCellSeed( 8 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 8 ).setCornerId( 1, 9 );
   meshBuilder.getCellSeed( 8 ).setCornerId( 2, 5 );
   meshBuilder.getCellSeed( 8 ).setCornerId( 3, 3 );

    //   1        2        0       11
   meshBuilder.getCellSeed( 9 ).setCornerId( 0, 1 );
   meshBuilder.getCellSeed( 9 ).setCornerId( 1, 2 );
   meshBuilder.getCellSeed( 9 ).setCornerId( 2, 0 );
   meshBuilder.getCellSeed( 9 ).setCornerId( 3, 11 );

    //   8       11        2        4
   meshBuilder.getCellSeed( 10 ).setCornerId( 0, 8 );
   meshBuilder.getCellSeed( 10 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 10 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 10 ).setCornerId( 3, 4 );

    //   1        2       11        4
   meshBuilder.getCellSeed( 11 ).setCornerId( 0, 1 );
   meshBuilder.getCellSeed( 11 ).setCornerId( 1, 2 );
   meshBuilder.getCellSeed( 11 ).setCornerId( 2, 11 );
   meshBuilder.getCellSeed( 11 ).setCornerId( 3, 4 );

    //   9        4        1       11
   meshBuilder.getCellSeed( 12 ).setCornerId( 0, 9 );
   meshBuilder.getCellSeed( 12 ).setCornerId( 1, 4 );
   meshBuilder.getCellSeed( 12 ).setCornerId( 2, 1 );
   meshBuilder.getCellSeed( 12 ).setCornerId( 3, 11 );

    //  10       11        8       12
   meshBuilder.getCellSeed( 13 ).setCornerId( 0, 10 );
   meshBuilder.getCellSeed( 13 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 13 ).setCornerId( 2, 8 );
   meshBuilder.getCellSeed( 13 ).setCornerId( 3, 12 );

    //  12        6        7       10
   meshBuilder.getCellSeed( 14 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 14 ).setCornerId( 1, 6 );
   meshBuilder.getCellSeed( 14 ).setCornerId( 2, 7 );
   meshBuilder.getCellSeed( 14 ).setCornerId( 3, 10 );

    //  10       11       12        3
   meshBuilder.getCellSeed( 15 ).setCornerId( 0, 10 );
   meshBuilder.getCellSeed( 15 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 15 ).setCornerId( 2, 12 );
   meshBuilder.getCellSeed( 15 ).setCornerId( 3, 3 );

    //  12        6        3        5
   meshBuilder.getCellSeed( 16 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 16 ).setCornerId( 1, 6 );
   meshBuilder.getCellSeed( 16 ).setCornerId( 2, 3 );
   meshBuilder.getCellSeed( 16 ).setCornerId( 3, 5 );

    //  12        3        6       10
   meshBuilder.getCellSeed( 17 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 17 ).setCornerId( 1, 3 );
   meshBuilder.getCellSeed( 17 ).setCornerId( 2, 6 );
   meshBuilder.getCellSeed( 17 ).setCornerId( 3, 10 );

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   testFinishedMesh( mesh );
}

TEST( MeshTest, RegularMeshOfTrianglesTest )
{
   using TriangleMeshEntityType = MeshEntity< TestTriangleMeshConfig, Devices::Host, Topologies::Triangle >;
   using VertexMeshEntityType = typename TriangleMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );

   const IndexType xSize( 5 ), ySize( 5 );
   const RealType width( 1.0 ), height( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize );
   const IndexType numberOfCells = 2 * xSize * ySize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 );

   typedef Mesh< TestTriangleMeshConfig > TestTriangleMesh;
   Mesh< TestTriangleMeshConfig > mesh;
   MeshBuilder< TestTriangleMesh > meshBuilder;
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
         const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i;
         const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i + 1;

         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex0 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex1 );
         meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 2, vertex2 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex1 );
         meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex2 );
         meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 2, vertex3 );
      }

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // Test cells -> vertices subentities
   cellIdx = 0;
   for( IndexType j = 0; j < ySize; j++ )
      for( IndexType i = 0; i < xSize; i++ )
      {
         const IndexType vertex0 = j * ( xSize + 1 ) + i;
         const IndexType vertex1 = j * ( xSize + 1 ) + i + 1;
         const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i;
         const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i + 1;

         const TriangleMeshEntityType& leftCell = mesh.template getEntity< 2 >( cellIdx++ );
         EXPECT_EQ( leftCell.template getSubentityIndex< 0 >( 0 ), vertex0 );
         EXPECT_EQ( leftCell.template getSubentityIndex< 0 >( 1 ), vertex1 );
         EXPECT_EQ( leftCell.template getSubentityIndex< 0 >( 2 ), vertex2 );

         const TriangleMeshEntityType& rightCell = mesh.template getEntity< 2 >( cellIdx++ );
         EXPECT_EQ( rightCell.template getSubentityIndex< 0 >( 0 ), vertex1 );
         EXPECT_EQ( rightCell.template getSubentityIndex< 0 >( 1 ), vertex2 );
         EXPECT_EQ( rightCell.template getSubentityIndex< 0 >( 2 ), vertex3 );
      }

   // Test vertices -> cells superentities
   for( IndexType j = 0; j <= ySize; j++ )
      for( IndexType i = 0; i <= xSize; i++ )
      {
         const IndexType vertexIndex = j * ( xSize + 1 ) + i;
         const VertexMeshEntityType& vertex = mesh.template getEntity< 0 >( vertexIndex );

         if( ( i == 0 && j == 0 ) || ( i == xSize && j == ySize ) ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 2 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 1 );
         }
         else if( ( i == 0 && j == ySize ) || ( i == xSize && j == 0 ) ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 3 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 2 );
         }
         else if( i == 0 || i == xSize || j == 0 || j == ySize ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 4 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 3 );
         }
         else {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 6 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 6 );
         }
      }

   testFinishedMesh( mesh );
}

TEST( MeshTest, RegularMeshOfQuadranglesTest )
{
   using QuadrangleMeshEntityType = MeshEntity< TestQuadrangleMeshConfig, Devices::Host, Topologies::Quadrangle >;
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
   TestQuadrangleMesh mesh;
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

   // Test cells -> vertices subentities
   cellIdx = 0;
   for( IndexType j = 0; j < ySize; j++ )
      for( IndexType i = 0; i < xSize; i++ )
      {
         const IndexType vertex0 = j * ( xSize + 1 ) + i;
         const IndexType vertex1 = j * ( xSize + 1 ) + i + 1;
         const IndexType vertex2 = ( j + 1 ) * ( xSize + 1 ) + i + 1;
         const IndexType vertex3 = ( j + 1 ) * ( xSize + 1 ) + i;

         const QuadrangleMeshEntityType& cell = mesh.template getEntity< 2 >( cellIdx++ );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 0 ), vertex0 );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 1 ), vertex1 );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 2 ), vertex2 );
         EXPECT_EQ( cell.template getSubentityIndex< 0 >( 3 ), vertex3 );
      }

   // Test vertices -> cells superentities
   for( IndexType j = 0; j <= ySize; j++ )
      for( IndexType i = 0; i <= xSize; i++ )
      {
         const IndexType vertexIndex = j * ( xSize + 1 ) + i;
         const VertexMeshEntityType& vertex = mesh.template getEntity< 0 >( vertexIndex );

         if( ( i == 0 || i == xSize ) && ( j == 0 || j == ySize ) ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 2 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 1 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 0 ),   ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
         }
         else if( i == 0 || i == xSize || j == 0 || j == ySize ) {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 3 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 2 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 0 ),   ( j - ( j == ySize || i == 0 || i == xSize ) ) * xSize + i - ( i == xSize ) - ( j == 0 || j == ySize ) );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 1 ),   ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
         }
         else {
            EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 4 );
            EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 4 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 0 ),   ( j - 1 ) * xSize + i - 1 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 1 ),   ( j - 1 ) * xSize + i     );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 2 ),   ( j     ) * xSize + i - 1 );
            EXPECT_EQ( vertex.template getSuperentityIndex< 2 >( 3 ),   ( j     ) * xSize + i     );
         }
      }

   // Tests for the dual graph layer
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), numberOfCells );
   cellIdx = 0;
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      IndexType nnbrs = 4;
      if( i == 0 || i == xSize - 1 )
         --nnbrs;
      if( j == 0 || j == ySize - 1 )
         --nnbrs;

      EXPECT_EQ( mesh.getCellNeighborsCount( cellIdx ), nnbrs );
      std::set< IndexType > neighbors;
      for( IndexType n = 0; n < nnbrs; n++ )
         neighbors.insert( mesh.getDualGraph().getRow( cellIdx ).getColumnIndex( n ) );

      // the cell itself should not be its own neighbor
      EXPECT_EQ( (IndexType) neighbors.count( cellIdx ), 0 );
      auto check_neighbor = [&]( IndexType i, IndexType j )
      {
         if( i >= 0 && i < xSize && j >= 0 && j < ySize ) {
            EXPECT_EQ( (IndexType) neighbors.count( j * xSize + i ), 1 );
         }
      };
      // check neighbors over face
      check_neighbor( i - 1, j );
      check_neighbor( i + 1, j );
      check_neighbor( i, j - 1 );
      check_neighbor( i, j + 1 );

      ++cellIdx;
   }

   testFinishedMesh( mesh );
}

TEST( MeshTest, RegularMeshOfHexahedronsTest )
{
   using HexahedronMeshEntityType = MeshEntity< TestHexahedronMeshConfig, Devices::Host, Topologies::Hexahedron >;
   using VertexMeshEntityType = typename HexahedronMeshEntityType::SubentityTraits< 0 >::SubentityType;

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 3, RealType > >::value,
                  "unexpected PointType" );

   const IndexType xSize( 3 ), ySize( 4 ), zSize( 5 );
   const RealType width( 1.0 ), height( 1.0 ), depth( 1.0 );
   const RealType hx( width / ( RealType ) xSize ),
                  hy( height / ( RealType ) ySize ),
                  hz( depth / ( RealType ) zSize );
   const IndexType numberOfCells = xSize * ySize * zSize;
   const IndexType numberOfVertices = ( xSize + 1 ) * ( ySize + 1 ) * ( zSize + 1 );

   typedef Mesh< TestHexahedronMeshConfig > TestHexahedronMesh;
   TestHexahedronMesh mesh;
   MeshBuilder< TestHexahedronMesh > meshBuilder;
   meshBuilder.setEntitiesCount( numberOfVertices, numberOfCells );

   /****
    * Setup vertices
    */
   for( IndexType k = 0; k <= zSize; k++ )
      for( IndexType j = 0; j <= ySize; j++ )
         for( IndexType i = 0; i <= xSize; i++ )
            meshBuilder.setPoint( k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i, PointType( i * hx, j * hy, k * hz ) );

   /****
    * Setup cells
    */
   IndexType cellIdx( 0 );
   for( IndexType k = 0; k < zSize; k++ )
      for( IndexType j = 0; j < ySize; j++ )
         for( IndexType i = 0; i < xSize; i++ )
         {
            const IndexType vertex0 = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
            const IndexType vertex1 = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i + 1;
            const IndexType vertex2 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;
            const IndexType vertex3 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;
            const IndexType vertex4 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
            const IndexType vertex5 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i + 1;
            const IndexType vertex6 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;
            const IndexType vertex7 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;

            meshBuilder.getCellSeed( cellIdx   ).setCornerId( 0, vertex0 );
            meshBuilder.getCellSeed( cellIdx   ).setCornerId( 1, vertex1 );
            meshBuilder.getCellSeed( cellIdx   ).setCornerId( 2, vertex2 );
            meshBuilder.getCellSeed( cellIdx   ).setCornerId( 3, vertex3 );
            meshBuilder.getCellSeed( cellIdx   ).setCornerId( 4, vertex4 );
            meshBuilder.getCellSeed( cellIdx   ).setCornerId( 5, vertex5 );
            meshBuilder.getCellSeed( cellIdx   ).setCornerId( 6, vertex6 );
            meshBuilder.getCellSeed( cellIdx++ ).setCornerId( 7, vertex7 );
         }

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // Test cells -> vertices subentities
   cellIdx = 0;
   for( IndexType k = 0; k < zSize; k++ )
      for( IndexType j = 0; j < ySize; j++ )
         for( IndexType i = 0; i < xSize; i++ )
         {
            const IndexType vertex0 = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
            const IndexType vertex1 = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i + 1;
            const IndexType vertex2 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;
            const IndexType vertex3 = k * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;
            const IndexType vertex4 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
            const IndexType vertex5 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i + 1;
            const IndexType vertex6 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i + 1;
            const IndexType vertex7 = ( k + 1 ) * ( xSize + 1 ) * ( ySize + 1 ) + ( j + 1 ) * ( xSize + 1 ) + i;

            const HexahedronMeshEntityType& cell = mesh.template getEntity< 3 >( cellIdx++ );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 0 ), vertex0 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 1 ), vertex1 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 2 ), vertex2 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 3 ), vertex3 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 4 ), vertex4 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 5 ), vertex5 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 6 ), vertex6 );
            EXPECT_EQ( cell.template getSubentityIndex< 0 >( 7 ), vertex7 );
         }

   // Test vertices -> cells superentities
   for( IndexType k = 0; k < zSize; k++ )
      for( IndexType j = 0; j <= ySize; j++ )
         for( IndexType i = 0; i <= xSize; i++ )
         {
            const IndexType vertexIndex = k * ( xSize + 1 ) * ( ySize + 1 ) + j * ( xSize + 1 ) + i;
            const VertexMeshEntityType& vertex = mesh.template getEntity< 0 >( vertexIndex );

            if( ( i == 0 || i == xSize ) && ( j == 0 || j == ySize ) && ( k == 0 || k == zSize ) ) {
               EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 3 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 3 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
            }
            else if( i == 0 || i == xSize || j == 0 || j == ySize || k == 0 || k == zSize ) {
               if( ( i != 0 && i != xSize && j != 0 && j != ySize ) ||
                   ( i != 0 && i != xSize && k != 0 && k != zSize ) ||
                   ( j != 0 && j != ySize && k != 0 && k != zSize ) )
               {
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 5 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 8 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 4 );
                  if( k == 0 || k == zSize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - 1 ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - 1 ) * xSize + i     );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j     ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j     ) * xSize + i     );
                  }
                  else if( j == 0 || j == ySize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - 1 ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i     );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k     ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k     ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i     );
                  }
                  else {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - 1 ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - 1 ) * xSize * ySize + ( j     ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k     ) * xSize * ySize + ( j - 1 ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k     ) * xSize * ySize + ( j     ) * xSize + i - ( i == xSize ) );
                  }
               }
               else {
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 4 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 5 );
                  EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 2 );
                  if( k != 0 && k != zSize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k     ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - ( i == xSize ) );
                  }
                  else if( j != 0 && j != ySize ) {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - 1 ) * xSize + i - ( i == xSize ) );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j     ) * xSize + i - ( i == xSize ) );
                  }
                  else {
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i - 1 );
                     EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - ( k == zSize ) ) * xSize * ySize + ( j - ( j == ySize ) ) * xSize + i     );
                  }
               }
            }
            else {
               EXPECT_EQ( vertex.template getSuperentitiesCount< 1 >(), 6 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 2 >(), 12 );
               EXPECT_EQ( vertex.template getSuperentitiesCount< 3 >(), 8 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 0 ),   ( k - 1 ) * xSize * ySize + ( j - 1 ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 1 ),   ( k - 1 ) * xSize * ySize + ( j - 1 ) * xSize + i     );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 2 ),   ( k - 1 ) * xSize * ySize + ( j     ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 3 ),   ( k - 1 ) * xSize * ySize + ( j     ) * xSize + i     );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 4 ),   ( k     ) * xSize * ySize + ( j - 1 ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 5 ),   ( k     ) * xSize * ySize + ( j - 1 ) * xSize + i     );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 6 ),   ( k     ) * xSize * ySize + ( j     ) * xSize + i - 1 );
               EXPECT_EQ( vertex.template getSuperentityIndex< 3 >( 7 ),   ( k     ) * xSize * ySize + ( j     ) * xSize + i     );
            }
         }

   // Tests for the dual graph layer
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), numberOfCells );
   cellIdx = 0;
   for( IndexType k = 0; k < zSize; k++ )
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      IndexType nnbrs = 6;
      if( i == 0 || i == xSize - 1 )
         --nnbrs;
      if( j == 0 || j == ySize - 1 )
         --nnbrs;
      if( k == 0 || k == zSize - 1 )
         --nnbrs;

      EXPECT_EQ( mesh.getCellNeighborsCount( cellIdx ), nnbrs );
      std::set< IndexType > neighbors;
      for( IndexType n = 0; n < nnbrs; n++ )
         neighbors.insert( mesh.getDualGraph().getRow( cellIdx ).getColumnIndex( n ) );

      // the cell itself should not be its own neighbor
      EXPECT_EQ( (IndexType) neighbors.count( cellIdx ), 0 );
      auto check_neighbor = [&]( IndexType i, IndexType j, IndexType k )
      {
         if( i >= 0 && i < xSize && j >= 0 && j < ySize && k >= 0 && k < zSize ) {
            EXPECT_EQ( (IndexType) neighbors.count( k * xSize * ySize + j * xSize + i ), 1 );
         }
      };
      // check neighbors over face
      check_neighbor( i - 1, j, k );
      check_neighbor( i + 1, j, k );
      check_neighbor( i, j - 1, k );
      check_neighbor( i, j + 1, k );
      check_neighbor( i, j, k - 1 );
      check_neighbor( i, j, k + 1 );

      ++cellIdx;
   }

   // Tests for the dual graph layer - with minCommonVertices override
   mesh.initializeDualGraph( mesh, 2 );
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), numberOfCells );
   cellIdx = 0;
   for( IndexType k = 0; k < zSize; k++ )
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      IndexType nnbrs = 18;
      if( i == 0 || i == xSize - 1 )
         nnbrs -= 5;
      if( j == 0 || j == ySize - 1 )
         nnbrs -= 5;
      if( k == 0 || k == zSize - 1 )
         nnbrs -= 5;
      if( (i == 0 || i == xSize - 1) && (j == 0 || j == ySize - 1) )
         ++nnbrs;
      if( (i == 0 || i == xSize - 1) && (k == 0 || k == zSize - 1) )
         ++nnbrs;
      if( (j == 0 || j == ySize - 1) && (k == 0 || k == zSize - 1) )
         ++nnbrs;

      EXPECT_EQ( mesh.getCellNeighborsCount( cellIdx ), nnbrs );
      std::set< IndexType > neighbors;
      for( IndexType n = 0; n < nnbrs; n++ )
         neighbors.insert( mesh.getDualGraph().getRow( cellIdx ).getColumnIndex( n ) );

      // the cell itself should not be its own neighbor
      EXPECT_EQ( (IndexType) neighbors.count( cellIdx ), 0 );
      auto check_neighbor = [&]( IndexType i, IndexType j, IndexType k )
      {
         if( i >= 0 && i < xSize && j >= 0 && j < ySize && k >= 0 && k < zSize ) {
            EXPECT_EQ( (IndexType) neighbors.count( k * xSize * ySize + j * xSize + i ), 1 );
         }
      };
      // check neighbors over face
      check_neighbor( i - 1, j, k );
      check_neighbor( i + 1, j, k );
      check_neighbor( i, j - 1, k );
      check_neighbor( i, j + 1, k );
      check_neighbor( i, j, k - 1 );
      check_neighbor( i, j, k + 1 );
      // check neighbors over edge
      check_neighbor( i - 1, j - 1, k );
      check_neighbor( i - 1, j + 1, k );
      check_neighbor( i + 1, j - 1, k );
      check_neighbor( i + 1, j + 1, k );
      check_neighbor( i - 1, j, k - 1 );
      check_neighbor( i - 1, j, k + 1 );
      check_neighbor( i + 1, j, k - 1 );
      check_neighbor( i + 1, j, k + 1 );
      check_neighbor( i, j - 1, k - 1 );
      check_neighbor( i, j - 1, k + 1 );
      check_neighbor( i, j + 1, k - 1 );
      check_neighbor( i, j + 1, k + 1 );

      ++cellIdx;
   }

   // Tests for the dual graph layer - with minCommonVertices override
   mesh.initializeDualGraph( mesh, 1 );
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), numberOfCells );
   cellIdx = 0;
   for( IndexType k = 0; k < zSize; k++ )
   for( IndexType j = 0; j < ySize; j++ )
   for( IndexType i = 0; i < xSize; i++ )
   {
      IndexType nnbrs = 26;
      if( i == 0 || i == xSize - 1 )
         nnbrs -= 9;
      if( j == 0 || j == ySize - 1 )
         nnbrs -= 9;
      if( k == 0 || k == zSize - 1 )
         nnbrs -= 9;
      if( (i == 0 || i == xSize - 1) && (j == 0 || j == ySize - 1) )
         nnbrs += 3;
      if( (i == 0 || i == xSize - 1) && (k == 0 || k == zSize - 1) )
         nnbrs += 3;
      if( (j == 0 || j == ySize - 1) && (k == 0 || k == zSize - 1) )
         nnbrs += 3;
      if( (i == 0 || i == xSize - 1) && (j == 0 || j == ySize - 1) && (k == 0 || k == zSize - 1) )
         --nnbrs;

      EXPECT_EQ( mesh.getCellNeighborsCount( cellIdx ), nnbrs );
      std::set< IndexType > neighbors;
      for( IndexType n = 0; n < nnbrs; n++ )
         neighbors.insert( mesh.getDualGraph().getRow( cellIdx ).getColumnIndex( n ) );

      // the cell itself should not be its own neighbor
      EXPECT_EQ( (IndexType) neighbors.count( cellIdx ), 0 );
      auto check_neighbor = [&]( IndexType i, IndexType j, IndexType k )
      {
         if( i >= 0 && i < xSize && j >= 0 && j < ySize && k >= 0 && k < zSize ) {
            EXPECT_EQ( (IndexType) neighbors.count( k * xSize * ySize + j * xSize + i ), 1 );
         }
      };
      // check neighbors over face
      check_neighbor( i - 1, j, k );
      check_neighbor( i + 1, j, k );
      check_neighbor( i, j - 1, k );
      check_neighbor( i, j + 1, k );
      check_neighbor( i, j, k - 1 );
      check_neighbor( i, j, k + 1 );
      // check neighbors over edge
      check_neighbor( i - 1, j - 1, k );
      check_neighbor( i - 1, j + 1, k );
      check_neighbor( i + 1, j - 1, k );
      check_neighbor( i + 1, j + 1, k );
      check_neighbor( i - 1, j, k - 1 );
      check_neighbor( i - 1, j, k + 1 );
      check_neighbor( i + 1, j, k - 1 );
      check_neighbor( i + 1, j, k + 1 );
      check_neighbor( i, j - 1, k - 1 );
      check_neighbor( i, j - 1, k + 1 );
      check_neighbor( i, j + 1, k - 1 );
      check_neighbor( i, j + 1, k + 1 );
      // check neighbors over vertex
      check_neighbor( i - 1, j - 1, k - 1 );
      check_neighbor( i - 1, j - 1, k + 1 );
      check_neighbor( i - 1, j + 1, k - 1 );
      check_neighbor( i - 1, j + 1, k + 1 );
      check_neighbor( i + 1, j - 1, k - 1 );
      check_neighbor( i + 1, j - 1, k + 1 );
      check_neighbor( i + 1, j + 1, k - 1 );
      check_neighbor( i + 1, j + 1, k + 1 );

      ++cellIdx;
   }

   // reset dual graph back to its default state
   mesh.initializeDualGraph( mesh );

   testFinishedMesh( mesh );
}

TEST( MeshTest, TwoPolygonsTest )
{
   using PolygonTestMesh = Mesh< TestTwoPolygonsMeshConfig >;
   using PolygonMeshEntityType = MeshEntity< TestTwoPolygonsMeshConfig, Devices::Host, Topologies::Polygon >;
   using EdgeMeshEntityType = typename PolygonMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename PolygonMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( PolygonMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing polygon entity does not store edges as required." );
   static_assert( PolygonMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing polygon entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store polygons as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store polygons as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );
   /****
    * We set-up the following situation

                     point4
                       /\
                      /  \
                     /    \
                    /      \
                 edge5    edge4
                  /          \
                 /  polygon1  \
                /  (triangle)  \
               /                \
              /                  \
      point3 /________edge2_______\ point2
             |                    |
             |                    |
             |      polygon0      |
             |       (quad)       |
           edge3                edge1
             |                    |
             |____________________|
      point0         edge0          point1

   */

   PointType point0( 0.0, 0.0 ),
             point1( 1.0, 0.0 ),
             point2( 0.0, 0.5 ),
             point3( 1.0, 0.5 ),
             point4( 0.5, 1.0 );

   PolygonTestMesh mesh;
   MeshBuilder< PolygonTestMesh > meshBuilder;

   meshBuilder.setEntitiesCount( 5, 2 );

   meshBuilder.setPoint( 0, point0 );
   meshBuilder.setPoint( 1, point1 );
   meshBuilder.setPoint( 2, point2 );
   meshBuilder.setPoint( 3, point3 );
   meshBuilder.setPoint( 4, point4 );

   meshBuilder.setCellCornersCounts( { 4, 3 } );

   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 3, 3 );

   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 3 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 2 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 4 );

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // tests for entities counts
   EXPECT_EQ( mesh.getEntitiesCount< 2 >(), 2 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(), 6 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(), 5 );

   // tests for points
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).getPoint(), point0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).getPoint(), point1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).getPoint(), point2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).getPoint(), point3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 4 ).getPoint(), point4 );

   // tests for the subentities layer
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSubentityIndex< 0 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSubentityIndex< 0 >( 1 ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).template getSubentityIndex< 0 >( 0 ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).template getSubentityIndex< 0 >( 1 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSubentityIndex< 0 >( 0 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSubentityIndex< 0 >( 1 ), 3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).template getSubentityIndex< 0 >( 0 ), 3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).template getSubentityIndex< 0 >( 1 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).template getSubentityIndex< 0 >( 0 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).template getSubentityIndex< 0 >( 1 ), 4 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 5 ).template getSubentityIndex< 0 >( 0 ), 4 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 5 ).template getSubentityIndex< 0 >( 1 ), 3 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 0 >(),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 1 ), 1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 2 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 0 >( 3 ), 3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 1 >(),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 1 ), 1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 2 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex< 1 >( 3 ), 3 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 0 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 0 ), 3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 1 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 0 >( 2 ), 4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 0 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 0 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 1 ), 4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex< 1 >( 2 ), 5 );

   // tests for the superentities layer
   ASSERT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 1 >( 1 ), 3 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 0 ).template getSuperentityIndex< 2 >( 0 ), 0 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 1 >( 1 ), 1 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 1 ).template getSuperentityIndex< 2 >( 0 ), 0 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 2 ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).template getSuperentityIndex< 1 >( 0 ), 1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).template getSuperentityIndex< 1 >( 1 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).template getSuperentityIndex< 1 >( 2 ), 4 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 2 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).template getSuperentityIndex< 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 2 ).template getSuperentityIndex< 2 >( 1 ), 1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 3 ).template getSuperentitiesCount< 1 >(),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).template getSuperentityIndex< 1 >( 0 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).template getSuperentityIndex< 1 >( 1 ), 3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).template getSuperentityIndex< 1 >( 2 ), 5 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 3 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).template getSuperentityIndex< 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 3 ).template getSuperentityIndex< 2 >( 1 ), 1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 4 ).template getSuperentitiesCount< 1 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 4 ).template getSuperentityIndex< 1 >( 0 ), 4 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 4 ).template getSuperentityIndex< 1 >( 1 ), 5 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 4 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 4 ).template getSuperentityIndex< 2 >( 0 ), 1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 0 ).template getSuperentityIndex< 2 >( 0 ), 0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 1 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 1 ).template getSuperentityIndex< 2 >( 0 ), 0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 2 ).template getSuperentitiesCount< 2 >(),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSuperentityIndex< 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 2 ).template getSuperentityIndex< 2 >( 1 ), 1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 3 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 3 ).template getSuperentityIndex< 2 >( 0 ), 0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 4 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 4 ).template getSuperentityIndex< 2 >( 0 ), 1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 5 ).template getSuperentitiesCount< 2 >(),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 5 ).template getSuperentityIndex< 2 >( 0 ), 1 );

   // tests for the dual graph layer
   ASSERT_EQ( mesh.getCellNeighborsCount( 0 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex( 0, 0 ), 1 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 1 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex( 1, 0 ), 0 );

   testFinishedMesh( mesh );
}

TEST( MeshTest, SevenPolygonsTest )
{
   using PolygonTestMesh = Mesh< TestSevenPolygonsMeshConfig >;
   using PolygonMeshEntityType = MeshEntity< TestSevenPolygonsMeshConfig, Devices::Host, Topologies::Polygon >;
   using EdgeMeshEntityType = typename PolygonMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename PolygonMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( PolygonMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing polygon entity does not store edges as required." );
   static_assert( PolygonMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing polygon entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store polygons as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store polygons as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 2, RealType > >::value,
                  "unexpected PointType" );

   PolygonTestMesh mesh;
   MeshBuilder< PolygonTestMesh > meshBuilder;

   meshBuilder.setEntitiesCount( 16, 7 );

   meshBuilder.setPoint(  0, PointType( 0.250, 0.150 ) );
   meshBuilder.setPoint(  1, PointType( 0.150, 0.250 ) );
   meshBuilder.setPoint(  2, PointType( 0.900, 0.500 ) );
   meshBuilder.setPoint(  3, PointType( 0.750, 0.275 ) );
   meshBuilder.setPoint(  4, PointType( 0.500, 0.900 ) );
   meshBuilder.setPoint(  5, PointType( 0.275, 0.750 ) );
   meshBuilder.setPoint(  6, PointType( 0.000, 0.250 ) );
   meshBuilder.setPoint(  7, PointType( 0.250, 0.000 ) );
   meshBuilder.setPoint(  8, PointType( 0.000, 0.000 ) );
   meshBuilder.setPoint(  9, PointType( 0.750, 0.000 ) );
   meshBuilder.setPoint( 10, PointType( 0.000, 0.750 ) );
   meshBuilder.setPoint( 11, PointType( 1.000, 0.500 ) );
   meshBuilder.setPoint( 12, PointType( 1.000, 0.000 ) );
   meshBuilder.setPoint( 13, PointType( 0.500, 1.000 ) );
   meshBuilder.setPoint( 14, PointType( 1.000, 1.000 ) );
   meshBuilder.setPoint( 15, PointType( 0.000, 1.000 ) );

   /****
    * Setup the following polygons:
    *
    *   1     0     3     2     4     5
    *   8     7     0     1     6
    *   9     3     0     7
    *   6     1     5    10
    *  12    11     2     3     9
    *  13     4     2    11    14
    *  10     5     4    13    15
    */

   meshBuilder.setCellCornersCounts( { 6, 5, 4, 4, 5, 5, 5 } );

   //   1     0     3     2     4     5
   meshBuilder.getCellSeed( 0 ).setCornerId( 0,  1 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1,  0 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2,  3 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 3,  2 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 4,  4 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 5,  5 );

   //   8     7     0     1     6
   meshBuilder.getCellSeed( 1 ).setCornerId( 0,  8 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1,  7 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2,  0 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 3,  1 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 4,  6 );

   //   9     3     0     7
   meshBuilder.getCellSeed( 2 ).setCornerId( 0,  9 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 1,  3 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 2,  0 );
   meshBuilder.getCellSeed( 2 ).setCornerId( 3,  7 );

   //   6     1     5    10
   meshBuilder.getCellSeed( 3 ).setCornerId( 0,  6 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 1,  1 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 2,  5 );
   meshBuilder.getCellSeed( 3 ).setCornerId( 3, 10 );

   //  12    11     2     3     9
   meshBuilder.getCellSeed( 4 ).setCornerId( 0, 12 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 1, 11 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 2,  2 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 3,  3 );
   meshBuilder.getCellSeed( 4 ).setCornerId( 4,  9 );

   //  13     4     2    11    14
   meshBuilder.getCellSeed( 5 ).setCornerId( 0, 13 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 1,  4 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 2,  2 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 3, 11 );
   meshBuilder.getCellSeed( 5 ).setCornerId( 4, 14 );

   //  10     5     4    13    15
   meshBuilder.getCellSeed( 6 ).setCornerId( 0, 10 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 1,  5 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 2,  4 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 3, 13 );
   meshBuilder.getCellSeed( 6 ).setCornerId( 4, 15 );

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // tests for entities counts
   EXPECT_EQ( mesh.getEntitiesCount< 2 >(), 7 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(), 22 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(), 16 );

   // tests for the subentities layer
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSubentityIndex< 0 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSubentityIndex< 0 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSubentityIndex< 0 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSubentityIndex< 0 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSubentityIndex< 0 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSubentityIndex< 0 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSubentityIndex< 0 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSubentityIndex< 0 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSubentityIndex< 0 >( 1 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSubentityIndex< 0 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSubentityIndex< 0 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSubentityIndex< 0 >( 1 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSubentityIndex< 0 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSubentityIndex< 0 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSubentityIndex< 0 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSubentityIndex< 0 >( 1 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSubentityIndex< 0 >( 0 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSubentityIndex< 0 >( 1 ), 10 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 0 >(   ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 2 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 3 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 5 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 1 >(   ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 5 ),  5 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 0 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 2 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 3 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 4 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 1 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 2 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 3 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 4 ),  9 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 2 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 3 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 2 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 3 ), 11 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 2 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 3 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 2 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 3 ), 13 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentitiesCount< 0 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 4 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentitiesCount< 1 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 1 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 3 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 4 ), 16 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentitiesCount< 0 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 3 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 4 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentitiesCount< 1 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 0 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 2 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 3 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 4 ), 19 );

   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentitiesCount< 0 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 3 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 4 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentitiesCount< 1 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 2 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 3 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 4 ), 21 );

   // tests for the superentities layer
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 2 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 2 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 2 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 2 ), 10 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 2 ), 17 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 2 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 2 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 2 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 2 ), 13 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 2 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentitiesCount< 1 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 1 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 1 >( 1 ),  9 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentitiesCount< 2 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 1 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 1 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 1 >( 2 ), 16 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 1 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 1 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 1 >( 2 ), 21 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 1 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 1 >( 1 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 1 >( 2 ), 18 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 2 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 2 >( 1 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentitiesCount< 1 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 1 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 1 >( 1 ), 16 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentitiesCount< 2 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 2 >( 0 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 1 >( 0 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 1 >( 1 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 1 >( 2 ), 20 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentitiesCount< 1 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 1 >( 0 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 1 >( 1 ), 19 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentitiesCount< 2 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentitiesCount< 1 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 1 >( 0 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 1 >( 1 ), 21 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentitiesCount< 2 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 2 >( 0 ),  6 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 1 ), 1 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 1 ), 2 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 1 ), 4 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 1 ), 5 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 1 ), 6 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 0 ), 0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 1 ), 3 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 0 ), 1 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 0 ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 1 ), 2 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 2 >( 0 ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 2 >( 1 ), 3 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 2 >( 0 ), 1 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 2 >( 0 ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 2 >( 1 ), 4 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentityIndex  < 2 >( 0 ), 2 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 2 >( 0 ), 3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 2 >( 1 ), 6 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentityIndex  < 2 >( 0 ), 3 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentityIndex  < 2 >( 0 ), 4 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentityIndex  < 2 >( 0 ), 4 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentityIndex  < 2 >( 1 ), 5 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentityIndex  < 2 >( 0 ), 4 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentitiesCount< 2 >(   ), 2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentityIndex  < 2 >( 0 ), 5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentityIndex  < 2 >( 1 ), 6 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 18 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSuperentityIndex  < 2 >( 0 ), 5 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 19 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSuperentityIndex  < 2 >( 0 ), 5 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 20 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSuperentityIndex  < 2 >( 0 ), 6 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 21 ).template getSuperentitiesCount< 2 >(   ), 1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSuperentityIndex  < 2 >( 0 ), 6 );

   // tests for the dual graph layer
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), 7 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 0    ), 6 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 0 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 1 ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 2 ), 2 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 3 ), 4 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 4 ), 5 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 5 ), 6 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 1    ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 1, 0 ), 2 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 1, 1 ), 0 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 1, 2 ), 3 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 2    ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 2, 0 ), 4 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 2, 1 ), 0 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 2, 2 ), 1 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 3    ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 3, 0 ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 3, 1 ), 0 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 3, 2 ), 6 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 4    ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 4, 0 ), 5 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 4, 1 ), 0 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 4, 2 ), 2 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 5    ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 5, 0 ), 6 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 5, 1 ), 0 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 5, 2 ), 4 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 6    ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 6, 0 ), 3 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 6, 1 ), 0 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 6, 2 ), 5 );

   //Tags test for boundaries
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).getTag() & TNL::Meshes::EntityTags::BoundaryEntity, 0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).getTag() & TNL::Meshes::EntityTags::BoundaryEntity, 1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).getTag() & TNL::Meshes::EntityTags::BoundaryEntity, 1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).getTag() & TNL::Meshes::EntityTags::BoundaryEntity, 1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).getTag() & TNL::Meshes::EntityTags::BoundaryEntity, 1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).getTag() & TNL::Meshes::EntityTags::BoundaryEntity, 1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).getTag() & TNL::Meshes::EntityTags::BoundaryEntity, 1 );

   testFinishedMesh( mesh );
}

TEST( MeshTest, TwoWedgesTest )
{
   using WedgeTestMesh = Mesh< TestTwoWedgesMeshConfig >;
   using WedgeMeshEntityType = MeshEntity< TestTwoWedgesMeshConfig, Devices::Host, Topologies::Wedge >;
   using PolygonMeshEntityType = typename WedgeMeshEntityType::SubentityTraits< 2 >::SubentityType;
   using EdgeMeshEntityType = typename WedgeMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename WedgeMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( WedgeMeshEntityType::SubentityTraits< 2 >::storageEnabled, "Testing wedge entity does not store polygons as required." );
   static_assert( WedgeMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing wedge entity does not store edges as required." );
   static_assert( WedgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing wedge entity does not store vertices as required." );

   static_assert( PolygonMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing polygon entity does not store edges as required." );
   static_assert( PolygonMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing polygon entity does not store vertices as required." );
   static_assert( PolygonMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing polygon entity does not store wedges as required." );

   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing edge entity does not store wedges as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store polygons as required." );

   static_assert( VertexMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing vertex entity does not store wedges as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 3, RealType > >::value,
                  "unexpected PointType" );

   PointType point0( 1.0,  0.0, 0.0 ),
             point1( 1.0,  0.0, 1.0 ),
             point2( 1.0,  1.0, 0.5 ),
             point3( 0.0,  0.0, 0.0 ),
             point4( 0.0,  0.0, 1.0 ),
             point5( 0.0,  1.0, 0.5 ),
             point6( 1.0, -1.0, 0.5 ),
             point7( 0.0, -1.0, 0.5 );

   WedgeTestMesh mesh;
   MeshBuilder< WedgeTestMesh > meshBuilder;

   meshBuilder.setEntitiesCount( 8, 2 );

   meshBuilder.setPoint( 0, point0 );
   meshBuilder.setPoint( 1, point1 );
   meshBuilder.setPoint( 2, point2 );
   meshBuilder.setPoint( 3, point3 );
   meshBuilder.setPoint( 4, point4 );
   meshBuilder.setPoint( 5, point5 );
   meshBuilder.setPoint( 6, point6 );
   meshBuilder.setPoint( 7, point7 );

   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 3, 3 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 4, 4 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 5, 5 );

   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 6 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 3, 3 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 4, 4 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 5, 7 );

   ASSERT_TRUE( meshBuilder.build( mesh ) );


   // tests for entities counts
   EXPECT_EQ( mesh.getEntitiesCount< 3 >(), 2 );
   EXPECT_EQ( mesh.getEntitiesCount< 2 >(), 9 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(), 14 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(), 8 );


   // tests for the subentities layer
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSubentityIndex< 0 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSubentityIndex< 0 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSubentityIndex< 0 >( 1 ),  6 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 2 ),  2 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 2 ),  5 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 3 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 2 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 3 ),  5 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 3 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 2 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 3 ),  4 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 2 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 3 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 2 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 3 ),  3 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 2 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 2 ), 10 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 2 ), 12 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 0 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 0 >( 3 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 1 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 1 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 1 >( 2 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 1 >( 3 ), 12 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 0 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 0 >( 3 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 1 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 1 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 1 >( 2 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 1 >( 3 ), 11 );

   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 0 >(   ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 5 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 1 >(   ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 5 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 6 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 7 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 8 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 4 ),  4 );

   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 0 >(   ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 5 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 1 >(   ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 2 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 4 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 5 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 6 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 7 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 8 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 2 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 3 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 4 ),  4 );


   // tests for the superentities layer
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 3 ), 10 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 3 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 4 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 2 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 3 ),  9 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 3 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 4 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 2 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 3 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 3 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 4 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 2 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 3 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 3 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 4 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 2 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 2 ), 13 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 1 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 2 ), 13 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 2 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentityIndex  < 2 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 2 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentityIndex  < 2 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 0 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 1 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 2 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 3 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 4 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSuperentityIndex  < 3 >( 1 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSuperentityIndex  < 3 >( 0 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 6 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSuperentityIndex  < 3 >( 0 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 7 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSuperentityIndex  < 3 >( 0 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 8 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSuperentityIndex  < 3 >( 0 ),  1 );


   // tests for the dual graph layer
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), 2 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 0    ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 0 ), 1 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 1    ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 1, 0 ), 0 );

   testFinishedMesh( mesh );
}

TEST( MeshTest, TwoPyramidsTest )
{
   using PyramidTestMesh = Mesh< TestTwoPyramidsMeshConfig >;
   using PyramidMeshEntityType = MeshEntity< TestTwoPyramidsMeshConfig, Devices::Host, Topologies::Pyramid >;
   using PolygonMeshEntityType = typename PyramidMeshEntityType::SubentityTraits< 2 >::SubentityType;
   using EdgeMeshEntityType = typename PyramidMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename PyramidMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( PyramidMeshEntityType::SubentityTraits< 2 >::storageEnabled, "Testing pyramid entity does not store polygons as required." );
   static_assert( PyramidMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing pyramid entity does not store edges as required." );
   static_assert( PyramidMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing pyramid entity does not store vertices as required." );

   static_assert( PolygonMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing polygon entity does not store edges as required." );
   static_assert( PolygonMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing polygon entity does not store vertices as required." );
   static_assert( PolygonMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing polygon entity does not store pyramids as required." );

   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing edge entity does not store pyramids as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store polygons as required." );

   static_assert( VertexMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing vertex entity does not store pyramids as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 3, RealType > >::value,
                  "unexpected PointType" );

   PointType point0( 0.0,  0.0, 0.0 ),
             point1( 1.0,  0.0, 0.0 ),
             point2( 1.0,  0.0, 1.0 ),
             point3( 0.0,  0.0, 1.0 ),
             point4( 0.5,  1.0, 0.5 ),
             point5( 0.5, -1.0, 0.5 );

   PyramidTestMesh mesh;
   MeshBuilder< PyramidTestMesh > meshBuilder;

   meshBuilder.setEntitiesCount( 6, 2 );

   meshBuilder.setPoint( 0, point0 );
   meshBuilder.setPoint( 1, point1 );
   meshBuilder.setPoint( 2, point2 );
   meshBuilder.setPoint( 3, point3 );
   meshBuilder.setPoint( 4, point4 );
   meshBuilder.setPoint( 5, point5 );

   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 3, 3 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 4, 4 );

   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 3, 3 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 4, 5 );

   ASSERT_TRUE( meshBuilder.build( mesh ) );


   // tests for entities counts
   EXPECT_EQ( mesh.getEntitiesCount< 3 >(), 2 );
   EXPECT_EQ( mesh.getEntitiesCount< 2 >(), 9 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(), 12 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(), 6 );


   // tests for the subentities layer
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 1 ),  5 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 0 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 0 >( 3 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 0 ).template getSubentityIndex  < 1 >( 3 ),  3 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 0 >( 2 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 1 ).template getSubentityIndex  < 1 >( 2 ),  4 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 0 >( 2 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 2 ).template getSubentityIndex  < 1 >( 2 ),  5 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 0 >( 2 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 3 ).template getSubentityIndex  < 1 >( 2 ),  6 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 0 >( 2 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 4 ).template getSubentityIndex  < 1 >( 2 ),  7 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 0 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 5 ).template getSubentityIndex  < 1 >( 2 ),  8 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 0 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 6 ).template getSubentityIndex  < 1 >( 2 ),  9 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 0 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 1 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 1 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 7 ).template getSubentityIndex  < 1 >( 2 ), 10 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentitiesCount< 0 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 0 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 1 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 8 ).template getSubentityIndex  < 1 >( 2 ), 11 );

   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 0 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 0 >( 4 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 1 >(   ),  8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 5 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 6 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 1 >( 7 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex  < 2 >( 4 ),  4 );

   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 0 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 0 >( 4 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 1 >(   ),  8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 4 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 5 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 6 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 1 >( 7 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 3 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex  < 2 >( 4 ),  8 );


   // tests for the superentities layer
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 3 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 3 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 4 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 2 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 3 ),  9 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 3 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 4 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 3 ), 10 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 2 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 3 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 4 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 2 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 3 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 3 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 4 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 3 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 2 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 2 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 3 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 2 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 3 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 2 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 2 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 3 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 2 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  0 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSuperentityIndex  < 3 >( 1 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  1 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  2 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  3 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  4 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  6 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSuperentityIndex  < 3 >( 0 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  7 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSuperentityIndex  < 3 >( 0 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  8 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSuperentityIndex  < 3 >( 0 ),  1 );


   // tests for the dual graph layer
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), 2 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 0    ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 0 ), 1 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 1    ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 1, 0 ), 0 );

   testFinishedMesh( mesh );
}

TEST( MeshTest, TwoPolyhedronsTest )
{
   using PolyhedronTestMesh = Mesh< TestTwoPolyhedronsMeshConfig >;
   using PolyhedronMeshEntityType = MeshEntity< TestTwoPolyhedronsMeshConfig, Devices::Host, Topologies::Polyhedron >;
   using PolygonMeshEntityType = typename PolyhedronMeshEntityType::SubentityTraits< 2 >::SubentityType;
   using EdgeMeshEntityType = typename PolyhedronMeshEntityType::SubentityTraits< 1 >::SubentityType;
   using VertexMeshEntityType = typename PolyhedronMeshEntityType::SubentityTraits< 0 >::SubentityType;

   static_assert( PolyhedronMeshEntityType::SubentityTraits< 2 >::storageEnabled, "Testing polyhedron entity does not store polygons as required." );
   static_assert( PolyhedronMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing polyhedron entity does not store edges as required." );
   static_assert( PolyhedronMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing polyhedron entity does not store vertices as required." );

   static_assert( PolygonMeshEntityType::SubentityTraits< 1 >::storageEnabled, "Testing polygon entity does not store edges as required." );
   static_assert( PolygonMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing polygon entity does not store vertices as required." );
   static_assert( PolygonMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing polygon entity does not store pyramids as required." );

   static_assert( EdgeMeshEntityType::SubentityTraits< 0 >::storageEnabled, "Testing edge entity does not store vertices as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing edge entity does not store pyramids as required." );
   static_assert( EdgeMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing edge entity does not store polygons as required." );

   static_assert( VertexMeshEntityType::SuperentityTraits< 3 >::storageEnabled, "Testing vertex entity does not store pyramids as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 2 >::storageEnabled, "Testing vertex entity does not store triangles as required." );
   static_assert( VertexMeshEntityType::SuperentityTraits< 1 >::storageEnabled, "Testing vertex entity does not store edges as required." );

   using PointType = typename VertexMeshEntityType::PointType;
   static_assert( std::is_same< PointType, Containers::StaticVector< 3, RealType > >::value,
                  "unexpected PointType" );

   PointType point0 ( -1.25000, 1.16650, 1.20300 ),
             point1 ( -1.20683, 1.16951, 1.20537 ),
             point2 ( -1.16843, 1.19337, 1.17878 ),
             point3 ( -1.21025, 1.21901, 1.15383 ),
             point4 ( -1.25000, 1.21280, 1.15670 ),
             point5 ( -1.20816, 1.25000, 1.16756 ),
             point6 ( -1.25000, 1.25000, 1.18056 ),
             point7 ( -1.14802, 1.21553, 1.21165 ),
             point8 ( -1.16186, 1.25000, 1.21385 ),
             point9 ( -1.20307, 1.17486, 1.25000 ),
             point10( -1.25000, 1.18056, 1.25000 ),
             point11( -1.15677, 1.22115, 1.25000 ),
             point12( -1.18056, 1.25000, 1.25000 ),
             point13( -1.25000, 1.25000, 1.25000 ),
             point14( -1.09277, 1.20806, 1.19263 ),
             point15( -1.07219, 1.22167, 1.17994 ),
             point16( -1.07215, 1.25000, 1.18679 ),
             point17( -1.05697, 1.21124, 1.19697 ),
             point18( -1.04607, 1.21508, 1.22076 ),
             point19( -1.02140, 1.25000, 1.22293 ),
             point20( -1.06418, 1.22115, 1.25000 ),
             point21( -1.04167, 1.25000, 1.25000 );

   PolyhedronTestMesh mesh;
   MeshBuilder< PolyhedronTestMesh > meshBuilder;

   meshBuilder.setEntitiesCount( 22, 2, 16 );

   meshBuilder.setPoint(  0, point0  );
   meshBuilder.setPoint(  1, point1  );
   meshBuilder.setPoint(  2, point2  );
   meshBuilder.setPoint(  3, point3  );
   meshBuilder.setPoint(  4, point4  );
   meshBuilder.setPoint(  5, point5  );
   meshBuilder.setPoint(  6, point6  );
   meshBuilder.setPoint(  7, point7  );
   meshBuilder.setPoint(  8, point8  );
   meshBuilder.setPoint(  9, point9  );
   meshBuilder.setPoint( 10, point10 );
   meshBuilder.setPoint( 11, point11 );
   meshBuilder.setPoint( 12, point12 );
   meshBuilder.setPoint( 13, point13 );
   meshBuilder.setPoint( 14, point14 );
   meshBuilder.setPoint( 15, point15 );
   meshBuilder.setPoint( 16, point16 );
   meshBuilder.setPoint( 17, point17 );
   meshBuilder.setPoint( 18, point18 );
   meshBuilder.setPoint( 19, point19 );
   meshBuilder.setPoint( 20, point20 );
   meshBuilder.setPoint( 21, point21 );

   /****
    * Setup the following faces (polygons):
    *
    *   0     1     2     3     4
    *   4     3     5     6
    *   5     3     2     7     8
    *   9     1     0    10
    *  11     7     2     1     9
    *   8     7    11    12
    *  13    12    11     9    10
    *  13    10     0     4     6
    *  13     6     5     8    12
    *   8     7    14    15    16
    *  16    15    17    18    19
    *  20    18    17    14     7    11
    *  17    15    14
    *  21    19    18    20
    *  21    20    11    12
    *  12     8    16    19    21
    *
    * NOTE: indices refer to the points
    */

   meshBuilder.setFaceCornersCounts( { 5, 4, 5, 4, 5, 4, 5, 5, 5, 5, 5, 6, 3, 4, 4, 5 } );

   //   0     1     2     3     4
   meshBuilder.getFaceSeed( 0 ).setCornerId( 0, 0 );
   meshBuilder.getFaceSeed( 0 ).setCornerId( 1, 1 );
   meshBuilder.getFaceSeed( 0 ).setCornerId( 2, 2 );
   meshBuilder.getFaceSeed( 0 ).setCornerId( 3, 3 );
   meshBuilder.getFaceSeed( 0 ).setCornerId( 4, 4 );

   //   4     3     5     6
   meshBuilder.getFaceSeed( 1 ).setCornerId( 0, 4 );
   meshBuilder.getFaceSeed( 1 ).setCornerId( 1, 3 );
   meshBuilder.getFaceSeed( 1 ).setCornerId( 2, 5 );
   meshBuilder.getFaceSeed( 1 ).setCornerId( 3, 6 );

   //   5     3     2     7     8
   meshBuilder.getFaceSeed( 2 ).setCornerId( 0, 5 );
   meshBuilder.getFaceSeed( 2 ).setCornerId( 1, 3 );
   meshBuilder.getFaceSeed( 2 ).setCornerId( 2, 2 );
   meshBuilder.getFaceSeed( 2 ).setCornerId( 3, 7 );
   meshBuilder.getFaceSeed( 2 ).setCornerId( 4, 8 );

   //   9     1     0    10
   meshBuilder.getFaceSeed( 3 ).setCornerId( 0, 9 );
   meshBuilder.getFaceSeed( 3 ).setCornerId( 1, 1 );
   meshBuilder.getFaceSeed( 3 ).setCornerId( 2, 0 );
   meshBuilder.getFaceSeed( 3 ).setCornerId( 3, 10 );

   //  11     7     2     1     9
   meshBuilder.getFaceSeed( 4 ).setCornerId( 0, 11 );
   meshBuilder.getFaceSeed( 4 ).setCornerId( 1, 7 );
   meshBuilder.getFaceSeed( 4 ).setCornerId( 2, 2 );
   meshBuilder.getFaceSeed( 4 ).setCornerId( 3, 1 );
   meshBuilder.getFaceSeed( 4 ).setCornerId( 4, 9 );

   //   8     7    11    12
   meshBuilder.getFaceSeed( 5 ).setCornerId( 0, 8 );
   meshBuilder.getFaceSeed( 5 ).setCornerId( 1, 7 );
   meshBuilder.getFaceSeed( 5 ).setCornerId( 2, 11 );
   meshBuilder.getFaceSeed( 5 ).setCornerId( 3, 12 );

   //  13    12    11     9    10
   meshBuilder.getFaceSeed( 6 ).setCornerId( 0, 13 );
   meshBuilder.getFaceSeed( 6 ).setCornerId( 1, 12 );
   meshBuilder.getFaceSeed( 6 ).setCornerId( 2, 11 );
   meshBuilder.getFaceSeed( 6 ).setCornerId( 3, 9 );
   meshBuilder.getFaceSeed( 6 ).setCornerId( 4, 10 );

   //  13    10     0     4     6
   meshBuilder.getFaceSeed( 7 ).setCornerId( 0, 13 );
   meshBuilder.getFaceSeed( 7 ).setCornerId( 1, 10 );
   meshBuilder.getFaceSeed( 7 ).setCornerId( 2, 0 );
   meshBuilder.getFaceSeed( 7 ).setCornerId( 3, 4 );
   meshBuilder.getFaceSeed( 7 ).setCornerId( 4, 6 );

   //  13     6     5     8    12
   meshBuilder.getFaceSeed( 8 ).setCornerId( 0, 13 );
   meshBuilder.getFaceSeed( 8 ).setCornerId( 1, 6 );
   meshBuilder.getFaceSeed( 8 ).setCornerId( 2, 5 );
   meshBuilder.getFaceSeed( 8 ).setCornerId( 3, 8 );
   meshBuilder.getFaceSeed( 8 ).setCornerId( 4, 12 );

   //   8     7    14    15    16
   meshBuilder.getFaceSeed( 9 ).setCornerId( 0, 8 );
   meshBuilder.getFaceSeed( 9 ).setCornerId( 1, 7 );
   meshBuilder.getFaceSeed( 9 ).setCornerId( 2, 14 );
   meshBuilder.getFaceSeed( 9 ).setCornerId( 3, 15 );
   meshBuilder.getFaceSeed( 9 ).setCornerId( 4, 16 );

   //  16    15    17    18    19
   meshBuilder.getFaceSeed( 10 ).setCornerId( 0, 16 );
   meshBuilder.getFaceSeed( 10 ).setCornerId( 1, 15 );
   meshBuilder.getFaceSeed( 10 ).setCornerId( 2, 17 );
   meshBuilder.getFaceSeed( 10 ).setCornerId( 3, 18 );
   meshBuilder.getFaceSeed( 10 ).setCornerId( 4, 19 );

   //  20    18    17    14     7    11
   meshBuilder.getFaceSeed( 11 ).setCornerId( 0, 20 );
   meshBuilder.getFaceSeed( 11 ).setCornerId( 1, 18 );
   meshBuilder.getFaceSeed( 11 ).setCornerId( 2, 17 );
   meshBuilder.getFaceSeed( 11 ).setCornerId( 3, 14 );
   meshBuilder.getFaceSeed( 11 ).setCornerId( 4, 7 );
   meshBuilder.getFaceSeed( 11 ).setCornerId( 5, 11 );

   //  17    15    14
   meshBuilder.getFaceSeed( 12 ).setCornerId( 0, 17 );
   meshBuilder.getFaceSeed( 12 ).setCornerId( 1, 15 );
   meshBuilder.getFaceSeed( 12 ).setCornerId( 2, 14 );

   //  21    19    18    20
   meshBuilder.getFaceSeed( 13 ).setCornerId( 0, 21 );
   meshBuilder.getFaceSeed( 13 ).setCornerId( 1, 19 );
   meshBuilder.getFaceSeed( 13 ).setCornerId( 2, 18 );
   meshBuilder.getFaceSeed( 13 ).setCornerId( 3, 20 );

   //  21    20    11    12
   meshBuilder.getFaceSeed( 14 ).setCornerId( 0, 21 );
   meshBuilder.getFaceSeed( 14 ).setCornerId( 1, 20 );
   meshBuilder.getFaceSeed( 14 ).setCornerId( 2, 11 );
   meshBuilder.getFaceSeed( 14 ).setCornerId( 3, 12 );

   //  12     8    16    19    21
   meshBuilder.getFaceSeed( 15 ).setCornerId( 0, 12 );
   meshBuilder.getFaceSeed( 15 ).setCornerId( 1, 8 );
   meshBuilder.getFaceSeed( 15 ).setCornerId( 2, 16 );
   meshBuilder.getFaceSeed( 15 ).setCornerId( 3, 19 );
   meshBuilder.getFaceSeed( 15 ).setCornerId( 4, 21 );

   /****
    * Setup the following cells (polyhedrons):
    *
    *   0     1     2     3     4     5      6     7     8
    *   9    10    11    12    13     5     14    15
    *
    * NOTE: indices refer to the faces
    */

   meshBuilder.setCellCornersCounts( { 9, 8 } );

   //   0     1     2     3     4     5      6     7     8
   meshBuilder.getCellSeed( 0 ).setCornerId( 0, 0 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 1, 1 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 2, 2 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 3, 3 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 4, 4 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 5, 5 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 6, 6 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 7, 7 );
   meshBuilder.getCellSeed( 0 ).setCornerId( 8, 8 );

   //   9    10    11    12    13     5     14    15
   meshBuilder.getCellSeed( 1 ).setCornerId( 0, 9 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 1, 10 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 2, 11 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 3, 12 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 4, 13 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 5, 5 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 6, 14 );
   meshBuilder.getCellSeed( 1 ).setCornerId( 7, 15 );

   ASSERT_TRUE( meshBuilder.build( mesh ) );

   // tests for entities counts
   EXPECT_EQ( mesh.getEntitiesCount< 3 >(), 2 );
   EXPECT_EQ( mesh.getEntitiesCount< 2 >(), 16 );
   EXPECT_EQ( mesh.getEntitiesCount< 1 >(), 35 );
   EXPECT_EQ( mesh.getEntitiesCount< 0 >(), 22 );

   // tests for the subentities layer
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSubentityIndex< 0 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSubentityIndex< 0 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSubentityIndex< 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSubentityIndex< 0 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSubentityIndex< 0 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSubentityIndex< 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSubentityIndex< 0 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSubentityIndex< 0 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSubentityIndex< 0 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSubentityIndex< 0 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSubentityIndex< 0 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSubentityIndex< 0 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSubentityIndex< 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSubentityIndex< 0 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSubentityIndex< 0 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSubentityIndex< 0 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSubentityIndex< 0 >( 1 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSubentityIndex< 0 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSubentityIndex< 0 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSubentityIndex< 0 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSubentityIndex< 0 >( 1 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSubentityIndex< 0 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSubentityIndex< 0 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSubentityIndex< 0 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSubentityIndex< 0 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSubentityIndex< 0 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSubentityIndex< 0 >( 1 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 22 ).template getSubentityIndex< 0 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 22 ).template getSubentityIndex< 0 >( 1 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 23 ).template getSubentityIndex< 0 >( 0 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 23 ).template getSubentityIndex< 0 >( 1 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 24 ).template getSubentityIndex< 0 >( 0 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 24 ).template getSubentityIndex< 0 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 25 ).template getSubentityIndex< 0 >( 0 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 25 ).template getSubentityIndex< 0 >( 1 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 26 ).template getSubentityIndex< 0 >( 0 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 26 ).template getSubentityIndex< 0 >( 1 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 27 ).template getSubentityIndex< 0 >( 0 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 27 ).template getSubentityIndex< 0 >( 1 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 28 ).template getSubentityIndex< 0 >( 0 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 28 ).template getSubentityIndex< 0 >( 1 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 29 ).template getSubentityIndex< 0 >( 0 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 29 ).template getSubentityIndex< 0 >( 1 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 30 ).template getSubentityIndex< 0 >( 0 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 30 ).template getSubentityIndex< 0 >( 1 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 31 ).template getSubentityIndex< 0 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 31 ).template getSubentityIndex< 0 >( 1 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 32 ).template getSubentityIndex< 0 >( 0 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 32 ).template getSubentityIndex< 0 >( 1 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 33 ).template getSubentityIndex< 0 >( 0 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 33 ).template getSubentityIndex< 0 >( 1 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 34 ).template getSubentityIndex< 0 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 34 ).template getSubentityIndex< 0 >( 1 ), 21 );


   ASSERT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 0 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 0 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 0 >( 4 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 1 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 1 >( 3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSubentityIndex< 1 >( 4 ),  4 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentitiesCount< 0 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 0 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 0 >( 2 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 0 >( 3 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentitiesCount< 1 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 1 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 1 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSubentityIndex< 1 >( 3 ),  7 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 0 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 0 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 0 >( 3 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 0 >( 4 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 1 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 1 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 1 >( 2 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 1 >( 3 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSubentityIndex< 1 >( 4 ), 10 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentitiesCount< 0 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 0 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 0 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 0 >( 2 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 0 >( 3 ), 10 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentitiesCount< 1 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 1 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 1 >( 1 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 1 >( 2 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSubentityIndex< 1 >( 3 ), 13 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 0 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 0 >( 2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 0 >( 3 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 0 >( 4 ),  9 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 1 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 1 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 1 >( 2 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 1 >( 3 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSubentityIndex< 1 >( 4 ), 15 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentitiesCount< 0 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 0 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 0 >( 2 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 0 >( 3 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentitiesCount< 1 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 1 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 1 >( 1 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 1 >( 2 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSubentityIndex< 1 >( 3 ), 17 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 0 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 0 >( 1 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 0 >( 2 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 0 >( 3 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 0 >( 4 ), 10 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 1 >( 0 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 1 >( 1 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 1 >( 2 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 1 >( 3 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSubentityIndex< 1 >( 4 ), 19 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 0 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 0 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 0 >( 2 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 0 >( 3 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 0 >( 4 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 1 >( 0 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 1 >( 1 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 1 >( 2 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 1 >( 3 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSubentityIndex< 1 >( 4 ), 20 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 0 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 0 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 0 >( 2 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 0 >( 3 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 0 >( 4 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 1 >( 0 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 1 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 1 >( 2 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 1 >( 3 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSubentityIndex< 1 >( 4 ), 18 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 0 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 0 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 0 >( 2 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 0 >( 3 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 0 >( 4 ), 16 );
   ASSERT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 1 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 1 >( 1 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 1 >( 2 ), 22 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 1 >( 3 ), 23 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSubentityIndex< 1 >( 4 ), 24 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 0 >( 0 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 0 >( 1 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 0 >( 2 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 0 >( 3 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 0 >( 4 ), 19 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 1 >( 0 ), 23 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 1 >( 1 ), 25 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 1 >( 2 ), 26 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 1 >( 3 ), 27 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSubentityIndex< 1 >( 4 ), 28 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentitiesCount< 0 >( ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 0 >( 0 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 0 >( 1 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 0 >( 2 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 0 >( 3 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 0 >( 4 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 0 >( 5 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentitiesCount< 1 >( ),  6 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 1 >( 0 ), 29 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 1 >( 1 ), 26 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 1 >( 2 ), 30 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 1 >( 3 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 1 >( 4 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSubentityIndex< 1 >( 5 ), 31 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentitiesCount< 0 >( ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentityIndex< 0 >( 0 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentityIndex< 0 >( 1 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentityIndex< 0 >( 2 ), 14 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentitiesCount< 1 >( ),  3 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentityIndex< 1 >( 0 ), 25 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentityIndex< 1 >( 1 ), 22 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 12 ).template getSubentityIndex< 1 >( 2 ), 30 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentitiesCount< 0 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 0 >( 0 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 0 >( 1 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 0 >( 2 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 0 >( 3 ), 20 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentitiesCount< 1 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 1 >( 0 ), 32 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 1 >( 1 ), 27 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 1 >( 2 ), 29 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSubentityIndex< 1 >( 3 ), 33 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentitiesCount< 0 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 0 >( 0 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 0 >( 1 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 0 >( 2 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 0 >( 3 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentitiesCount< 1 >( ),  4 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 1 >( 0 ), 33 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 1 >( 1 ), 31 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 1 >( 2 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSubentityIndex< 1 >( 3 ), 34 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentitiesCount< 0 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 0 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 0 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 0 >( 2 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 0 >( 3 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 0 >( 4 ), 21 );
   ASSERT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentitiesCount< 1 >( ),  5 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 1 >( 0 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 1 >( 1 ), 24 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 1 >( 2 ), 28 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 1 >( 3 ), 32 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSubentityIndex< 1 >( 4 ), 34 );


   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 0 >( ),  14 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  5 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  6 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  7 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  8 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >(  9 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >( 10 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >( 11 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >( 12 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 0 >( 13 ), 13 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 1 >( ),  21 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  5 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  6 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  7 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  8 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >(  9 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 10 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 11 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 12 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 13 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 14 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 15 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 16 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 17 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 18 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 19 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 1 >( 20 ), 20 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentitiesCount< 2 >( ),   9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  2 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  3 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  4 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  5 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  6 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  7 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 0 ).template getSubentityIndex< 2 >(  8 ),  8 );

   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 0 >( ),  12 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  2 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  3 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  4 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  5 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  6 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  7 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  8 ), 20 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >(  9 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >( 10 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 0 >( 11 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 1 >( ),  18 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  1 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  2 ), 22 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  3 ), 23 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  4 ), 24 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  5 ), 25 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  6 ), 26 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  7 ), 27 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  8 ), 28 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >(  9 ), 29 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 10 ), 30 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 11 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 12 ), 31 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 13 ), 32 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 14 ), 33 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 15 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 16 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 1 >( 17 ), 34 );
   ASSERT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentitiesCount< 2 >( ),   8 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  2 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  3 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  4 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  5 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  6 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 3 >( 1 ).template getSubentityIndex< 2 >(  7 ), 15 );


   // tests for the superentities layer
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 1 >( 2 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 2 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 1 >( 2 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 1 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 2 >( 2 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 1 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 1 >( 2 ),  5 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 2 >( 2 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 1 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 2 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 1 >( 2 ), 10 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 1 >( 2 ), 20 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  6 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 0 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 1 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 2 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 1 >( 3 ), 21 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 2 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 3 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 2 >( 4 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  7 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 1 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 1 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 1 >( 2 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 1 >( 3 ), 24 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 2 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 2 >( 3 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 2 >( 4 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  8 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 1 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 1 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 1 >( 2 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 2 >( 2 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >(  9 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 1 >( 0 ), 12 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 1 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 1 >( 2 ), 19 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 2 >( 2 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 10 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 1 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 1 >( 1 ), 15 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 1 >( 2 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 1 >( 3 ), 31 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 2 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 2 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 2 >( 2 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 2 >( 3 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 2 >( 4 ), 14 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 11 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentitiesCount< 1 >(   ),  4 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 1 >( 0 ), 16 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 1 >( 1 ), 17 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 1 >( 2 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 1 >( 3 ), 34 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentitiesCount< 2 >(   ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 2 >( 3 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 2 >( 4 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 12 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 1 >( 0 ), 18 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 1 >( 1 ), 19 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 1 >( 2 ), 20 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 2 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 2 >( 2 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 13 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 1 >( 0 ), 21 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 1 >( 1 ), 22 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 1 >( 2 ), 30 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 2 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 2 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 2 >( 2 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 14 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 1 >( 0 ), 22 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 1 >( 1 ), 23 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 1 >( 2 ), 25 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 2 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 2 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 2 >( 2 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 15 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentityIndex  < 1 >( 0 ), 23 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentityIndex  < 1 >( 1 ), 24 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentityIndex  < 1 >( 2 ), 28 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentityIndex  < 2 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentityIndex  < 2 >( 1 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentityIndex  < 2 >( 2 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 16 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentityIndex  < 1 >( 0 ), 25 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentityIndex  < 1 >( 1 ), 26 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentityIndex  < 1 >( 2 ), 30 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentityIndex  < 2 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentityIndex  < 2 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentityIndex  < 2 >( 2 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 17 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentityIndex  < 1 >( 0 ), 26 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentityIndex  < 1 >( 1 ), 27 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentityIndex  < 1 >( 2 ), 29 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentityIndex  < 2 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentityIndex  < 2 >( 1 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentityIndex  < 2 >( 2 ), 13 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 18 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentityIndex  < 1 >( 0 ), 27 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentityIndex  < 1 >( 1 ), 28 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentityIndex  < 1 >( 2 ), 32 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentityIndex  < 2 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentityIndex  < 2 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentityIndex  < 2 >( 2 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 19 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentityIndex  < 1 >( 0 ), 29 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentityIndex  < 1 >( 1 ), 31 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentityIndex  < 1 >( 2 ), 33 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentityIndex  < 2 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentityIndex  < 2 >( 1 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentityIndex  < 2 >( 2 ), 14 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 20 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentitiesCount< 1 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentityIndex  < 1 >( 0 ), 32 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentityIndex  < 1 >( 1 ), 33 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentityIndex  < 1 >( 2 ), 34 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentityIndex  < 2 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentityIndex  < 2 >( 1 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentityIndex  < 2 >( 2 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 0 >( 21 ).template getSuperentityIndex  < 3 >( 0 ),  1 );


   ASSERT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 2 >( 1 ),  3 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 2 >( 1 ),  1 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 2 >( 1 ),  2 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  6 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 0 ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  7 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  8 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 2 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 2 >( 2 ),  9 );
   ASSERT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >(  9 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 2 >( 0 ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 10 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentityIndex  < 2 >( 1 ),  4 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 11 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 12 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentityIndex  < 2 >( 0 ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 13 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentityIndex  < 2 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentityIndex  < 2 >( 1 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentityIndex  < 2 >( 2 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 14 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentityIndex  < 2 >( 0 ),  4 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 15 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentityIndex  < 2 >( 1 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentityIndex  < 2 >( 2 ), 14 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 16 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentitiesCount< 2 >(   ),  3 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentityIndex  < 2 >( 0 ),  5 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentityIndex  < 2 >( 2 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 17 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 18 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSuperentityIndex  < 2 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 18 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 18 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 19 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSuperentityIndex  < 2 >( 0 ),  6 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSuperentityIndex  < 2 >( 1 ),  7 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 19 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 19 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 20 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSuperentityIndex  < 2 >( 0 ),  7 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSuperentityIndex  < 2 >( 1 ),  8 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 20 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 20 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 21 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSuperentityIndex  < 2 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSuperentityIndex  < 2 >( 1 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 21 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 21 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 22 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 22 ).template getSuperentityIndex  < 2 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 22 ).template getSuperentityIndex  < 2 >( 1 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 22 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 22 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 23 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 23 ).template getSuperentityIndex  < 2 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 23 ).template getSuperentityIndex  < 2 >( 1 ), 10 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 23 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 23 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 24 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 24 ).template getSuperentityIndex  < 2 >( 0 ),  9 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 24 ).template getSuperentityIndex  < 2 >( 1 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 24 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 24 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 25 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 25 ).template getSuperentityIndex  < 2 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 25 ).template getSuperentityIndex  < 2 >( 1 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 25 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 25 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 26 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 26 ).template getSuperentityIndex  < 2 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 26 ).template getSuperentityIndex  < 2 >( 1 ), 11 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 26 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 26 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 27 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 27 ).template getSuperentityIndex  < 2 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 27 ).template getSuperentityIndex  < 2 >( 1 ), 13 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 27 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 27 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 28 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 28 ).template getSuperentityIndex  < 2 >( 0 ), 10 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 28 ).template getSuperentityIndex  < 2 >( 1 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 28 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 28 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 29 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 29 ).template getSuperentityIndex  < 2 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 29 ).template getSuperentityIndex  < 2 >( 1 ), 13 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 29 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 29 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 30 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 30 ).template getSuperentityIndex  < 2 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 30 ).template getSuperentityIndex  < 2 >( 1 ), 12 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 30 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 30 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 31 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 31 ).template getSuperentityIndex  < 2 >( 0 ), 11 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 31 ).template getSuperentityIndex  < 2 >( 1 ), 14 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 31 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 31 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 32 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 32 ).template getSuperentityIndex  < 2 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 32 ).template getSuperentityIndex  < 2 >( 1 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 32 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 32 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 33 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 33 ).template getSuperentityIndex  < 2 >( 0 ), 13 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 33 ).template getSuperentityIndex  < 2 >( 1 ), 14 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 33 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 33 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 1 >( 34 ).template getSuperentitiesCount< 2 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 34 ).template getSuperentityIndex  < 2 >( 0 ), 14 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 34 ).template getSuperentityIndex  < 2 >( 1 ), 15 );
   ASSERT_EQ( mesh.template getEntity< 1 >( 34 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 1 >( 34 ).template getSuperentityIndex  < 3 >( 0 ),  1 );


   ASSERT_EQ( mesh.template getEntity< 2 >(  0 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  0 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  1 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  1 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  2 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  2 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  3 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  3 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  4 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  4 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  5 ).template getSuperentitiesCount< 3 >(   ),  2 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSuperentityIndex  < 3 >( 0 ),  0 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  5 ).template getSuperentityIndex  < 3 >( 1 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  6 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  6 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  7 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  7 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  8 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  8 ).template getSuperentityIndex  < 3 >( 0 ),  0 );

   ASSERT_EQ( mesh.template getEntity< 2 >(  9 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >(  9 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 10 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 10 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 11 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 11 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 12 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 12 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 13 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 13 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 14 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 14 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   ASSERT_EQ( mesh.template getEntity< 2 >( 15 ).template getSuperentitiesCount< 3 >(   ),  1 );
   EXPECT_EQ( mesh.template getEntity< 2 >( 15 ).template getSuperentityIndex  < 3 >( 0 ),  1 );

   // tests for the dual graph layer
   ASSERT_EQ( mesh.getNeighborCounts().getSize(), 2 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 0    ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 0, 0 ), 1 );

   ASSERT_EQ( mesh.getCellNeighborsCount( 1    ), 1 );
   EXPECT_EQ( mesh.getCellNeighborIndex ( 1, 0 ), 0 );

   testFinishedMesh( mesh );
}

} // namespace MeshTest

#endif
