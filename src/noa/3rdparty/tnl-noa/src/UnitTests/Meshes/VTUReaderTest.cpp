#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Readers/VTUReader.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include "data/loader.h"
#include "MeshReaderTest.h"

using namespace TNL::Meshes;

static const char* TEST_FILE_NAME = "test_VTUReaderTest.vtu";

struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >{ static constexpr bool enabled = false; };

// enable meshes used in the tests
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Edge > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Hexahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polyhedron > { static constexpr bool enabled = true; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

TEST( VTUReaderTest, empty )
{
   // the cell topology does not matter for an empty mesh
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "empty.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 0 );
   EXPECT_EQ( cells, 0 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   // resolveAndLoadMesh cannot be tested since the empty mesh has Topologies::Vertex as cell topology
//   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// TODO: test case for 1D mesh of edges

TEST( VTUReaderTest, mrizka_1 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "triangles/mrizka_1.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 142 );
   EXPECT_EQ( cells, 242 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

TEST( VTUReaderTest, tetrahedrons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Tetrahedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "tetrahedrons/cube1m_1.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 395 );
   EXPECT_EQ( cells, 1312 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTUReaderTest, triangles_2x2x2_minimized_ascii )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "triangles_2x2x2/minimized_ascii.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// encoded data, produced by TNL writer
TEST( VTUReaderTest, triangles_2x2x2_minimized_encoded_tnl )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "triangles_2x2x2/minimized_encoded_tnl.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// encoded data, produced by Paraview
TEST( VTUReaderTest, triangles_2x2x2_minimized_encoded_paraview )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "triangles_2x2x2/minimized_encoded_paraview.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// compressed data, produced by TNL
TEST( VTUReaderTest, triangles_2x2x2_minimized_compressed_tnl )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "triangles_2x2x2/minimized_compressed_tnl.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// compressed data, produced by Paraview
TEST( VTUReaderTest, triangles_2x2x2_minimized_compressed_paraview )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "triangles_2x2x2/minimized_compressed_paraview.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// encoded data, produced by Paraview
TEST( VTUReaderTest, quadrangles )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Quadrangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "quadrangles/grid_2x3.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 12 );
   EXPECT_EQ( cells, 6 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// encoded data, produced by Paraview
TEST( VTUReaderTest, hexahedrons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Hexahedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "hexahedrons/grid_2x3x4.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 60 );
   EXPECT_EQ( cells, 24 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTUReaderTest, polygons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polygon > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "polygons/unicorn.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 193 );
   EXPECT_EQ( cells, 90 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, hand-converted from the FPMA format
TEST( VTUReaderTest, two_polyhedra )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polyhedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "polyhedrons/two_polyhedra.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto faces = mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 22 );
   EXPECT_EQ( faces, 16 );
   EXPECT_EQ( cells, 2 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// compressed data, produced by TNL writer
TEST( VTUReaderTest, cube1m_1 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polyhedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTUReader >( "polyhedrons/cube1m_1.vtu" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto faces = mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 2358 );
   EXPECT_EQ( faces, 2690 );
   EXPECT_EQ( cells, 395 );

   test_reader< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTUWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTUReader, Writers::VTUWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// TODO: test cases for the appended data block: minimized_appended_binary_compressed.vtu, minimized_appended_binary.vtu, minimized_appended_encoded_compressed.vtu, minimized_appended_encoded.vtu
// TODO: test case for mixed 3D mesh: data/polyhedrons/hexahedron_and_two_polyhedra.vtu
#endif

#include "../main.h"
