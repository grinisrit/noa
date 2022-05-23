#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include "data/loader.h"
#include "MeshReaderTest.h"

using namespace TNL::Meshes;

static const char* TEST_FILE_NAME = "test_VTKReaderTest.vtk";

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

// TODO: test case for 1D mesh of edges

// ASCII data, produced by Gmsh
TEST( VTKReaderTest, mrizka_1 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles/mrizka_1.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 142 );
   EXPECT_EQ( cells, 242 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by Gmsh
TEST( VTKReaderTest, tetrahedrons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Tetrahedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "tetrahedrons/cube1m_1.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 395 );
   EXPECT_EQ( cells, 1312 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// binary data, produced by RF writer
TEST( VTKReaderTest, triangles_2x2x2_original_with_metadata_and_cell_data )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/original_with_metadata_and_cell_data.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTKReaderTest, triangles_2x2x2_minimized_ascii )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/minimized_ascii.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// binary data, produced by TNL writer
TEST( VTKReaderTest, triangles_2x2x2_minimized_binary )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/minimized_binary.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by Paraview (DataFile version 5.1)
TEST( VTKReaderTest, triangles_2x2x2_ascii_51 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/version_5.1_ascii.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// binary data, produced by Paraview (DataFile version 5.1)
TEST( VTKReaderTest, triangles_2x2x2_binary_51 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "triangles_2x2x2/version_5.1_binary.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 9 );
   EXPECT_EQ( cells, 8 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}


// binary data, produced by TNL writer
TEST( VTKReaderTest, quadrangles )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Quadrangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "quadrangles/grid_2x3.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 12 );
   EXPECT_EQ( cells, 6 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// binary data, produced by TNL writer
TEST( VTKReaderTest, hexahedrons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Hexahedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "hexahedrons/grid_2x3x4.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 60 );
   EXPECT_EQ( cells, 24 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTKReaderTest, polygons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polygon > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "polygons/unicorn.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 193 );
   EXPECT_EQ( cells, 90 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by Paraview
TEST( VTKReaderTest, two_polyhedra )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polyhedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "polyhedrons/two_polyhedra.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto faces = mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 22 );
   EXPECT_EQ( faces, 16 );
   EXPECT_EQ( cells, 2 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// binary data, produced by Paraview
TEST( VTKReaderTest, cube1m_1 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polyhedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::VTKReader >( "polyhedrons/cube1m_1.vtk" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto faces = mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 2358 );
   EXPECT_EQ( faces, 2690 );
   EXPECT_EQ( cells, 395 );

   test_reader< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTKWriter, MyConfigTag >( mesh, TEST_FILE_NAME, "int" );  // force GlobalIndex to int (VTK DataFormat 2.0 uses int32, but 5.1 uses int64)
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTKReader, Writers::VTKWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

#endif

#include "../main.h"
