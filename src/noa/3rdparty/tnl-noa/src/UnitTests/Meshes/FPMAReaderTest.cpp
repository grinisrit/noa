#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Readers/FPMAReader.h>
#include <TNL/Meshes/Writers/FPMAWriter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include "data/loader.h"
#include "MeshReaderTest.h"

using namespace TNL::Meshes;

static const char* TEST_FILE_NAME = "test_FPMAReaderTest.fpma";

struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// disable all grids
template< int Dimension, typename Real, typename Device, typename Index >
struct GridTag< MyConfigTag, Grid< Dimension, Real, Device, Index > >{ static constexpr bool enabled = false; };

// enable meshes used in the tests
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polyhedron > { static constexpr bool enabled = true; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

TEST( FPMAReaderTest, two_polyhedra )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polyhedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::FPMAReader >( "polyhedrons/two_polyhedra.fpma" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto faces = mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 22 );
   EXPECT_EQ( faces, 16 );
   EXPECT_EQ( cells, 2 );

   test_reader< Readers::FPMAReader, Writers::FPMAWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::FPMAWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
}

TEST( FPMAReaderTest, cube1m_1 )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Polyhedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::FPMAReader >( "polyhedrons/cube1m_1.fpma" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto faces = mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 2358 );
   EXPECT_EQ( faces, 2690 );
   EXPECT_EQ( cells, 395 );

   test_reader< Readers::FPMAReader, Writers::FPMAWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::FPMAWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
}

#endif

#include "../main.h"
