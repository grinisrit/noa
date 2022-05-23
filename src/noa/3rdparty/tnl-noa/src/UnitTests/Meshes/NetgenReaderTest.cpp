#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Readers/NetgenReader.h>
#include <TNL/Meshes/Writers/NetgenWriter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include "data/loader.h"

using namespace TNL::Meshes;

static const char* TEST_FILE_NAME = "test_NetgenReaderTest.ng";

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
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

template< typename MeshType >
void test_NetgenReader( const MeshType& mesh )
{
   // write the mesh into the file (new scope is needed to properly close the file)
   {
      std::ofstream file( TEST_FILE_NAME );
      Writers::NetgenWriter< MeshType > writer;
      writer.writeMesh( mesh, file );
   }

   MeshType mesh_in;
   Readers::NetgenReader reader( TEST_FILE_NAME );
   reader.loadMesh( mesh_in );

   EXPECT_EQ( mesh_in, mesh );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

// Test that:
// 1. resolveMeshType resolves the mesh type correctly
// 2. resolveAndLoadMesh loads the mesh
template< typename ConfigTag, typename MeshType >
void test_resolveAndLoadMesh( const MeshType& mesh )
{
   // write the mesh into the file (new scope is needed to properly close the file)
   {
      std::ofstream file( TEST_FILE_NAME );
      Writers::NetgenWriter< MeshType > writer;
      writer.writeMesh( mesh, file );
   }

   auto wrapper = [&] ( Readers::MeshReader& reader, auto&& mesh2 )
   {
      using MeshType2 = std::decay_t< decltype(mesh2) >;

      // static_assert does not work, the wrapper is actually instantiated for all resolved types
//      static_assert( std::is_same< MeshType2, MeshType >::value, "mesh type was not resolved as expected" );
      EXPECT_EQ( std::string( TNL::getType< MeshType2 >() ), std::string( TNL::getType< MeshType >() ) );

      // operator== does not work for instantiations of the wrapper with MeshType2 != MeshType
//      EXPECT_EQ( mesh2, mesh );
      std::stringstream str1, str2;
      str1 << mesh;
      str2 << mesh2;
      EXPECT_EQ( str2.str(), str1.str() );

      return true;
   };

   const bool status = resolveAndLoadMesh< ConfigTag, TNL::Devices::Host >( wrapper, TEST_FILE_NAME );
   EXPECT_TRUE( status );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

// TODO: test case for 1D mesh of edges

TEST( NetgenReaderTest, triangles )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Triangle > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::NetgenReader >( "triangles/netgen_square.ng" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 289 );
   EXPECT_EQ( cells, 512 );

   test_NetgenReader( mesh );
   test_resolveAndLoadMesh< MyConfigTag >( mesh );
}

TEST( NetgenReaderTest, tetrahedrons )
{
   using MeshType = Mesh< DefaultConfig< Topologies::Tetrahedron > >;
   const MeshType mesh = loadMeshFromFile< MeshType, Readers::NetgenReader >( "tetrahedrons/netgen_cube.ng" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();
   EXPECT_EQ( vertices, 4913 );
   EXPECT_EQ( cells, 24576 );

   test_NetgenReader( mesh );
   test_resolveAndLoadMesh< MyConfigTag >( mesh );
}
#endif

#include "../main.h"
