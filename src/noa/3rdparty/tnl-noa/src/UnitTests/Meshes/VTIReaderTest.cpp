
#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Meshes/Readers/VTIReader.h>
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include "data/loader.h"
#include "MeshReaderTest.h"

using namespace TNL::Meshes;

static const char* TEST_FILE_NAME = "test_VTIReaderTest.vti";

struct MyConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// enable all index types in the GridTypeResolver
template<> struct GridIndexTag< MyConfigTag, short int >{ static constexpr bool enabled = true; };
template<> struct GridIndexTag< MyConfigTag, int >{ static constexpr bool enabled = true; };
// GOTCHA: the tests work only for integer types that have a fixed-width alias
// (long int may be a 32-bit type, but different from int (e.g. on Windows), which would make some tests fail)
template<> struct GridIndexTag< MyConfigTag, std::int64_t >{ static constexpr bool enabled = true; };

// disable float and long double (RealType is not stored in VTI and double is the default)
template<> struct GridRealTag< MyConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MyConfigTag, long double > { static constexpr bool enabled = false; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

TEST( VTIReaderTest, Grid1D )
{
   using GridType = Grid< 1, double, TNL::Devices::Host, short int >;
   using PointType = GridType::PointType;
   using CoordinatesType = GridType::CoordinatesType;

   GridType grid;
   grid.setDomain( PointType( 1 ), PointType( 2 ) );
   grid.setDimensions( CoordinatesType( 10 ) );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( grid, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "CellData" );
}

TEST( VTIReaderTest, Grid2D )
{
   using GridType = Grid< 2, double, TNL::Devices::Host, int >;
   using PointType = GridType::PointType;
   using CoordinatesType = GridType::CoordinatesType;

   GridType grid;
   grid.setDomain( PointType( 1, 2 ), PointType( 3, 4 ) );
   grid.setDimensions( CoordinatesType( 10, 20 ) );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( grid, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "CellData" );
}

TEST( VTIReaderTest, Grid3D )
{
   using GridType = Grid< 3, double, TNL::Devices::Host, std::int64_t >;
   using PointType = GridType::PointType;
   using CoordinatesType = GridType::CoordinatesType;

   GridType grid;
   grid.setDomain( PointType( 1, 2, 3 ), PointType( 4, 5, 6 ) );
   grid.setDimensions( CoordinatesType( 10, 20, 30 ) );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( grid, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( grid, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTIReaderTest, Grid2D_vti )
{
   using GridType = Grid< 2, double, TNL::Devices::Host, int >;
   const GridType mesh = loadMeshFromFile< GridType, Readers::VTIReader >( "quadrangles/grid_2x3.vti" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< GridType::getMeshDimension() >();
   EXPECT_EQ( vertices, 12 );
   EXPECT_EQ( cells, 6 );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( mesh, TEST_FILE_NAME, "CellData" );
}

// ASCII data, produced by TNL writer
TEST( VTIReaderTest, Grid3D_vti )
{
   using GridType = Grid< 3, double, TNL::Devices::Host, std::int64_t >;
   const GridType mesh = loadMeshFromFile< GridType, Readers::VTIReader >( "hexahedrons/grid_2x3x4.vti" );

   // test that the mesh was actually loaded
   const auto vertices = mesh.template getEntitiesCount< 0 >();
   const auto cells = mesh.template getEntitiesCount< GridType::getMeshDimension() >();
   EXPECT_EQ( vertices, 60 );
   EXPECT_EQ( cells, 24 );

   test_reader< Readers::VTIReader, Writers::VTIWriter >( mesh, TEST_FILE_NAME );
   test_resolveAndLoadMesh< Writers::VTIWriter, MyConfigTag >( mesh, TEST_FILE_NAME );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( mesh, TEST_FILE_NAME, "PointData" );
   test_meshfunction< Readers::VTIReader, Writers::VTIWriter >( mesh, TEST_FILE_NAME, "CellData" );
}
#endif

#include "../main.h"
