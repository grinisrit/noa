#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>
#include <TNL/Meshes/Geometry/getEntityCircumradius.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

using namespace TNL;
using namespace TNL::Meshes;

struct MyConfigTag
{};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< MyConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MyConfigTag, double > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MyConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Unstructured meshes.
 */
//template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Edge >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Hexahedron >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polyhedron >{ static constexpr bool enabled = true; };

// Meshes are enabled only for the world dimension equal to the cell dimension.
template< typename CellTopology, int WorldDimension >
struct MeshSpaceDimensionTag< MyConfigTag, CellTopology, WorldDimension >
{ static constexpr bool enabled = WorldDimension == CellTopology::dimension; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MyConfigTag, float >{ static constexpr bool enabled = true; };
template<> struct MeshRealTag< MyConfigTag, double >{ static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MyConfigTag, int >{ static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MyConfigTag, long int >{ static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< MyConfigTag, short int >{ static constexpr bool enabled = true; };

}  // namespace BuildConfigTags
}  // namespace Meshes
}  // namespace TNL

template< typename MeshConfig >
bool
printInfo( Mesh< MeshConfig, Devices::Host >& mesh, const std::string& fileName )
{
   using MeshType = Mesh< MeshConfig, Devices::Host >;
   using CellType = typename MeshType::Cell;
   using RealType = typename MeshType::RealType;
   using GlobalIndexType = typename MeshType::GlobalIndexType;
   using VectorType = TNL::Containers::Vector< RealType, TNL::Devices::Host, GlobalIndexType >;

   const auto verticesCount = mesh.template getEntitiesCount< 0 >();
   const auto facesCount = mesh.template getEntitiesCount< MeshType::getMeshDimension() - 1 >();
   const auto cellsCount = mesh.template getEntitiesCount< MeshType::getMeshDimension() >();

   VectorType diameters( cellsCount );
   VectorType circumradii( cellsCount );
   VectorType cellSubvertices( cellsCount );
   VectorType faceSubvertices( facesCount );
   VectorType cellSubfaces( cellsCount );

   mesh.template forAll< MeshType::getMeshDimension() >(
      [&] (GlobalIndexType i) {
         const auto cell = mesh.template getEntity< MeshType::getMeshDimension() >( i );
         if( MeshType::getMeshDimension() == 3 )
            diameters[ i ] = std::cbrt( getEntityMeasure( mesh, cell ) * 6 / 3.1415926535897932384626433 );
         else
            diameters[ i ] = std::sqrt( getEntityMeasure( mesh, cell ) * 4 / 3.1415926535897932384626433 );
         circumradii[ i ] = getEntityCircumradius( mesh, cell );

         cellSubvertices[ i ] = cell.template getSubentitiesCount< 0 >();
         cellSubfaces[ i ] = cell.template getSubentitiesCount< MeshType::getMeshDimension() - 1 >();
      } );

   mesh.template forAll< MeshType::getMeshDimension() - 1 >(
      [&] (GlobalIndexType i) {
         const auto face = mesh.template getEntity< MeshType::getMeshDimension() - 1 >( i );
         faceSubvertices[ i ] = face.template getSubentitiesCount< 0 >();
      } );

   const double avgCellSubvertices = TNL::sum( cellSubvertices ) / cellsCount;
   const double avgFaceSubvertices = TNL::sum( faceSubvertices ) / facesCount;
   const double avgSubfaces = TNL::sum( cellSubfaces ) / cellsCount;

   std::cout << fileName << ":\n"
             << "\tMesh dimension:\t" << MeshType::getMeshDimension() << "\n"
             << "\tCell topology:\t" << getType( typename CellType::EntityTopology{} ) << "\n"
             << "\tVertices count:\t" << verticesCount << "\n"
             << "\tFaces count:\t" << facesCount << "\n"
             << "\tCells count:\t" << cellsCount << "\n"
             << "\tLargest cell circumradius:\t" << TNL::max( circumradii ) << "\n"
             << "\tSmallest cell circumradius:\t" << TNL::min( circumradii ) << "\n"
             << "\tDiameter of a ball with the same volume as the largest cell:\t" << TNL::max( diameters ) << "\n"
             << "\tDiameter of a ball with the same volume as the smallest cell:\t" << TNL::min( diameters ) << "\n"
             << "\tAverage cell diameter:\t" << TNL::sum( diameters ) / cellsCount << "\n"
             << "\tAverage number of subvertices per cell:\t" << avgCellSubvertices << "\n"
             << "\tAverage number of subvertices per face:\t" << avgFaceSubvertices << "\n"
             << "\tAverage number of faces per cell:\t" << avgSubfaces << "\n"
             << std::endl;

   return true;
}

int
main( int argc, char* argv[] )
{
   if( argc < 2 ) {
      std::cerr << "Usage: " << argv[ 0 ] << " filename.[tnl|ng|vtk|vtu|fpma] ..." << std::endl;
      return EXIT_FAILURE;
   }

   bool result = true;

   for( int i = 1; i < argc; i++ ) {
      const std::string fileName = argv[ i ];
      auto wrapper = [&]( auto& reader, auto&& mesh ) -> bool
      {
         return printInfo( mesh, fileName );
      };
      result &= resolveAndLoadMesh< MyConfigTag, Devices::Host >( wrapper, fileName );
   }

   return static_cast< int >( ! result );
}
