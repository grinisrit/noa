#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshOrdering.h>
#include <TNL/Meshes/MeshOrdering/CuthillMcKeeOrdering.h>
#include <TNL/Meshes/MeshOrdering/KdTreeOrdering.h>
#include <TNL/Meshes/MeshOrdering/HilbertOrdering.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTUWriter.h>

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
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Edge >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Triangle >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Quadrangle >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Tetrahedron >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Hexahedron >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Wedge >{ static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MyConfigTag, Topologies::Pyramid >{ static constexpr bool enabled = true; };
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

template< typename MeshEntity, typename MeshType >
void
writeMeshOrdering( const MeshType& mesh, const String& outputFileName )
{
   std::ofstream outputFile( outputFileName.getString() );

   for( typename MeshType::GlobalIndexType i = 0; i < mesh.template getEntitiesCount< MeshEntity >(); i++ ) {
      const auto cell = mesh.template getEntity< MeshEntity >( i );
      const auto center = getEntityCenter( mesh, cell );
      for( int j = 0; j < center.getSize(); j++ )
         outputFile << center[ j ] << " ";
      outputFile << std::endl;
   }
}

template< typename Mesh >
void
writeMeshVTU( const Mesh& mesh, const std::string& fileName )
{
   std::ofstream file( fileName );
   using MeshWriter = TNL::Meshes::Writers::VTUWriter< Mesh >;
   MeshWriter writer( file );
   writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
}

template< typename MeshConfig >
bool
plotOrdering( Mesh< MeshConfig, Devices::Host >& mesh, const String& fileName )
{
   using MeshType = Mesh< MeshConfig, Devices::Host >;
   using MeshEntity = typename MeshType::Cell;

   // strip the ".tnl" or ".vtk" or ".vtu" suffix
   const String baseName( fileName.getString(), 0, fileName.getSize() - 4 );

   writeMeshOrdering< MeshEntity >( mesh, baseName + ".original.gplt" );

   // reverse Cuthill-McKee ordering
   MeshOrdering< MeshType, CuthillMcKeeOrdering<> > rcm_ordering;
   rcm_ordering.reorder( mesh );
   writeMeshOrdering< MeshEntity >( mesh, baseName + ".rcm.gplt" );
   writeMeshVTU( mesh, baseName + ".rcm.vtu" );

#ifdef HAVE_CGAL
   // k-d tree ordering
   MeshOrdering< MeshType, KdTreeOrdering > kdtree_ordering;
   kdtree_ordering.reorder( mesh );
   writeMeshOrdering< MeshEntity >( mesh, baseName + ".kdtree.gplt" );
   writeMeshVTU( mesh, baseName + ".kdtree.vtu" );
#else
   std::cerr << "CGAL support is missing. Skipping KdTreeOrdering." << std::endl;
#endif

#ifdef HAVE_CGAL
   // Hilbert curve ordering
   MeshOrdering< MeshType, HilbertOrdering > hilbert_ordering;
   hilbert_ordering.reorder( mesh );
   writeMeshOrdering< MeshEntity >( mesh, baseName + ".hilbert.gplt" );
   writeMeshVTU( mesh, baseName + ".hilbert.vtu" );
#else
   std::cerr << "CGAL support is missing. Skipping HilbertOrdering." << std::endl;
#endif

   return true;
}

int
main( int argc, char* argv[] )
{
   if( argc < 2 ) {
      std::cerr << "Usage: " << argv[ 0 ] << " filename.[tnl|ng|vtk|vtu] ..." << std::endl;
      return EXIT_FAILURE;
   }

   bool result = true;

   for( int i = 1; i < argc; i++ ) {
      String fileName = argv[ i ];
      auto wrapper = [ & ]( auto& reader, auto&& mesh ) -> bool
      {
         return plotOrdering( mesh, fileName );
      };
      result &= resolveAndLoadMesh< MyConfigTag, Devices::Host >( wrapper, fileName );
   }

   return static_cast< int >( ! result );
}
