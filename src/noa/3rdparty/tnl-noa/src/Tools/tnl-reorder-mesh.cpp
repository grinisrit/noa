#include <filesystem>

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/MeshOrdering.h>
#include <TNL/Meshes/MeshOrdering/CuthillMcKeeOrdering.h>
#include <TNL/Meshes/MeshOrdering/KdTreeOrdering.h>
#include <TNL/Meshes/MeshOrdering/HilbertOrdering.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
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

template< typename Mesh >
bool
reorder( Mesh&& mesh, const std::string& ordering, const std::string& outputFileName, std::string outputFileFormat )
{
   if( ordering == "rcm" ) {
      // reverse Cuthill-McKee ordering
      MeshOrdering< Mesh, CuthillMcKeeOrdering<> > rcm_ordering;
      rcm_ordering.reorder( mesh );
   }
   else if( ordering == "kdtree" ) {
#ifdef HAVE_CGAL
      // k-d tree ordering
      MeshOrdering< Mesh, KdTreeOrdering > kdtree_ordering;
      kdtree_ordering.reorder( mesh );
#else
      std::cerr << "CGAL support is missing. Recompile with -DHAVE_CGAL and try again." << std::endl;
      return false;
#endif
   }
   else if( ordering == "hilbert" ) {
#ifdef HAVE_CGAL
      // Hilbert curve ordering
      MeshOrdering< Mesh, HilbertOrdering > hilbert_ordering;
      hilbert_ordering.reorder( mesh );
#else
      std::cerr << "CGAL support is missing. Recompile with -DHAVE_CGAL and try again." << std::endl;
      return false;
#endif
   }
   else {
      std::cerr << "unknown ordering algorithm: '" << ordering << "'. Available options are: rcm, kdtree, hilbert."
                << std::endl;
      return false;
   }

   namespace fs = std::filesystem;
   if( outputFileFormat == "auto" ) {
      outputFileFormat = fs::path( outputFileName ).extension().string();
      if( outputFileFormat.length() > 0 )
         // remove dot from the extension
         outputFileFormat = outputFileFormat.substr( 1 );
   }

   if( outputFileFormat == "vtk" ) {
      std::ofstream file( outputFileName );
      using MeshWriter = TNL::Meshes::Writers::VTKWriter< Mesh >;
      MeshWriter writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
   }
   else if( outputFileFormat == "vtu" ) {
      std::ofstream file( outputFileName );
      using MeshWriter = TNL::Meshes::Writers::VTUWriter< Mesh >;
      MeshWriter writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
   }
   else {
      std::cerr << "unknown output file format: '" << outputFileFormat << "'. Available options are: vtk, vtu." << std::endl;
      return false;
   }

   return true;
}

void
configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-mesh", "Input file with the distributed mesh." );
   config.addEntry< String >( "input-mesh-format", "Input mesh file format.", "auto" );
   config.addRequiredEntry< String >( "ordering", "Algorithm to reorder the mesh." );
   config.addEntryEnum( "rcm" );
   config.addEntryEnum( "kdtree" );
   config.addRequiredEntry< String >( "output-mesh", "Output file with the distributed mesh." );
   config.addEntry< String >( "output-mesh-format", "Output mesh file format.", "auto" );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String inputFileName = parameters.getParameter< String >( "input-mesh" );
   const String inputFileFormat = parameters.getParameter< String >( "input-mesh-format" );
   const String ordering = parameters.getParameter< String >( "ordering" );
   const String outputFileName = parameters.getParameter< String >( "output-mesh" );
   const String outputFileFormat = parameters.getParameter< String >( "output-mesh-format" );

   auto wrapper = [ & ]( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype( mesh ) >;
      return reorder( std::forward< MeshType >( mesh ), ordering, outputFileName, outputFileFormat );
   };
   const bool status = resolveAndLoadMesh< MyConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
   return static_cast< int >( ! status );
}
