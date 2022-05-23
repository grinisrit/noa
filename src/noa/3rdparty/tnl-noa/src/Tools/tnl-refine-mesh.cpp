#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Geometry/getRefinedMesh.h>

using namespace TNL;

struct MeshRefineConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< MeshRefineConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MeshRefineConfigTag, double > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MeshRefineConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< MeshRefineConfigTag, Topologies::Triangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshRefineConfigTag, Topologies::Quadrangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshRefineConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshRefineConfigTag, Topologies::Hexahedron > { static constexpr bool enabled = true; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MeshRefineConfigTag, CellTopology, SpaceDimension >
{ static constexpr bool enabled = SpaceDimension == CellTopology::dimension; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MeshRefineConfigTag, float > { static constexpr bool enabled = true; };
template<> struct MeshRealTag< MeshRefineConfigTag, double > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshRefineConfigTag, long int > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshRefineConfigTag, int > { static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< MeshRefineConfigTag, short int > { static constexpr bool enabled = true; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MeshRefineConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = float,
             typename GlobalIndex = int,
             typename LocalIndex = short int >
   struct MeshConfig : public DefaultConfig< Cell, SpaceDimension, Real, GlobalIndex, LocalIndex >
   {
      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         return subentityDimension == 0 && entityDimension == Cell::dimension;
      }

      static constexpr bool superentityStorage( int entityDimension, int superentityDimension )
      {
         return false;
      }

      static constexpr bool entityTagsStorage( int entityDimension )
      {
         return false;
      }

      static constexpr bool dualGraphStorage()
      {
         return false;
      }
   };
};

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

template< typename Mesh >
Mesh getRefinedMeshHelper( const Mesh& mesh, const std::string& decompositionType )
{
   using namespace TNL::Meshes;
   return getRefinedMesh< EntityRefinerVersion::EdgeBisection >( mesh );
}

template< typename Mesh >
bool refineMesh( Mesh& mesh, const std::string& outputFileName, const std::string& outputFormat, const std::string& decompositionType, int iterations )
{
   for( int i = 1; i <= iterations; i++ ) {
      std::cout << "Refining mesh (iteration " << i << ")" << std::endl;
      mesh = getRefinedMeshHelper( mesh, decompositionType );
   }

   std::string format = outputFormat;
   if( outputFormat == "auto" ) {
      namespace fs = std::experimental::filesystem;
      format = fs::path( outputFileName ).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   if( format == "vtk" ) {
      using Writer = Meshes::Writers::VTKWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   if( format == "vtu" ) {
      using Writer = Meshes::Writers::VTUWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }

   if( outputFormat == "auto" )
      std::cerr << "File '" << outputFileName << "' has unsupported format (based on the file extension): " << format << ".";
   else
      std::cerr << "Unsupported output file format: " << outputFormat << ".";
   std::cerr << " Supported formats are 'vtk' and 'vtu'." << std::endl;
   return false;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< std::string >( "input-file", "Input file with the mesh." );
   config.addEntry< std::string >( "input-file-format", "Input mesh file format.", "auto" );
   config.addEntry< std::string >( "real-type", "Type to use for the representation of spatial coordinates in the output mesh. When 'auto', the real type from the input mesh is used.", "auto" );
   config.addEntryEnum( "auto" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntry< std::string >( "global-index-type", "Type to use for the representation of global indices in the output mesh. When 'auto', the global index type from the input mesh is used.", "auto" );
   config.addEntryEnum( "auto" );
   config.addEntryEnum( "std::int32_t" );
   config.addEntryEnum( "std::int64_t" );
   config.addRequiredEntry< std::string >( "output-file", "Output mesh file path." );
   config.addEntry< std::string >( "output-file-format", "Output mesh file format.", "auto" );
   config.addEntryEnum( "auto" );
   config.addEntryEnum( "vtk" );
   config.addEntryEnum( "vtu" );
   config.addEntry< std::string >( "decomposition-type", "Type of decomposition to use.", "edge-bisection" );
   config.addEntryEnum( "edge-bisection" );
   config.addEntry< int >( "iterations", "Number of mesh refinement iterations.", 1 );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const std::string inputFileName = parameters.getParameter< std::string >( "input-file" );
   const std::string inputFileFormat = parameters.getParameter< std::string >( "input-file-format" );
   const std::string realType = parameters.getParameter< std::string >( "real-type" );
   const std::string globalIndexType = parameters.getParameter< std::string >( "global-index-type" );
   const std::string outputFileName = parameters.getParameter< std::string >( "output-file" );
   const std::string outputFileFormat = parameters.getParameter< std::string >( "output-file-format" );
   const std::string decompositionType = parameters.getParameter< std::string >( "decomposition-type" );
   const int iterations = parameters.getParameter< int >( "iterations" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return refineMesh( mesh, outputFileName, outputFileFormat, decompositionType, iterations );
   };
   const bool status = Meshes::resolveAndLoadMesh< MeshRefineConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat, realType, globalIndexType );
   return static_cast< int >( ! status );
}
