#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/FPMAWriter.h>
#include <TNL/Meshes/Geometry/getPlanarMesh.h>

using namespace TNL;

struct MeshPlanarCorrectConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< MeshPlanarCorrectConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MeshPlanarCorrectConfigTag, double > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MeshPlanarCorrectConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< MeshPlanarCorrectConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshPlanarCorrectConfigTag, Topologies::Polyhedron > { static constexpr bool enabled = true; };

// Meshes are enabled only for the space dimension equal to 3
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MeshPlanarCorrectConfigTag, CellTopology, SpaceDimension >
{ static constexpr bool enabled = SpaceDimension == 3; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MeshPlanarCorrectConfigTag, float > { static constexpr bool enabled = true; };
template<> struct MeshRealTag< MeshPlanarCorrectConfigTag, double > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshPlanarCorrectConfigTag, long int > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshPlanarCorrectConfigTag, int > { static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< MeshPlanarCorrectConfigTag, short int > { static constexpr bool enabled = true; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MeshPlanarCorrectConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = float,
             typename GlobalIndex = int,
             typename LocalIndex = short int >
   struct MeshConfig
   {
      using CellTopology = Cell;
      using RealType = Real;
      using GlobalIndexType = GlobalIndex;
      using LocalIndexType = LocalIndex;

      static constexpr int spaceDimension = SpaceDimension;
      static constexpr int meshDimension = Cell::dimension;

      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         return   (subentityDimension == 0 && entityDimension == meshDimension)
               || (subentityDimension == meshDimension - 1 && entityDimension == meshDimension )
               || (subentityDimension == 0 && entityDimension == meshDimension - 1 );
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

using namespace TNL::Meshes;

template< typename Mesh >
auto getPlanarMeshHelper( const Mesh& mesh, const std::string& decompositionType )
{
   using namespace TNL::Meshes;

   if( decompositionType[0] == 'c' ) {
      return getPlanarMesh< EntityDecomposerVersion::ConnectEdgesToCentroid >( mesh );
   }
   else { // decompositionType[0] == 'p'
      return getPlanarMesh< EntityDecomposerVersion::ConnectEdgesToPoint >( mesh );
   }
}

template< typename Topology >
struct PlanarMeshWriter;

template<>
struct PlanarMeshWriter< Topologies::Polygon >
{
   template< typename Mesh >
   static bool exec( const Mesh& mesh, const std::string& outputFileName, const std::string& outputFormat )
   {
      std::string format = outputFormat;
      if( outputFormat == "auto" ) {
         namespace fs = std::filesystem;
         format = fs::path( outputFileName ).extension().string();
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
};

template<>
struct PlanarMeshWriter< Topologies::Polyhedron >
{
   template< typename Mesh >
   static bool exec( const Mesh& mesh, const std::string& outputFileName, const std::string& outputFormat )
   {
      if( outputFormat != "auto" && outputFormat != "fpma" ) {
         std::cerr << "Unsupported output file format: " << outputFormat << ". Only 'fpma' is supported for polyhedral meshes." << std::endl;
         return false;
      }

      using Writer = Meshes::Writers::FPMAWriter< Mesh >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.writeEntities( mesh );
      return true;
   }
};

template< typename Mesh >
bool triangulateMesh( const Mesh& mesh, const std::string& outputFileName, const std::string& outputFormat, const std::string& decompositionType )
{
   const auto planarMesh = getPlanarMeshHelper( mesh, decompositionType );
   using PlanarMesh = decltype( planarMesh );
   using CellTopology = typename PlanarMesh::Cell::EntityTopology;
   return PlanarMeshWriter< CellTopology >::exec( planarMesh, outputFileName, outputFormat );
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< std::string >( "input-file", "Input file with the mesh." );
   config.addEntry< std::string >( "input-file-format", "Input mesh file format.", "auto" );
   config.addRequiredEntry< std::string >( "output-file", "Output mesh file path." );
   config.addEntry< std::string >( "output-file-format", "Output mesh file format.", "auto" );
   config.addRequiredEntry< std::string >( "decomposition-type", "Type of decomposition to use for non-planar polygons." );
   config.addEntryEnum( "c" );
   config.addEntryEnum( "p" );
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
   const std::string outputFileName = parameters.getParameter< std::string >( "output-file" );
   const std::string outputFileFormat = parameters.getParameter< std::string >( "output-file-format" );
   const std::string decompositionType = parameters.getParameter< std::string >( "decomposition-type" );

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return triangulateMesh( mesh, outputFileName, outputFileFormat, decompositionType );
   };
   return ! Meshes::resolveAndLoadMesh< MeshPlanarCorrectConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
}
