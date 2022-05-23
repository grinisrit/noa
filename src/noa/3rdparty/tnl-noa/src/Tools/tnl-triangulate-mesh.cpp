#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Geometry/getDecomposedMesh.h>

using namespace TNL;

struct MeshTriangulatorConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off all grids.
 */
template<> struct GridRealTag< MeshTriangulatorConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MeshTriangulatorConfigTag, double > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MeshTriangulatorConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< MeshTriangulatorConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshTriangulatorConfigTag, Topologies::Polyhedron > { static constexpr bool enabled = true; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MeshTriangulatorConfigTag, CellTopology, SpaceDimension >
{ static constexpr bool enabled = SpaceDimension == CellTopology::dimension; };

// Polygonal Meshes are enable for the space dimension equal to 2 or 3
template< int SpaceDimension >
struct MeshSpaceDimensionTag< MeshTriangulatorConfigTag, Topologies::Polygon, SpaceDimension >
{ static constexpr bool enabled = SpaceDimension >= 2 && SpaceDimension <= 3; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MeshTriangulatorConfigTag, float > { static constexpr bool enabled = true; };
template<> struct MeshRealTag< MeshTriangulatorConfigTag, double > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshTriangulatorConfigTag, long int > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshTriangulatorConfigTag, int > { static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< MeshTriangulatorConfigTag, short int > { static constexpr bool enabled = true; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MeshTriangulatorConfigTag >
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

template< typename Mesh >
auto getDecomposedMeshHelper( const Mesh& mesh, const std::string& decompositionType )
{
   using namespace TNL::Meshes;

   if( decompositionType[0] == 'c' ) {
      if( decompositionType[1] == 'c' ) {
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToCentroid,
                                   EntityDecomposerVersion::ConnectEdgesToCentroid >( mesh );
      }
      else { // decompositionType[1] == 'p'
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToCentroid,
                                   EntityDecomposerVersion::ConnectEdgesToPoint >( mesh );
      }
   }
   else { // decompositionType[0] == 'p'
      if( decompositionType[1] == 'c' ) {
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToPoint,
                                   EntityDecomposerVersion::ConnectEdgesToCentroid >( mesh );
      }
      else { // decompositionType[1] == 'p'
         return getDecomposedMesh< EntityDecomposerVersion::ConnectEdgesToPoint,
                                   EntityDecomposerVersion::ConnectEdgesToPoint >( mesh );
      }
   }
}

template< typename Mesh >
bool triangulateMesh( const Mesh& mesh, const std::string& outputFileName, const std::string& outputFormat, const std::string& decompositionType )
{
   const auto decomposedMesh = getDecomposedMeshHelper( mesh, decompositionType );

   std::string format = outputFormat;
   if( outputFormat == "auto" ) {
      namespace fs = std::experimental::filesystem;
      format = fs::path( outputFileName ).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   if( format == "vtk" ) {
      using Writer = Meshes::Writers::VTKWriter< decltype( decomposedMesh ) >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( decomposedMesh );
      return true;
   }
   if( format == "vtu" ) {
      using Writer = Meshes::Writers::VTUWriter< decltype( decomposedMesh ) >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( decomposedMesh );
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
   config.addRequiredEntry< std::string >( "output-file", "Output mesh file path." );
   config.addRequiredEntry< std::string >( "output-file-format", "Output mesh file format." );
   config.addEntryEnum( "vtk" );
   config.addEntryEnum( "vtu" );
   config.addRequiredEntry< std::string >( "decomposition-type", "Type of decomposition to use." );
   config.addEntryEnum( "cc" );
   config.addEntryEnum( "cp" );
   config.addEntryEnum( "pc" );
   config.addEntryEnum( "pp" );
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
   const bool status = Meshes::resolveAndLoadMesh< MeshTriangulatorConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
   return static_cast< int >( ! status );
}
