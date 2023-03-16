#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/FPMAWriter.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Meshes/Writers/NetgenWriter.h>

using namespace TNL;

struct MeshConverterConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< MeshConverterConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< MeshConverterConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< MeshConverterConfigTag, short int >{ static constexpr bool enabled = false; };
template<> struct GridIndexTag< MeshConverterConfigTag, long int >{ static constexpr bool enabled = false; };

/****
 * Unstructured meshes.
 */
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Edge > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Triangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Quadrangle > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Polygon > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Tetrahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Hexahedron > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Wedge > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Pyramid > { static constexpr bool enabled = true; };
template<> struct MeshCellTopologyTag< MeshConverterConfigTag, Topologies::Polyhedron > { static constexpr bool enabled = true; };

// Meshes are enabled only for the space dimension equal to the cell dimension.
template< typename CellTopology, int SpaceDimension >
struct MeshSpaceDimensionTag< MeshConverterConfigTag, CellTopology, SpaceDimension >
{ static constexpr bool enabled = SpaceDimension == CellTopology::dimension; };

// Meshes are enabled only for types explicitly listed below.
template<> struct MeshRealTag< MeshConverterConfigTag, float > { static constexpr bool enabled = true; };
template<> struct MeshRealTag< MeshConverterConfigTag, double > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshConverterConfigTag, int > { static constexpr bool enabled = true; };
template<> struct MeshGlobalIndexTag< MeshConverterConfigTag, long int > { static constexpr bool enabled = true; };
template<> struct MeshLocalIndexTag< MeshConverterConfigTag, short int > { static constexpr bool enabled = true; };

// Config tag specifying the MeshConfig template to use.
template<>
struct MeshConfigTemplateTag< MeshConverterConfigTag >
{
   template< typename Cell,
             int SpaceDimension = Cell::dimension,
             typename Real = double,
             typename GlobalIndex = int,
             typename LocalIndex = GlobalIndex >
   struct MeshConfig : public DefaultConfig< Cell, SpaceDimension, Real, GlobalIndex, LocalIndex >
   {
      static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
      {
         // faces must be stored for polyhedral meshes
         if( std::is_same< Cell, TNL::Meshes::Topologies::Polyhedron >::value ) {
            if( subentityDimension == 0 && entityDimension == Cell::dimension - 1 )
               return true;
            if( subentityDimension == Cell::dimension - 1 && entityDimension == Cell::dimension )
               return true;
         }
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

// overload for meshes
template< typename Mesh >
bool writeMesh( const Mesh& mesh, std::ostream& out, const std::string& format )
{
   if constexpr( std::is_same_v< typename Mesh::Cell::EntityTopology, TNL::Meshes::Topologies::Polyhedron > ) {
      if( format == "fpma" ) {
         using Writer = Meshes::Writers::FPMAWriter< Mesh >;
         Writer writer( out );
         writer.writeEntities( mesh );
         return true;
      }
   }
   else {
      if( format == "ng" ) {
         using NetgenWriter = Meshes::Writers::NetgenWriter< Mesh >;
         NetgenWriter::writeMesh( mesh, out );
         return true;
      }
   }
   if( format == "vtk" ) {
      using Writer = Meshes::Writers::VTKWriter< Mesh >;
      Writer writer( out );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   if( format == "vtu" ) {
      using Writer = Meshes::Writers::VTUWriter< Mesh >;
      Writer writer( out );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   return false;
}

// overload for grids
template< int Dimension, typename Real, typename Device, typename Index >
bool writeMesh( const TNL::Meshes::Grid< Dimension, Real, Device, Index >& mesh,
                std::ostream& out,
                const std::string& format )
{
   using Mesh = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   if( format == "vtk" ) {
      using Writer = Meshes::Writers::VTKWriter< Mesh >;
      Writer writer( out );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   if( format == "vtu" ) {
      using Writer = Meshes::Writers::VTUWriter< Mesh >;
      Writer writer( out );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   if( format == "vti" ) {
      using Writer = Meshes::Writers::VTIWriter< Mesh >;
      Writer writer( out );
      writer.writeImageData( mesh );
      return true;
   }
   return false;
}

template< typename Mesh >
bool convertMesh( const Mesh& mesh, const std::string& inputFileName, const std::string& outputFileName, const std::string& outputFormat )
{
   std::string format = outputFormat;
   if( outputFormat == "auto" ) {
      namespace fs = std::filesystem;
      format = fs::path( outputFileName ).extension().string();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   std::ofstream file( outputFileName );
   if( writeMesh( mesh, file, format ) )
      return true;

   if( outputFormat == "auto" )
      std::cerr << "File '" << outputFileName << "' has unsupported format (based on the file extension): " << format << ".";
   else
      std::cerr << "Unsupported output file format: " << outputFormat << ".";
   std::cerr << " Supported formats are 'vtk', 'vtu', 'vti' (only grids), 'ng' (only static unstructured meshes) and 'fpma' (only polyhedral meshes)." << std::endl;
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
   config.addEntryEnum( "vti" );
   config.addEntryEnum( "ng" );
   config.addEntryEnum( "fpma" );
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

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      return convertMesh( mesh, inputFileName, outputFileName, outputFileFormat );
   };
   const bool status = Meshes::resolveAndLoadMesh< MeshConverterConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat, realType, globalIndexType );
   return static_cast< int >( ! status );
}
