#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/NetgenWriter.h>

using namespace TNL;

struct GridToMeshConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
template<> struct GridRealTag< GridToMeshConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< GridToMeshConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< GridToMeshConfigTag, short int >{ static constexpr bool enabled = false; };
template<> struct GridIndexTag< GridToMeshConfigTag, long int >{ static constexpr bool enabled = false; };

/****
 * Unstructured meshes are disabled, only grids can be on input.
 */

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

// cannot be deduced from GridType
using LocalIndexType = short int;

template< typename Mesh >
struct MeshCreator
{
   using MeshType = Mesh;

   static bool run( const Mesh& meshIn, Mesh& meshOut )
   {
      std::cerr << "Got a mesh on the input." << std::endl;
      return false;
   }
};

template< typename Real, typename Device, typename Index >
struct MeshCreator< Meshes::Grid< 1, Real, Device, Index > >
{
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   using CellTopology = Meshes::Topologies::Edge;
   using MeshConfig = Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      Meshes::MeshBuilder< MeshType > meshBuilder;
      meshBuilder.setEntitiesCount( numberOfVertices, numberOfCells );

      for( Index i = 0; i < numberOfVertices; i++ ) {
         const auto vertex = grid.template getEntity< typename GridType::Vertex >( i );
         meshBuilder.setPoint( i, vertex.getCenter() );
      }

      for( Index i = 0; i < numberOfCells; i++ ) {
         auto cell = grid.template getEntity< typename GridType::Cell >( i );
         cell.refresh();
         meshBuilder.getCellSeed( i ).setCornerId( 0, cell.template getNeighbourEntity< 0 >( { 0 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 1, cell.template getNeighbourEntity< 0 >( { 1 } ).getIndex() );
      }

      return meshBuilder.build( mesh );
   }
};

template< typename Real, typename Device, typename Index >
struct MeshCreator< Meshes::Grid< 2, Real, Device, Index > >
{
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   using CellTopology = Meshes::Topologies::Quadrangle;
   using MeshConfig = Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      Meshes::MeshBuilder< MeshType > meshBuilder;
      meshBuilder.setEntitiesCount( numberOfVertices, numberOfCells );

      for( Index i = 0; i < numberOfVertices; i++ ) {
         const auto vertex = grid.template getEntity< typename GridType::Vertex >( i );
         meshBuilder.setPoint( i, vertex.getCenter() );
      }

      for( Index i = 0; i < numberOfCells; i++ ) {
         const auto cell = grid.template getEntity< typename GridType::Cell >( i );
         meshBuilder.getCellSeed( i ).setCornerId( 0, cell.template getNeighbourEntity< 0 >( { 0, 0 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 1, cell.template getNeighbourEntity< 0 >( { 1, 0 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 2, cell.template getNeighbourEntity< 0 >( { 1, 1 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 3, cell.template getNeighbourEntity< 0 >( { 0, 1 } ).getIndex() );
      }

      return meshBuilder.build( mesh );
   }
};

template< typename Real, typename Device, typename Index >
struct MeshCreator< Meshes::Grid< 3, Real, Device, Index > >
{
   using GridType = Meshes::Grid< 3, Real, Device, Index >;
   using CellTopology = Meshes::Topologies::Hexahedron;
   using MeshConfig = Meshes::DefaultConfig< CellTopology,
                                                  CellTopology::dimension,
                                                  typename GridType::RealType,
                                                  typename GridType::GlobalIndexType,
                                                  LocalIndexType >;
   using MeshType = Meshes::Mesh< MeshConfig >;

   static bool run( const GridType& grid, MeshType& mesh )
   {
      const Index numberOfVertices = grid.template getEntitiesCount< typename GridType::Vertex >();
      const Index numberOfCells = grid.template getEntitiesCount< typename GridType::Cell >();

      Meshes::MeshBuilder< MeshType > meshBuilder;
      meshBuilder.setEntitiesCount( numberOfVertices, numberOfCells );

      for( Index i = 0; i < numberOfVertices; i++ ) {
         const auto vertex = grid.template getEntity< typename GridType::Vertex >( i );
         meshBuilder.setPoint( i, vertex.getCenter() );
      }

      for( Index i = 0; i < numberOfCells; i++ ) {
         const auto cell = grid.template getEntity< typename GridType::Cell >( i );
         meshBuilder.getCellSeed( i ).setCornerId( 0, cell.template getNeighbourEntity< 0 >( { 0, 0, 0 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 1, cell.template getNeighbourEntity< 0 >( { 1, 0, 0 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 2, cell.template getNeighbourEntity< 0 >( { 1, 1, 0 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 3, cell.template getNeighbourEntity< 0 >( { 0, 1, 0 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 4, cell.template getNeighbourEntity< 0 >( { 0, 0, 1 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 5, cell.template getNeighbourEntity< 0 >( { 1, 0, 1 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 6, cell.template getNeighbourEntity< 0 >( { 1, 1, 1 } ).getIndex() );
         meshBuilder.getCellSeed( i ).setCornerId( 7, cell.template getNeighbourEntity< 0 >( { 0, 1, 1 } ).getIndex() );
      }

      return meshBuilder.build( mesh );
   }
};

template< typename Grid >
bool convertGrid( Grid& grid, const std::string& outputFileName, const std::string& outputFormat )
{
   using MeshCreator = MeshCreator< Grid >;
   using Mesh = typename MeshCreator::MeshType;

   Mesh mesh;
   if( ! MeshCreator::run( grid, mesh ) ) {
      std::cerr << "Unable to build mesh from grid." << std::endl;
      return false;
   }

   std::string format = outputFormat;
   if( outputFormat == "auto" ) {
      namespace fs = std::filesystem;
      format = fs::path( outputFileName ).extension().string();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   if( format == "vtk" ) {
      using VTKWriter = Meshes::Writers::VTKWriter< Mesh >;
      std::ofstream file( outputFileName );
      VTKWriter writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   if( format == "vtu" ) {
      using VTKWriter = Meshes::Writers::VTUWriter< Mesh >;
      std::ofstream file( outputFileName );
      VTKWriter writer( file );
      writer.template writeEntities< Mesh::getMeshDimension() >( mesh );
      return true;
   }
   if( format == "ng" ) {
      using NetgenWriter = Meshes::Writers::NetgenWriter< Mesh >;
      std::fstream file( outputFileName );
      NetgenWriter::writeMesh( mesh, file );
      return true;
   }

   if( outputFormat == "auto" )
      std::cerr << "File '" << outputFileName << "' has unsupported format (based on the file extension): " << format << ".";
   else
      std::cerr << "Unsupported output file format: " << outputFormat << ".";
   std::cerr << " Supported formats are 'vtk', 'vtu' and 'ng'." << std::endl;
   return false;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< std::string >( "input-file", "Input file with the mesh." );
   config.addEntry< std::string >( "input-file-format", "Input mesh file format.", "auto" );
   config.addRequiredEntry< std::string >( "output-file", "Output mesh file path." );
   config.addEntry< std::string >( "output-file-format", "Output mesh file format.", "auto" );
   config.addEntryEnum( "auto" );
   config.addEntryEnum( "vtk" );
   config.addEntryEnum( "vtu" );
   config.addEntryEnum( "ng" );
}

int
main( int argc, char* argv[] )
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

   auto wrapper = [&] ( const auto& reader, auto&& grid )
   {
      return convertGrid( grid, outputFileName, outputFileFormat );
   };
   const bool status = Meshes::resolveAndLoadMesh< GridToMeshConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
   return static_cast< int >( ! status );
}
