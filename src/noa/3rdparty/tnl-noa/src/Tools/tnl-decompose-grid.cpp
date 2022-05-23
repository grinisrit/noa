// Implemented by: Jakub Klinkovský

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Meshes/Writers/PVTIWriter.h>

using namespace TNL;

struct DecomposeGridConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn on all grids.
 */
template<> struct GridRealTag< DecomposeGridConfigTag, float > { static constexpr bool enabled = true; };
template<> struct GridRealTag< DecomposeGridConfigTag, double > { static constexpr bool enabled = true; };
template<> struct GridRealTag< DecomposeGridConfigTag, long double > { static constexpr bool enabled = true; };

template<> struct GridIndexTag< DecomposeGridConfigTag, int > { static constexpr bool enabled = true; };
template<> struct GridIndexTag< DecomposeGridConfigTag, long int > { static constexpr bool enabled = true; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL


void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file with the grid." );
   config.addEntry< String >( "input-file-format", "Input grid file format.", "auto" );
   config.addRequiredEntry< String >( "output-file", "Output mesh file (PVTI format)." );
   config.addEntry< unsigned >( "subdomains-x", "Number of grid subdomains along the x-axis.", 1 );
   config.addEntry< unsigned >( "subdomains-y", "Number of grid subdomains along the y-axis.", 1 );
   config.addEntry< unsigned >( "subdomains-z", "Number of grid subdomains along the z-axis.", 1 );
   config.addEntry< unsigned >( "ghost-levels", "Number of ghost levels by which the subdomains overlap.", 0 );
// TODO: implement this in the distributed grid (it should configure communication over faces/edges/vertices in the grid synchronizer)
//   config.addEntry< unsigned >( "min-common-vertices",
//                                "Specifies the number of common nodes that two elements must have in order to put an "
//                                "edge between them in the dual graph. By default it is equal to the mesh dimension." );
}

template< typename CoordinatesType >
CoordinatesType getRankCoordinates( typename CoordinatesType::ValueType rank,
                                    CoordinatesType decomposition )
{
   CoordinatesType coordinates;
   using Index = typename CoordinatesType::ValueType;
   Index size = TNL::product( decomposition );
   for( int i = decomposition.getSize() - 1; i >= 0; i-- )
   {
      size = size / decomposition[ i ];
      coordinates[ i ] = rank / size;
      rank = rank % size;
   }
   return coordinates;
}

template< typename GridType >
void run( const GridType& globalGrid, const Config::ParameterContainer& parameters )
{
   using CoordinatesType = typename GridType::CoordinatesType;

   // prepare the grid decomposition
   CoordinatesType decomposition;
   decomposition[ 0 ] = parameters.getParameter< unsigned >( "subdomains-x" );
   if( decomposition.getSize() > 1 )
      decomposition[ 1 ] = parameters.getParameter< unsigned >( "subdomains-y" );
   if( decomposition.getSize() > 2 )
      decomposition[ 2 ] = parameters.getParameter< unsigned >( "subdomains-z" );

   // prepare the ghost levels
   const unsigned ghost_levels = parameters.getParameter< unsigned >( "ghost-levels" );

   // write a .pvti file
   using PVTI = Meshes::Writers::PVTIWriter< GridType >;
   const std::string pvtiFileName = parameters.template getParameter< String >( "output-file" );
   std::ofstream file( pvtiFileName );
   PVTI pvti( file );
   pvti.writeImageData( globalGrid, ghost_levels ); // TODO: ..., ncommon );
   // TODO
//   if( ghost_levels > 0 ) {
//      // the PointData and CellData from the individual files should be added here
//      pvtu.template writePPointData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
//      pvtu.template writePCellData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
//   }

   std::cout << "Writing subdomains..." << std::endl;
   const unsigned nproc = TNL::product( decomposition );
   for( unsigned p = 0; p < nproc; p++ ) {
      // cartesian coordinates of the p-th rank in the decomposition
      const CoordinatesType rank_coordinates = getRankCoordinates( p, decomposition );
      std::cout << p << "-th rank_coordinates: " << rank_coordinates << std::endl;

      // prepare local grid attributes
      CoordinatesType globalBegin;
      CoordinatesType localSize;
      for( int i = 0; i < GridType::getMeshDimension(); i++ ) {
         const auto numberOfLarger = globalGrid.getDimensions()[ i ] % decomposition[ i ];
         localSize[ i ] = globalGrid.getDimensions()[ i ] / decomposition[ i ];
         if( numberOfLarger > rank_coordinates[ i ] )
            ++localSize[ i ];

         if( numberOfLarger > rank_coordinates[ i ] )
             globalBegin[ i ] = rank_coordinates[ i ] * localSize[ i ];
         else
             globalBegin[ i ] = numberOfLarger * (localSize[ i ] + 1) + (rank_coordinates[ i ] - numberOfLarger) * localSize[ i ];
      }

      const std::string outputFileName = pvti.addPiece( pvtiFileName, p, globalBegin, globalBegin + localSize );
      std::cout << outputFileName << std::endl;

      // write the subdomain
      using Writer = Meshes::Writers::VTIWriter< GridType >;
      std::ofstream file( outputFileName );
      Writer writer( file );
      writer.writeImageData( globalGrid.getOrigin(),
                             globalBegin,
                             globalBegin + localSize,
                             globalGrid.getSpaceSteps() );
      // TODO
//      if( ghost_levels > 0 ) {
//         writer.writePointData( pointGhosts, Meshes::VTK::ghostArrayName() );
//         writer.writeCellData( cellGhosts, Meshes::VTK::ghostArrayName() );
//      }
   }
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String inputFileName = parameters.getParameter< String >( "input-file" );
   const String inputFileFormat = parameters.getParameter< String >( "input-file-format" );
   const String outputFile = parameters.template getParameter< String >( "output-file" );
   if( ! outputFile.endsWith( ".pvti" ) ) {
      std::cerr << "Error: the output file must have a '.pvti' extension." << std::endl;
      return EXIT_FAILURE;
   }

   auto wrapper = [&] ( const auto& reader, auto&& grid )
   {
      using GridType = std::decay_t< decltype(grid) >;
      run( std::forward<GridType>(grid), parameters );
      return true;
   };
   const bool status = Meshes::resolveAndLoadMesh< DecomposeGridConfigTag, Devices::Host >( wrapper, inputFileName, inputFileFormat );
   return static_cast< int >( ! status );
}
