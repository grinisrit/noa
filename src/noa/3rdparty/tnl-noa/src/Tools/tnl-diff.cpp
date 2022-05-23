#include "tnl-diff.h"
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

struct TNLDiffBuildConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

/****
 * Turn off support for float and long double.
 */
//template<> struct GridRealTag< TNLDiffBuildConfigTag, float > { static constexpr bool enabled = false; };
template<> struct GridRealTag< TNLDiffBuildConfigTag, long double > { static constexpr bool enabled = false; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct GridIndexTag< TNLDiffBuildConfigTag, short int >{ static constexpr bool enabled = false; };
template<> struct GridIndexTag< TNLDiffBuildConfigTag, long int >{ static constexpr bool enabled = false; };

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

void setupConfig( Config::ConfigDescription& config )
{
   config.addEntry< String >( "mesh", "Input mesh file.", "mesh.vti" );
   config.addEntry< String >( "mesh-format", "Mesh file format.", "auto" );
   config.addRequiredList< String >( "input-files", "Input files containing the mesh functions to be compared." );
   config.addEntry< String >( "mesh-function-name", "Name of the mesh function in the input files.", "f" );
   config.addEntry< String >( "output-file", "File for the output data.", "tnl-diff.log" );
   config.addEntry< String >( "mode", "Mode 'couples' compares two subsequent files. Mode 'sequence' compares the input files against the first one. 'halves' compares the files from the first and the second half of the intput files.", "couples" );
      config.addEntryEnum< String >( "couples" );
      config.addEntryEnum< String >( "sequence" );
      config.addEntryEnum< String >( "halves" );
   config.addEntry< bool >( "exact-match", "Check if the data are exactly the same.", false );
   config.addEntry< bool >( "write-difference", "Write difference grid function.", false );
//   config.addEntry< bool >( "write-exact-curve", "Write exact curve with given radius.", false );
   config.addEntry< int >( "edges-skip", "Width of the edges that will be skipped - not included into the error norms.", 0 );
//   config.addEntry< bool >( "write-graph", "Draws a graph in the Gnuplot format of the dependence of the error norm on t.", true );
//   config.addEntry< bool >( "write-log-graph", "Draws a logarithmic graph in the Gnuplot format of the dependence of the error norm on t.", true );
   config.addEntry< double >( "snapshot-period", "The period between consecutive snapshots.", 0.0 );
   config.addEntry< bool >( "verbose", "Sets verbosity.", true );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;
   setupConfig( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const String meshFile = parameters.getParameter< String >( "mesh" );
   const String meshFileFormat = parameters.getParameter< String >( "mesh-format" );
   auto wrapper = [&] ( const auto& reader, auto&& mesh )
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      return processFiles< MeshType >( parameters );
   };
   return ! TNL::Meshes::resolveMeshType< TNLDiffBuildConfigTag, Devices::Host >( wrapper, meshFile, meshFileFormat );
}
