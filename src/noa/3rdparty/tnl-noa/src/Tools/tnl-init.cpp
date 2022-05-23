#include "tnl-init.h"

#include <TNL/File.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Functions/TestFunction.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>

#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>

using namespace TNL;

struct TnlInitConfigTag {};

namespace TNL {
namespace Meshes {
namespace BuildConfigTags {

// Configure real types
template<> struct GridRealTag< TnlInitConfigTag, float > { static constexpr bool enabled = true; };
template<> struct GridRealTag< TnlInitConfigTag, double > { static constexpr bool enabled = true; };
template<> struct GridRealTag< TnlInitConfigTag, long double > { static constexpr bool enabled = false; };

// Configure index types
template<> struct GridIndexTag< TnlInitConfigTag, short int >{ static constexpr bool enabled = false; };
template<> struct GridIndexTag< TnlInitConfigTag, int >{ static constexpr bool enabled = true; };
template<> struct GridIndexTag< TnlInitConfigTag, long int >{ static constexpr bool enabled = true; };

// Unstructured meshes are disabled, only grids can be on input.

} // namespace BuildConfigTags
} // namespace Meshes
} // namespace TNL

void setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "General settings:" );
   config.addEntry< String >( "mesh", "Input mesh file.", "mesh.vti" );
   config.addEntry< String >( "mesh-function-name", "Name of the mesh function in the VTI files.", "f" );
   config.addEntry< String >( "real-type", "Precision of the function evaluation.", "mesh-real-type" );
      config.addEntryEnum< String >( "mesh-real-type" );
      config.addEntryEnum< String >( "float" );
      config.addEntryEnum< String >( "double" );
//      config.addEntryEnum< String >( "long-double" );
   config.addEntry< double >( "initial-time", "Initial time for a serie of snapshots of the time-dependent function.", 0.0 );
   config.addEntry< double >( "final-time", "Final time for a serie of snapshots of the time-dependent function.", 0.0 );
   config.addEntry< double >( "snapshot-period", "Period between snapshots in a serie of the time-dependent function.", 0.0 );
   config.addEntry< int >( "x-derivative", "Order of the partial derivative w.r.t x.", 0 );
   config.addEntry< int >( "y-derivative", "Order of the partial derivative w.r.t y.", 0 );
   config.addEntry< int >( "z-derivative", "Order of the partial derivative w.r.t <.", 0 );
   config.addEntry< bool >( "numerical-differentiation", "The partial derivatives will be computed numerically.", false );
   config.addRequiredEntry< String >( "output-file", "Output file name." );
   config.addEntry< bool >( "check-output-file", "If the output file already exists, do not recreate it.", false );
   config.addEntry< String >( "help", "Write help." );

   config.addDelimiter( "Functions parameters:" );
   Functions::TestFunction< 1 >::configSetup( config );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription configDescription;

   setupConfig( configDescription );
   TNL::MPI::configSetup( configDescription );

   TNL::MPI::ScopedInitializer mpi(argc, argv);

   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return EXIT_FAILURE;

   const String meshFileName = parameters.getParameter< String >( "mesh" );
   const String meshFileFormat = "auto";

   auto wrapper = [&] ( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype(mesh) >;
      return resolveRealType< MeshType >( parameters );
   };
   return ! Meshes::resolveMeshType< TnlInitConfigTag, Devices::Host >( wrapper, meshFileName, meshFileFormat );
}
