// Implemented by: Ján Bobot, Jakub Klinkovský

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Meshes/Readers/getMeshReader.h>

#include "MeshBenchmarks.h"


using namespace TNL;
using namespace TNL::Meshes;
using namespace TNL::Meshes::Readers;
using namespace TNL::Benchmarks;

template< typename CellTopology,
          int SpaceDimension = CellTopology::dimension,
          typename... Params >
bool
setMeshParameters( Params&&... params )
{
   bool status = MeshBenchmarksRunner< MinimalConfig, CellTopology, SpaceDimension, float, int, short int >::run( std::forward<Params>(params)... ) &&
                 MeshBenchmarksRunner< FullConfig, CellTopology, SpaceDimension, float, int, short int >::run( std::forward<Params>(params)... );
   return status;
}

bool
resolveCellTopology( Benchmark<> & benchmark,
                     const Config::ParameterContainer& parameters )
{
   const String & meshFile = parameters.getParameter< String >( "mesh-file" );

   auto reader = getMeshReader( meshFile, "auto" );

   try {
      reader->detectMesh();
   }
   catch( const Meshes::Readers::MeshReaderError& e ) {
      std::cerr << "Failed to detect mesh from file '" << meshFile << "'." << std::endl;
      std::cerr << e.what() << std::endl;
      return false;
   }

   if( reader->getMeshType() != "Meshes::Mesh" ) {
      std::cerr << "The mesh type " << reader->getMeshType() << " is not supported." << std::endl;
      return false;
   }

   using VTK::EntityShape;
   switch( reader->getCellShape() )
   {
      case EntityShape::Triangle:
         return setMeshParameters< Topologies::Triangle >( benchmark, parameters );
      case EntityShape::Tetra:
         return setMeshParameters< Topologies::Tetrahedron >( benchmark, parameters );
      case EntityShape::Polygon:
         switch( reader->getSpaceDimension() )
         {
            case 2:
               return setMeshParameters< Topologies::Polygon, 2 >( benchmark, parameters );
            case 3:
               return setMeshParameters< Topologies::Polygon, 3 >( benchmark, parameters );
            default:
               std::cerr << "unsupported dimension for polygon mesh: " << reader->getSpaceDimension() << std::endl;
               return false;
         }
      case EntityShape::Polyhedron:
         return setMeshParameters< Topologies::Polyhedron >( benchmark, parameters );
      default:
         std::cerr << "unsupported cell topology: " << getShapeName( reader->getCellShape() ) << std::endl;
         return false;
   }
}

void
setupConfig( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-mesh.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< size_t >( "mem-limit", "Memory limit in MiB for mesh memory measurement.", 8000 );
   config.addRequiredEntry< String >( "mesh-file", "Path of the mesh to load for the benchmark." );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
   #ifdef HAVE_CUDA
   config.addEntryEnum( "cuda" );
   #endif

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
       return 1;

   Devices::Host::setup( parameters );
   Devices::Cuda::setup( parameters );

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName.getString(), mode );

   // init benchmark and common metadata
   Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   if( ! resolveCellTopology( benchmark, parameters ) )
      return EXIT_FAILURE;

   return EXIT_SUCCESS;
}
