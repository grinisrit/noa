// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/parseCommandLine.h>

#include "spmv.h"

#include <TNL/Matrices/MatrixReader.h>

#ifdef HAVE_PETSC
#include <petscmat.h>
#endif

using namespace TNL::Matrices;

#include <exception>
#include <ctime> // Used for file naming, so logs don't get overwritten.
#include <experimental/filesystem> // check file existence

using namespace TNL;
using namespace TNL::Benchmarks;

template< typename Real >
void
runSpMVBenchmarks( TNL::Benchmarks::SpMV::BenchmarkType & benchmark,
                   const String & inputFileName,
                   const Config::ParameterContainer& parameters,
                   bool verboseMR = false )
{
   // Start the actual benchmark in spmv.h
   try {
      TNL::Benchmarks::SpMV::benchmarkSpmv< Real >( benchmark, inputFileName, parameters, verboseMR );
   }
   catch( const std::exception& ex ) {
      std::cerr << ex.what() << std::endl;
   }
}

// Get current date time to have different log files names and avoid overwriting.
std::string getCurrDateTime()
{
   time_t rawtime;
   struct tm * timeinfo;
   char buffer[ 80 ];
   time( &rawtime );
   timeinfo = localtime( &rawtime );
   strftime( buffer, sizeof( buffer ), "%Y-%m-%d--%H:%M:%S", timeinfo );
   std::string curr_date_time( buffer );
   return curr_date_time;
}

void
setupConfig( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addRequiredEntry< String >( "input-file", "Input file name." );
   config.addEntry< bool >( "with-symmetric-matrices", "Perform benchmark even for symmetric matrix formats.", true );
   config.addEntry< bool >( "with-legacy-matrices", "Perform benchmark even for legacy TNL matrix formats.", true );
   config.addEntry< bool >( "with-all-cpu-tests", "All matrix formats are tested on both CPU and GPU. ", false );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-spmv::" + getCurrDateTime() + ".log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "append" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
   config.addEntryEnum( "float" );
   config.addEntryEnum( "double" );
   config.addEntryEnum( "all" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< int >( "verbose-MReader", "Verbose mode for Matrix Reader.", 0 );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

int
main( int argc, char* argv[] )
{
#ifdef HAVE_PETSC
   PetscInitialize( &argc, &argv, nullptr, nullptr );
#endif
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   setupConfig( conf_desc );

   // FIXME: When ./tnl-benchmark-spmv-dbg is called without parameters:
   //           * The guide on what parameters to use prints twice.
   // FIXME: When ./tnl-benchmark-spmv-dbg is called with '--help':
   //           * The guide on what parameter to use print once.
   //              But then it CRASHES due to segfault:
   //              The program attempts to get unknown parameter openmp-enabled
   //              Aborting the program.
   //              terminate called after throwing an instance of 'int'
   //      [1]    17156 abort (core dumped)  ~/tnl-dev/Debug/bin/./tnl-benchmark-spmv-dbg --help

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const String & inputFileName = parameters.getParameter< String >( "input-file" );
   const String & logFileName = parameters.getParameter< String >( "log-file" );
   String outputMode = parameters.getParameter< String >( "output-mode" );
   const String & precision = parameters.getParameter< String >( "precision" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   const int verboseMR = parameters.getParameter< int >( "verbose-MReader" );

   // open log file
   if( inputFileName == "" )
   {
      std::cerr << "ERROR: Input file name is required." << std::endl;
      return EXIT_FAILURE;
   }
   if( std::experimental::filesystem::exists(logFileName.getString()) )
   {
      std::cout << "Log file " << logFileName << " exists and ";
      if( outputMode == "append" )
         std::cout << "new logs will be appended." << std::endl;
      else
         std::cout << "will be overwritten." << std::endl;
   }

   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName, mode );

   // init benchmark and set parameters
   TNL::Benchmarks::SpMV::BenchmarkType benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   // Initiate setup of benchmarks
   if( precision == "all" || precision == "float" )
      runSpMVBenchmarks< float >( benchmark, inputFileName, parameters, verboseMR );
   if( precision == "all" || precision == "double" )
      runSpMVBenchmarks< double >( benchmark, inputFileName, parameters, verboseMR );

   // Confirm that the benchmark has finished
   std::cout << "\n==> BENCHMARK FINISHED" << std::endl;
   return EXIT_SUCCESS;
}
