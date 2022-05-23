// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
//#include "grid-traversing.h"
#include "GridTraversersBenchmark.h"

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Algorithms/ParallelFor.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Benchmarks::Traversers;


template< typename T, typename S >
bool containsValue( const std::vector< T >& container, const S& value )
{
   return std::find( container.begin(), container.end(), value ) != container.end();
}

template< int Dimension,
          typename Real = float,
          typename Index = int >
bool runBenchmark( const Config::ParameterContainer& parameters,
                   Benchmark<>& benchmark )
{
   const std::vector< String >& tests = parameters.getParameter< std::vector< String > >( "tests" );
   // FIXME: getParameter< std::size_t >() does not work with parameters added with addEntry< int >(),
   // which have a default value. The workaround below works for int values, but it is not possible
   // to pass 64-bit integer values
   // const std::size_t minSize = parameters.getParameter< std::size_t >( "min-size" );
   // const std::size_t maxSize = parameters.getParameter< std::size_t >( "max-size" );
   const std::size_t minSize = parameters.getParameter< int >( "min-size" );
   const std::size_t maxSize = parameters.getParameter< int >( "max-size" );
   const bool withHost = parameters.getParameter< bool >( "with-host" );
#ifdef HAVE_CUDA
   const bool withCuda = parameters.getParameter< bool >( "with-cuda" );
//#else
//   const bool withCuda = false;
#endif
   const bool check = parameters.getParameter< bool >( "check" );

   /****
    * Full grid traversing with no boundary conditions
    */
   for( std::size_t size = minSize; size <= maxSize; size *= 2 )
   {
      GridTraversersBenchmark< Dimension, Devices::Host, Real, Index > hostTraverserBenchmark( size );
#ifdef HAVE_CUDA
      GridTraversersBenchmark< Dimension, Devices::Cuda, Real, Index > cudaTraverserBenchmark( size );
#endif

      auto hostReset = [&]()
      {
         hostTraverserBenchmark.reset();
      };

#ifdef HAVE_CUDA
      auto cudaReset = [&]()
      {
         cudaTraverserBenchmark.reset();
      };
#endif
      benchmark.setMetadataColumns({
            { "dimension", convertToString( Dimension ) },
            { "traverser", "without BC" },
            { "size", convertToString( size ) },
      });

      /****
       * Add one using pure C code
       */
      if( containsValue( tests, "all" ) || containsValue( tests, "add-one-pure-c"  ) )
      {
         benchmark.setOperation( "Pure C", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );

         auto hostWriteOneUsingPureC = [&] ()
         {
            hostTraverserBenchmark.addOneUsingPureC();
         };
         if( withHost )
         {
            const BenchmarkResult result = benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingPureC );
            if( check && ! hostTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#ifdef HAVE_CUDA
         auto cudaWriteOneUsingPureC = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingPureC();
         };
         if( withCuda )
         {
            const BenchmarkResult result = benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingPureC );
            if( check && ! cudaTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#endif
      }

      /****
       * Add one using parallel for
       */
      if( containsValue( tests, "all" ) || containsValue( tests, "add-one-parallel-for" ) )
      {
         benchmark.setOperation( "parallel for", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );

         auto hostWriteOneUsingParallelFor = [&] ()
         {
            hostTraverserBenchmark.addOneUsingParallelFor();
         };
         if( withHost )
         {
            const BenchmarkResult result = benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingParallelFor );
            if( check && ! hostTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }

#ifdef HAVE_CUDA
         auto cudaWriteOneUsingParallelFor = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingParallelFor();
         };
         if( withCuda )
         {
            const BenchmarkResult result = benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingParallelFor );
            if( check && ! cudaTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#endif
      }

      /****
       * Add one using parallel for with grid entity
       */
      if( containsValue( tests, "all" ) || containsValue( tests, "add-one-simple-cell" ) )
      {
         auto hostAddOneUsingSimpleCell = [&] ()
         {
            hostTraverserBenchmark.addOneUsingSimpleCell();
         };
         benchmark.setOperation( "simple cell", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
         {
            const BenchmarkResult result = benchmark.time< Devices::Host >( hostReset, "CPU", hostAddOneUsingSimpleCell );
            if( check && ! hostTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#ifdef HAVE_CUDA
         auto cudaAddOneUsingSimpleCell = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingSimpleCell();
         };
         if( withCuda )
         {
            const BenchmarkResult result = benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaAddOneUsingSimpleCell );
            if( check && ! cudaTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#endif
      }

      /****
       * Add one using parallel for with mesh function
       */
      if( containsValue( tests, "all" ) || containsValue( tests, "add-one-parallel-for-and-mesh-function" ) )
      {
         auto hostAddOneUsingParallelForAndMeshFunction = [&] ()
         {
            hostTraverserBenchmark.addOneUsingParallelForAndMeshFunction();
         };
         benchmark.setOperation( "par.for+mesh fc.", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
         {
            const BenchmarkResult result = benchmark.time< Devices::Host >( hostReset, "CPU", hostAddOneUsingParallelForAndMeshFunction );
            if( check && ! hostTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#ifdef HAVE_CUDA
         auto cudaAddOneUsingParallelForAndMeshFunction = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingParallelForAndMeshFunction();
         };
         if( withCuda )
         {
            const BenchmarkResult result = benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaAddOneUsingParallelForAndMeshFunction );
            if( check && ! cudaTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#endif
      }

      /****
       * Add one using traverser
       */
      if( containsValue( tests, "all" ) || containsValue( tests, "add-one-traverser" ) )
      {
         benchmark.setOperation( "traverser", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         auto hostWriteOneUsingTraverser = [&] ()
         {
            hostTraverserBenchmark.addOneUsingTraverser();
         };
         if( withHost )
         {
            const BenchmarkResult result = benchmark.time< Devices::Host >( hostReset, "CPU", hostWriteOneUsingTraverser );
            if( check && ! hostTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }

#ifdef HAVE_CUDA
         auto cudaWriteOneUsingTraverser = [&] ()
         {
            cudaTraverserBenchmark.addOneUsingTraverser();
         };
         if( withCuda )
         {
            const BenchmarkResult result = benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaWriteOneUsingTraverser );
            if( check && ! cudaTraverserBenchmark.checkAddOne(
                  result.loops,
                  benchmark.isResetingOn() ) )
               benchmark.addErrorMessage( "Test results are not correct." );
         }
#endif
      }
      std::cout << "--------------------------------------------------------------------------------------------------------" << std::endl;
   }

   /****
    * Full grid traversing including boundary conditions
    */
   for( std::size_t size = minSize; size <= maxSize; size *= 2 )
   {
      GridTraversersBenchmark< Dimension, Devices::Host, Real, Index > hostTraverserBenchmark( size );
      GridTraversersBenchmark< Dimension, Devices::Cuda, Real, Index > cudaTraverserBenchmark( size );

      auto hostReset = [&]()
      {
         hostTraverserBenchmark.reset();
      };

#ifdef HAVE_CUDA
      auto cudaReset = [&]()
      {
         cudaTraverserBenchmark.reset();
      };
#endif

      benchmark.setMetadataColumns({
            { "dimension", convertToString( Dimension ) },
            { "traverser", "with BC" },
            { "size", convertToString( size ) },
      });

      /****
       * Write one and two (as BC) using C for
       */
      auto hostTraverseUsingPureC = [&] ()
      {
         hostTraverserBenchmark.traverseUsingPureC();
      };

#ifdef HAVE_CUDA
      auto cudaTraverseUsingPureC = [&] ()
      {
         cudaTraverserBenchmark.traverseUsingPureC();
      };
#endif

      if( containsValue( tests, "all" ) || containsValue( tests, "bc-pure-c" ) )
      {
         benchmark.setOperation( "Pure C", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( "CPU", hostTraverseUsingPureC );

#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingPureC );
#endif
         benchmark.setOperation( "Pure C RST", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingPureC );

#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingPureC );
#endif
      }

      /****
       * Write one and two (as BC) using parallel for
       */
      auto hostTraverseUsingParallelFor = [&] ()
      {
         hostTraverserBenchmark.addOneUsingParallelFor();
      };

#ifdef HAVE_CUDA
      auto cudaTraverseUsingParallelFor = [&] ()
      {
         cudaTraverserBenchmark.addOneUsingParallelFor();
      };
#endif

      if( containsValue( tests, "all" ) || containsValue( tests, "bc-parallel-for" ) )
      {
         benchmark.setOperation( "parallel for", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( "CPU", hostTraverseUsingParallelFor );
#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingParallelFor );
#endif

         benchmark.setOperation( "parallel for RST", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingParallelFor );
#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingParallelFor );
#endif
      }
// TODO: implement the benchmark (addOneUsingParallelFor does not consider BC)
//      auto hostTraverseUsingParallelFor = [&] ()
//      {
//         hostTraverserBenchmark.addOneUsingParallelFor();
//      };
//
//      auto cudaTraverseUsingParallelFor = [&] ()
//      {
//         cudaTraverserBenchmark.addOneUsingParallelFor();
//      };
//
//      if( containsValue( tests, "all" ) || containsValue( tests, "bc-parallel-for" ) )
//      {
//         benchmark.setOperation( "parallel for", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
//         if( withHost )
//            benchmark.time< Devices::Host >( "CPU", hostTraverseUsingParallelFor );
//         if( withCuda )
//            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingParallelFor );
//
//         benchmark.setOperation( "parallel for RST", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
//         if( withHost )
//            benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingParallelFor );
//         if( withCuda )
//            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingParallelFor );
//      }

      /****
       * Write one and two (as BC) using traverser
       */
      auto hostTraverseUsingTraverser = [&] ()
      {
         hostTraverserBenchmark.addOneUsingTraverser();
      };

#ifdef HAVE_CUDA
      auto cudaTraverseUsingTraverser = [&] ()
      {
         cudaTraverserBenchmark.addOneUsingTraverser();
      };
#endif

      if( containsValue( tests, "all" ) || containsValue( tests, "bc-traverser" ) )
      {
         benchmark.setOperation( "traverser", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( "CPU", hostTraverseUsingTraverser );

#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( "GPU", cudaTraverseUsingTraverser );
#endif

         benchmark.setOperation( "traverser RST", 2 * pow( ( double ) size, ( double ) Dimension ) * sizeof( Real ) / oneGB );
         if( withHost )
            benchmark.time< Devices::Host >( hostReset, "CPU", hostTraverseUsingTraverser );

#ifdef HAVE_CUDA
         if( withCuda )
            benchmark.time< Devices::Cuda >( cudaReset, "GPU", cudaTraverseUsingTraverser );
#endif
      }
   }
   return true;
}

void configSetup( Config::ConfigDescription& config )
{
   config.addList< String >( "tests", "Tests to be performed.", {"all"} );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "add-one-pure-c" );
   config.addEntryEnum( "add-one-parallel-for" );
   config.addEntryEnum( "add-one-parallel-for-and-grid-entity" );
   config.addEntryEnum( "add-one-traverser" );
   config.addEntryEnum( "bc-pure-c" );
   config.addEntryEnum( "bc-parallel-for" );
   config.addEntryEnum( "bc-traverser" );
   config.addEntry< bool >( "with-host", "Perform CPU benchmarks.", true );
#ifdef HAVE_CUDA
   config.addEntry< bool >( "with-cuda", "Perform CUDA benchmarks.", true );
#else
   config.addEntry< bool >( "with-cuda", "Perform CUDA benchmarks.", false );
#endif
   config.addEntry< bool >( "check", "Checking correct results of benchmark tests.", false );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-traversers.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );

//   config.addEntry< String >( "precision", "Precision of the arithmetics.", "double" );
//   config.addEntryEnum( "float" );
//   config.addEntryEnum( "double" );
//   config.addEntryEnum( "all" );
   config.addEntry< int >( "dimension", "Set the problem dimension. 0 means all dimensions 1,2 and 3.", 0 );
   config.addEntry< int >( "min-size", "Minimum size of arrays/vectors used in the benchmark.", 10 );
   config.addEntry< int >( "max-size", "Minimum size of arrays/vectors used in the benchmark.", 1000 );
//   config.addEntry< int >( "size-step-factor", "Factor determining the size of arrays/vectors used in the benchmark. First size is min-size and each following size is stepFactor*previousSize, up to max-size.", 2 );
   Benchmark<>::configSetup( config );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
}

template< int Dimension >
bool setupBenchmark( const Config::ParameterContainer& parameters )
{
   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
//   const String & precision = parameters.getParameter< String >( "precision" );
//   const unsigned sizeStepFactor = parameters.getParameter< unsigned >( "size-step-factor" );

   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile( logFileName, mode );

   // init benchmark and set parameters
   Benchmark<> benchmark( logFile ); //( loops, verbose );
   benchmark.setup( parameters );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   metadata["loops"] = convertToString( parameters.getParameter< int >( "loops" ) );
   metadata["reset"] = convertToString( parameters.getParameter< bool >( "reset" ) );
   metadata["minimal test time"] = convertToString( parameters.getParameter< double >( "min-time" ) );
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   runBenchmark< Dimension >( parameters, benchmark );

   return true;
}

int main( int argc, char* argv[] )
{
   Config::ConfigDescription config;
   Config::ParameterContainer parameters;

   configSetup( config );
   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) )
      return EXIT_FAILURE;

   const int dimension = parameters.getParameter< int >( "dimension" );
   bool status( false );
   if( ! dimension )
   {
      status = setupBenchmark< 1 >( parameters );
      status |= setupBenchmark< 2 >( parameters );
      status |= setupBenchmark< 3 >( parameters );
   }
   else
   {
      switch( dimension )
      {
         case 1:
            status = setupBenchmark< 1 >( parameters );
            break;
         case 2:
            status = setupBenchmark< 2 >( parameters );
            break;
         case 3:
            status = setupBenchmark< 3 >( parameters );
            break;
      }
   }
   return ! status;
}
