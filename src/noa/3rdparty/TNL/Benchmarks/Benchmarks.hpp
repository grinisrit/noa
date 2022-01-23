// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include "Benchmarks.h"
#include "Utils.h"

#include <iostream>
#include <exception>

namespace noa::TNL {
namespace Benchmarks {


template< typename Logger >
Benchmark< Logger >::
Benchmark( std::ostream& output, int loops, bool verbose )
: logger(output, verbose), loops(loops)
{}

template< typename Logger >
void
Benchmark< Logger >::
configSetup( Config::ConfigDescription& config )
{
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< bool >( "reset", "Call reset function between loops.", true );
   config.addEntry< double >( "min-time", "Minimal real time in seconds for every computation.", 0.0 );
   config.addEntry< int >( "verbose", "Verbose mode, the higher number the more verbosity.", 1 );
}

template< typename Logger >
void
Benchmark< Logger >::
setup( const Config::ParameterContainer& parameters )
{
   this->loops = parameters.getParameter< int >( "loops" );
   this->reset = parameters.getParameter< bool >( "reset" );
   this->minTime = parameters.getParameter< double >( "min-time" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   logger.setVerbose( verbose );
}

template< typename Logger >
void
Benchmark< Logger >::
setLoops( int loops )
{
   this->loops = loops;
}

template< typename Logger >
void
Benchmark< Logger >::
setMinTime( double minTime )
{
   this->minTime = minTime;
}

template< typename Logger >
bool
Benchmark< Logger >::
isResetingOn() const
{
   return reset;
}

template< typename Logger >
void
Benchmark< Logger >::
setMetadataColumns( const MetadataColumns & metadata )
{
   logger.setMetadataColumns( metadata );
}

template< typename Logger >
void
Benchmark< Logger >::
setMetadataElement( const typename MetadataColumns::value_type & element )
{
   logger.setMetadataElement( element );
}

template< typename Logger >
void
Benchmark< Logger >::
setMetadataWidths( const std::map< std::string, int > & widths )
{
   logger.setMetadataWidths( widths );
}

template< typename Logger >
void
Benchmark< Logger >::
setDatasetSize( const double datasetSize,
                const double baseTime )
{
   this->datasetSize = datasetSize;
   this->baseTime = baseTime;
}

template< typename Logger >
void
Benchmark< Logger >::
setOperation( const String & operation,
              const double datasetSize,
              const double baseTime )
{
   monitor.setStage( operation.getString() );
   logger.setMetadataElement( {"operation", operation}, 0 );
   setDatasetSize( datasetSize, baseTime );
}

template< typename Logger >
   template< typename Device,
             typename ResetFunction,
             typename ComputeFunction >
void
Benchmark< Logger >::
time( ResetFunction reset,
      const String & performer,
      ComputeFunction & compute,
      BenchmarkResult & result )
{
   result.time = std::numeric_limits<double>::quiet_NaN();
   result.stddev = std::numeric_limits<double>::quiet_NaN();

   // run the monitor main loop
   Solvers::SolverMonitorThread monitor_thread( monitor );
   if( logger.getVerbose() <= 1 )
      // stop the main loop when not verbose
      monitor.stopMainLoop();

   std::string errorMessage;
   try {
      if( this->reset )
         std::tie( result.loops, result.time, result.stddev ) = timeFunction< Device >( compute, reset, loops, minTime, monitor );
      else {
         auto noReset = [] () {};
         std::tie( result.loops, result.time, result.stddev ) = timeFunction< Device >( compute, noReset, loops, minTime, monitor );
      }
   }
   catch ( const std::exception& e ) {
      errorMessage = "timeFunction failed due to a C++ exception with description: " + std::string(e.what());
      std::cerr << errorMessage << std::endl;
   }

   result.bandwidth = datasetSize / result.time;
   result.speedup = this->baseTime / result.time;
   if( this->baseTime == 0.0 )
      this->baseTime = result.time;

   logger.logResult( performer, result.getTableHeader(), result.getRowElements(), result.getColumnWidthHints(), errorMessage );
}

template< typename Logger >
   template< typename Device,
             typename ResetFunction,
             typename ComputeFunction >
BenchmarkResult
Benchmark< Logger >::
time( ResetFunction reset,
      const String& performer,
      ComputeFunction& compute )
{
   BenchmarkResult result;
   time< Device >( reset, performer, compute, result );
   return result;
}

template< typename Logger >
   template< typename Device,
             typename ComputeFunction >
void
Benchmark< Logger >::
time( const String & performer,
      ComputeFunction & compute,
      BenchmarkResult & result )
{
   auto noReset = [] () {};
   time< Device >( noReset, performer, compute, result );
}

template< typename Logger >
   template< typename Device,
             typename ComputeFunction >
BenchmarkResult
Benchmark< Logger >::
time( const String & performer,
      ComputeFunction & compute )
{
   BenchmarkResult result;
   time< Device >( performer, compute, result );
   return result;
}

template< typename Logger >
void
Benchmark< Logger >::
addErrorMessage( const std::string& message )
{
   logger.writeErrorMessage( message );
   std::cerr << message << std::endl;
}

template< typename Logger >
auto
Benchmark< Logger >::
getMonitor() -> SolverMonitorType&
{
   return monitor;
}

template< typename Logger >
double
Benchmark< Logger >::
getBaseTime() const
{
   return baseTime;
}

} // namespace Benchmarks
} // namespace noa::TNL
