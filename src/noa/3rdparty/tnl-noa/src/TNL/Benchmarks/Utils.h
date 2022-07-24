// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include <tuple>
#include <map>
#include <fstream>
#include <filesystem>

#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/IterativeSolverMonitor.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>
#include <noa/3rdparty/tnl-noa/src/TNL/SystemInfo.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/DeviceInfo.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Wrappers.h>

namespace noa::TNL {
namespace Benchmarks {

// returns a tuple of (loops, mean, stddev) where loops is the number of
// performed loops (i.e. timing samples), mean is the arithmetic mean of the
// computation times and stddev is the sample standard deviation
template< typename Device,
          typename ComputeFunction,
          typename ResetFunction,
          typename Monitor = TNL::Solvers::IterativeSolverMonitor< double, int > >
std::tuple< int, double, double >
timeFunction( ComputeFunction compute, ResetFunction reset, int maxLoops, const double& minTime, Monitor&& monitor = Monitor() )
{
   // the timer is constructed zero-initialized and stopped
   Timer timer;

   // set timer to the monitor
   monitor.setTimer( timer );

   // warm up
   reset();
   compute();

   Containers::Vector< double > results( maxLoops );
   results.setValue( 0.0 );

   int loops;
   for( loops = 0; loops < maxLoops || sum( results ) < minTime; loops++ ) {
      // abuse the monitor's "time" for loops
      monitor.setTime( loops + 1 );
      reset();

      // Explicit synchronization of the CUDA device
#ifdef HAVE_CUDA
      if( std::is_same< Device, Devices::Cuda >::value )
         cudaDeviceSynchronize();
#endif

      // reset timer before each computation
      timer.reset();
      timer.start();
      compute();
#ifdef HAVE_CUDA
      if( std::is_same< Device, Devices::Cuda >::value )
         cudaDeviceSynchronize();
#endif
      timer.stop();

      results[ loops ] = timer.getRealTime();
   }

   const double mean = sum( results ) / (double) loops;
   double stddev;
   if( loops > 1 )
      stddev = 1.0 / std::sqrt( loops - 1 ) * l2Norm( results - mean );
   else
      stddev = std::numeric_limits< double >::quiet_NaN();
   return std::make_tuple( loops, mean, stddev );
}

inline std::map< std::string, std::string >
getHardwareMetadata()
{
   const int cpu_id = 0;
   const CacheSizes cacheSizes = SystemInfo::getCPUCacheSizes( cpu_id );
   const std::string cacheInfo = std::to_string( cacheSizes.L1data ) + ", " + std::to_string( cacheSizes.L1instruction ) + ", "
                               + std::to_string( cacheSizes.L2 ) + ", " + std::to_string( cacheSizes.L3 );
#ifdef HAVE_CUDA
   const int activeGPU = Cuda::DeviceInfo::getActiveDevice();
   const std::string deviceArch = std::to_string( Cuda::DeviceInfo::getArchitectureMajor( activeGPU ) ) + "."
                                + std::to_string( Cuda::DeviceInfo::getArchitectureMinor( activeGPU ) );
#endif

#ifdef HAVE_MPI
   int nproc = 1;
   // check if MPI was initialized (some benchmarks do not initialize MPI even when
   // they are built with HAVE_MPI and thus MPI::GetSize() cannot be used blindly)
   if( TNL::MPI::Initialized() )
      nproc = TNL::MPI::GetSize();
#endif

   std::map< std::string, std::string > metadata{
      { "host name", SystemInfo::getHostname() },
      { "architecture", SystemInfo::getArchitecture() },
      { "system", SystemInfo::getSystemName() },
      { "system release", SystemInfo::getSystemRelease() },
      { "start time", SystemInfo::getCurrentTime() },
#ifdef HAVE_MPI
      { "number of MPI processes", std::to_string( nproc ) },
#endif
      { "OpenMP enabled", Devices::Host::isOMPEnabled() ? "yes" : "no" },
      { "OpenMP threads", std::to_string( Devices::Host::getMaxThreadsCount() ) },
      { "CPU model name", SystemInfo::getCPUModelName( cpu_id ) },
      { "CPU cores", std::to_string( SystemInfo::getNumberOfCores( cpu_id ) ) },
      { "CPU threads per core",
        std::to_string( SystemInfo::getNumberOfThreads( cpu_id ) / SystemInfo::getNumberOfCores( cpu_id ) ) },
      { "CPU max frequency (MHz)", std::to_string( SystemInfo::getCPUMaxFrequency( cpu_id ) / 1e3 ) },
      { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
#ifdef HAVE_CUDA
      { "GPU name", Cuda::DeviceInfo::getDeviceName( activeGPU ) },
      { "GPU architecture", deviceArch },
      { "GPU CUDA cores", std::to_string( Cuda::DeviceInfo::getCudaCores( activeGPU ) ) },
      { "GPU clock rate (MHz)", std::to_string( (double) Cuda::DeviceInfo::getClockRate( activeGPU ) / 1e3 ) },
      { "GPU global memory (GB)", std::to_string( (double) Cuda::DeviceInfo::getGlobalMemory( activeGPU ) / 1e9 ) },
      { "GPU memory clock rate (MHz)", std::to_string( (double) Cuda::DeviceInfo::getMemoryClockRate( activeGPU ) / 1e3 ) },
      { "GPU memory ECC enabled", std::to_string( Cuda::DeviceInfo::getECCEnabled( activeGPU ) ) },
#endif
   };

   return metadata;
}

inline void
writeMapAsJson( const std::map< std::string, std::string >& data, std::ostream& out )
{
   out << "{\n";
   for( auto it = data.begin(); it != data.end(); ) {
      out << "\t\"" << it->first << "\": \"" << it->second << "\"";
      // increment the iterator now to peek at the next element
      it++;
      // write a comma if there are still elements remaining
      if( it != data.end() )
         out << ",";
      out << "\n";
   }
   out << "}\n" << std::flush;
}

inline void
writeMapAsJson( const std::map< std::string, std::string >& data, std::string filename, const std::string& newExtension = "" )
{
   namespace fs = std::filesystem;

   if( ! newExtension.empty() ) {
      const fs::path oldPath = filename;
      const fs::path newPath = oldPath.parent_path() / ( oldPath.stem().string() + newExtension );
      filename = newPath;
   }

   std::ofstream file( filename );
   // enable exceptions
   file.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
   writeMapAsJson( data, file );
}

}  // namespace Benchmarks
}  // namespace noa::TNL
