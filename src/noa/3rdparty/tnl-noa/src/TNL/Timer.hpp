// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Logger.h>

// check if we are on a POSIX system or Windows,
// see https://stackoverflow.com/a/4575466
#if ! defined( _WIN32 ) && ! defined( _WIN64 )
   #include <sys/resource.h>
#endif

namespace noa::TNL {

inline Timer::Timer()
{
   reset();
}

inline void
Timer::reset()
{
   this->initialCPUTime = 0;
   this->totalCPUTime = 0.0;
   this->initialRealTime = TimePoint();
   this->totalRealTime = Duration();
   this->initialCPUCycles = 0;
   this->totalCPUCycles = 0;
   this->stopState = true;
}

inline void
Timer::stop()
{
   if( ! this->stopState ) {
      this->totalRealTime += readRealTime() - this->initialRealTime;
      this->totalCPUTime += readCPUTime() - this->initialCPUTime;
      this->totalCPUCycles += readCPUCycles() - this->initialCPUCycles;
      this->stopState = true;
   }
}

inline void
Timer::start()
{
   this->initialRealTime = readRealTime();
   this->initialCPUTime = readCPUTime();
   this->initialCPUCycles = readCPUCycles();
   this->stopState = false;
}

inline double
Timer::getRealTime() const
{
   if( ! this->stopState )
      return durationToDouble( readRealTime() - this->initialRealTime );
   return durationToDouble( this->totalRealTime );
}

inline double
Timer::getCPUTime() const
{
   if( ! this->stopState )
      return readCPUTime() - this->initialCPUTime;
   return this->totalCPUTime;
}

inline unsigned long long int
Timer::getCPUCycles() const
{
   if( ! this->stopState )
      return readCPUCycles() - this->initialCPUCycles;
   return this->totalCPUCycles;
}

inline bool
Timer::writeLog( Logger& logger, int logLevel ) const
{
   logger.writeParameter< double >( "Real time:", this->getRealTime(), logLevel );
   logger.writeParameter< double >( "CPU time:", this->getCPUTime(), logLevel );
   logger.writeParameter< unsigned long long int >( "CPU Cycles:", this->getCPUCycles(), logLevel );
   return true;
}

inline typename Timer::TimePoint
Timer::readRealTime()
{
   return std::chrono::high_resolution_clock::now();
}

inline double
Timer::readCPUTime()
{
#if ! defined( _WIN32 ) && ! defined( _WIN64 )
   rusage initUsage;
   getrusage( RUSAGE_SELF, &initUsage );
   return initUsage.ru_utime.tv_sec + 1.0e-6 * (double) initUsage.ru_utime.tv_usec;
#else
   return -1;
#endif
}

inline unsigned long long int
Timer::readCPUCycles()
{
   return rdtsc();
}

inline double
Timer::durationToDouble( const Duration& duration )
{
   std::chrono::duration< double > dur( duration );
   return dur.count();
}

inline unsigned long long
Timer::rdtsc()
{
   unsigned hi;
   unsigned lo;
   __asm__ __volatile__( "rdtsc" : "=a"( lo ), "=d"( hi ) );
   return ( (unsigned long long) lo ) | ( ( (unsigned long long) hi ) << 32 );
}

}  // namespace noa::TNL
