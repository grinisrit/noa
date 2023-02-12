// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

#ifdef HAVE_OPENMP
   #include <omp.h>
#endif

namespace noa::TNL {
namespace Devices {

class Host
{
public:
   //! For compatibility with \ref TNL::Devices::Cuda only. In the future, it may be used to specify parameters for OpenMP
   //! execution.
   struct LaunchConfiguration
   {};

   static void
   disableOMP()
   {
      ompEnabled() = false;
   }

   static void
   enableOMP()
   {
      ompEnabled() = true;
   }

   static inline bool
   isOMPEnabled()
   {
#ifdef HAVE_OPENMP
      return ompEnabled();
#else
      return false;
#endif
   }

   static void
   setMaxThreadsCount( int maxThreadsCount_ )
   {
      maxThreadsCount() = maxThreadsCount_;
#ifdef HAVE_OPENMP
      omp_set_num_threads( maxThreadsCount() );
#endif
   }

   static int
   getMaxThreadsCount()
   {
#ifdef HAVE_OPENMP
      if( maxThreadsCount() == -1 )
         return omp_get_max_threads();
      return maxThreadsCount();
#else
      return 0;
#endif
   }

   static int
   getThreadIdx()
   {
#ifdef HAVE_OPENMP
      return omp_get_thread_num();
#else
      return 0;
#endif
   }

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" )
   {
#ifdef HAVE_OPENMP
      config.addEntry< bool >( prefix + "openmp-enabled", "Enable support of OpenMP.", true );
      config.addEntry< int >( prefix + "openmp-max-threads", "Set maximum number of OpenMP threads.", omp_get_max_threads() );
#else
      config.addEntry< bool >( prefix + "openmp-enabled", "Enable support of OpenMP (not supported on this system).", false );
      config.addEntry< int >(
         prefix + "openmp-max-threads", "Set maximum number of OpenMP threads (not supported on this system).", 0 );
#endif
   }

   static bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" )
   {
      if( parameters.getParameter< bool >( prefix + "openmp-enabled" ) ) {
#ifdef HAVE_OPENMP
         enableOMP();
#else
         std::cerr << "OpenMP is not supported - please recompile the TNL library with OpenMP." << std::endl;
         return false;
#endif
      }
      else
         disableOMP();
      const int threadsCount = parameters.getParameter< int >( prefix + "openmp-max-threads" );
      if( threadsCount > 1 && ! isOMPEnabled() )
         std::cerr << "Warning: openmp-max-threads was set to " << threadsCount << ", but OpenMP is disabled." << std::endl;
      setMaxThreadsCount( threadsCount );
      return true;
   }

protected:
   static bool&
   ompEnabled()
   {
#ifdef HAVE_OPENMP
      static bool ompEnabled = true;
#else
      static bool ompEnabled = false;
#endif
      return ompEnabled;
   }

   static int&
   maxThreadsCount()
   {
      static int maxThreadsCount = -1;
      return maxThreadsCount;
   }
};

}  // namespace Devices
}  // namespace noa::TNL
