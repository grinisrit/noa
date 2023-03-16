// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>

namespace noa::TNL {

struct CacheSizes
{
   int L1instruction = 0;
   int L1data = 0;
   int L2 = 0;
   int L3 = 0;
};

class SystemInfo
{
public:
   static String
   getHostname();
   static String
   getArchitecture();
   static String
   getSystemName();
   static String
   getSystemRelease();
   static String
   getCurrentTime( const char* format = "%a %b %d %Y, %H:%M:%S" );

   static int
   getNumberOfProcessors();
   static String
   getOnlineCPUs();
   static int
   getNumberOfCores( int cpu_id );
   static int
   getNumberOfThreads( int cpu_id );
   static String
   getCPUModelName( int cpu_id );
   static int
   getCPUMaxFrequency( int cpu_id );
   static CacheSizes
   getCPUCacheSizes( int cpu_id );
   static size_t
   getFreeMemory();

protected:
   struct CPUInfo
   {
      int numberOfProcessors = 0;
      String CPUModelName;
      int CPUThreads = 0;
      int CPUCores = 0;
   };

   static CPUInfo
   parseCPUInfo();

   template< typename ResultType >
   static ResultType
   readFile( const String& fileName )
   {
      std::ifstream file( fileName.getString() );
      if( ! file ) {
         std::cerr << "Unable to read information from " << fileName << "." << std::endl;
         return 0;  // NOLINT(modernize-use-nullptr)
      }
      ResultType result;
      file >> result;
      return result;
   }
};

}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/SystemInfo.hpp>
