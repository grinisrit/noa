// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <limits>

namespace noa::TNL {
namespace Debugging {

/*
 * Prints memory usage of the current process into the specified stream.
 *
 * The information is obtained from /proc/self/status, which is assumed to be
 * present on the system. The meaning of the printed values is following:
 *  
 *  - VmSize: Virtual memory size.
 *  - VmRSS: Resident set size.
 *  - VmHWM: Peak resident set size ("high water mark").
 *
 * See the proc(5) manual on Linux for details.
 */
static void
printMemoryUsage( std::ostream& str = std::cerr )
{
   std::ifstream meminfo("/proc/self/status");
   if( meminfo.fail() ) {
      std::cerr << "error: unable to open /proc/self/status" << std::endl;
      return;
   }

   unsigned vm = 0;
   unsigned rss = 0;
   unsigned hwm = 0;

   std::string desc;
   while( meminfo.good() ) {
       // extract description (first column)
       meminfo >> desc;

       if( desc == "VmSize:" )
           meminfo >> vm;
       if( desc == "VmHWM:" )
           meminfo >> hwm;
       if( desc == "VmRSS:" )
           meminfo >> rss;

       // ignore the rest of irrelevant lines
       meminfo.ignore( std::numeric_limits< std::streamsize >::max(), '\n' );
   }

   str << "Memory usage (MiB): "
       << "VmSize = " << vm / 1024 << "MiB, "
       << "VmRSS = " << rss / 1024 << "MiB, "
       << "VmHWM = " << hwm / 1024 << "MiB, "
       << std::endl;
}

} // namespace Debugging
} // namespace noa::TNL
