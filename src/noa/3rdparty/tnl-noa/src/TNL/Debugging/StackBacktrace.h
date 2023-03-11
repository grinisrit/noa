// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <memory>

#ifndef _WIN32
   #include <execinfo.h>
#endif

#include <noa/3rdparty/tnl-noa/src/TNL/TypeInfo.h>  // TNL::detail::demangle

namespace noa::TNL {
namespace Debugging {

/*
 * Print a demangled stack backtrace of the caller function to FILE* out.
 *
 * Reference: http://panthema.net/2008/0901-stacktrace-demangled/
 *
 * Note that the program must be linked with the -rdynamic flag, otherwise
 * demangling will not work.
 */
static void
printStackBacktrace( std::ostream& out = std::cout, unsigned int max_frames = 64 )
{
#ifndef _WIN32
   out << "stack trace:\n";

   // storage array for stack trace address data
   std::unique_ptr< void* > addrlist{ new void*[ max_frames ] };

   // retrieve current stack addresses
   int addrlen = backtrace( addrlist.get(), max_frames );

   if( addrlen == 0 ) {
      out << "  <empty, possibly corrupt>" << std::endl;
      return;
   }

   // resolve addresses into strings containing "filename(function+address)",
   // this array must be free()-ed
   std::unique_ptr< char*, void ( * )( void* ) > symbollist( backtrace_symbols( addrlist.get(), addrlen ), std::free );

   // iterate over the returned symbol lines. skip the first, it is the
   // address of this function.
   for( int i = 1; i < addrlen; i++ ) {
      char* begin_name = nullptr;
      char* begin_offset = nullptr;
      char* end_offset = nullptr;

      // find parentheses and +address offset surrounding the mangled name:
      // ./module(function+0x15c) [0x8048a6d]
      for( char* p = symbollist.get()[ i ]; *p != '\0'; ++p ) {
         if( *p == '(' )
            begin_name = p;
         else if( *p == '+' )
            begin_offset = p;
         else if( *p == ')' && begin_offset != nullptr ) {
            end_offset = p;
            break;
         }
      }

      if( begin_name != nullptr && begin_offset != nullptr && end_offset != nullptr && begin_name < begin_offset ) {
         *begin_name++ = '\0';
         *begin_offset++ = '\0';
         *end_offset = '\0';

         // mangled name is now in [begin_name, begin_offset) and caller
         // offset in [begin_offset, end_offset).
         std::cout << "  " << i << " " << symbollist.get()[ i ] << " : " << TNL::detail::demangle( begin_name ) << "+"
                   << begin_offset << "\n";
      }
      else {
         // couldn't parse the line? print the whole line.
         std::cout << "  " << i << " " << symbollist.get()[ i ] << "\n";
      }
   }
#else
   out << "stack trace: [not supported on Windows]\n";
#endif
}

}  // namespace Debugging
}  // namespace noa::TNL
