// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#if defined( __APPLE__ ) && defined( __MACH__ )
   #include <fenv.h>
#else
   #include <cfenv>
#endif
#include <csignal>

#include <noa/3rdparty/tnl-noa/src/TNL/Debugging/StackBacktrace.h>

namespace noa::TNL {
namespace Debugging {

static void
printStackBacktraceAndAbort( int sig = 0 )
{
   if( sig == SIGSEGV )
      fprintf( stderr, "Invalid memory reference, printing backtrace and aborting...\n" );
   else if( sig == SIGFPE ) {
      /*
       * Unfortunately it is not possible to get the floating-point exception type
       * from a signal handler. Otherwise, it would be done this way:
       *
       *    fprintf(stderr, "Floating-point exception");
       *    if(fetestexcept(FE_DIVBYZERO))  fprintf(stderr, " FE_DIVBYZERO");
       *    if(fetestexcept(FE_INEXACT))    fprintf(stderr, " FE_INEXACT");
       *    if(fetestexcept(FE_INVALID))    fprintf(stderr, " FE_INVALID");
       *    if(fetestexcept(FE_OVERFLOW))   fprintf(stderr, " FE_OVERFLOW");
       *    if(fetestexcept(FE_UNDERFLOW))  fprintf(stderr, " FE_UNDERFLOW");
       *    fprintf(stderr, " occurred, printing backtrace and aborting...\n");
       */
      fprintf( stderr, "Floating-point exception occurred, printing backtrace and aborting...\n" );
   }
   else
      fprintf( stderr, "Aborting due to signal %d...\n", sig );
   printStackBacktrace();
   // TODO: maybe use MPI_Abort(MPI_COMM_WORLD, 1); if we can detect we run under MPI
   abort();
}

#if defined( __APPLE__ ) && defined( __MACH__ )
// https://stackoverflow.com/questions/69059981/how-to-trap-floating-point-exceptions-on-m1-macs

static void
fpe_signal_handler( int sig, siginfo_t* sip, void* scp )
{
   int fe_code = sip->si_code;

   printf( "In signal handler : " );

   if( fe_code == ILL_ILLTRP )
      printf( "Illegal trap detected\n" );
   else
      printf( "Code detected : %d\n", fe_code );

   printStackBacktraceAndAbort( sig );
}
#endif

/*
 * Registers handler for SIGSEGV and SIGFPE signals and enables conversion of
 * floating-point exceptions into SIGFPE. This is useful e.g. for tracing where
 * NANs occurred. Example usage:
 *
 * int main()
 * {
 *    #ifndef NDEBUG
 *       TNL::Debugging::trackFloatingPointExceptions()
 *    #endif
 *    [start some computation here...]
 * }
 */
static void
trackFloatingPointExceptions()
{
#if defined( __APPLE__ ) && defined( __MACH__ )
   fenv_t env;
   fegetenv( &env );

   env.__fpcr = env.__fpcr | __fpcr_trap_invalid;
   fesetenv( &env );

   struct sigaction act;
   act.sa_sigaction = fpe_signal_handler;
   sigemptyset( &act.sa_mask );
   act.sa_flags = SA_SIGINFO;
   sigaction( SIGILL, &act, NULL );
#else
   signal( SIGSEGV, printStackBacktraceAndAbort );
   signal( SIGFPE, printStackBacktraceAndAbort );
   // TODO: find a workaround for Windows, e.g. https://stackoverflow.com/a/30175525
   #ifndef _WIN32
   feenableexcept( FE_ALL_EXCEPT & ~FE_INEXACT );
   #endif
#endif
}

}  // namespace Debugging
}  // namespace noa::TNL
