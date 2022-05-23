// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include "Wrappers.h"
#include "Utils.h"

namespace noa::TNL {
namespace MPI {

// defined here rather than in Wrappers.h to break cyclic header inclusion (due to selectGPU.h)
inline void
Init( int& argc, char**& argv, int required_thread_level = MPI_THREAD_SINGLE )
{
#ifdef HAVE_MPI
   switch( required_thread_level ) {
      case MPI_THREAD_SINGLE:      // application is single-threaded
      case MPI_THREAD_FUNNELED:    // application is multithreaded, but all MPI calls will be issued from the master thread only
      case MPI_THREAD_SERIALIZED:  // application is multithreaded and any thread may issue MPI calls, but different threads
                                   // will never issue MPI calls at the same time
      case MPI_THREAD_MULTIPLE:    // application is multithreaded and any thread may issue MPI calls at any time
         break;
      default:
         std::cerr << "ERROR: invalid argument for the 'required' thread level support: " << required_thread_level << std::endl;
         MPI_Abort( MPI_COMM_WORLD, 1 );
   }

   int provided;
   MPI_Init_thread( &argc, &argv, required_thread_level, &provided );
   if( provided < required_thread_level ) {
      const char* level = "";
      switch( required_thread_level ) {
         case MPI_THREAD_SINGLE:
            level = "MPI_THREAD_SINGLE";
            break;
         case MPI_THREAD_FUNNELED:
            level = "MPI_THREAD_FUNNELED";
            break;
         case MPI_THREAD_SERIALIZED:
            level = "MPI_THREAD_SERIALIZED";
            break;
         case MPI_THREAD_MULTIPLE:
            level = "MPI_THREAD_MULTIPLE";
            break;
      }
      std::cerr << "ERROR: The MPI library does not have the required level of thread support: " << level << std::endl;
      MPI_Abort( MPI_COMM_WORLD, 1 );
   }

   selectGPU();
#endif
}

inline void
Finalize()
{
#ifdef HAVE_MPI
   MPI_Finalize();
#endif
}

struct ScopedInitializer
{
   ScopedInitializer( int& argc, char**& argv, int required_thread_level = MPI_THREAD_SINGLE )
   {
      Init( argc, argv, required_thread_level );
   }

   ~ScopedInitializer()
   {
      restoreRedirection();
      Finalize();
   }
};

}  // namespace MPI
}  // namespace noa::TNL
