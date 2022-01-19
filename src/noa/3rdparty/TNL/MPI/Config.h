// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#ifdef HAVE_MPI
#ifdef OMPI_MAJOR_VERSION
   // header specific to OpenMPI (needed for CUDA-aware detection)
   #include <mpi-ext.h>
#endif

#include <unistd.h>  // getpid
#endif

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include "Utils.h"

namespace TNL {
namespace MPI {

inline void configSetup( Config::ConfigDescription& config, const String& prefix = "" )
{
#ifdef HAVE_MPI
   config.addEntry< bool >( "redirect-mpi-output", "Only process with rank 0 prints to console. Other processes are redirected to files.", true );
   config.addEntry< String >( "redirect-mpi-output-dir", "Directory where ranks will store the files if their output is redirected.", "." );
   config.addEntry< bool >( "mpi-gdb-debug", "Wait for GDB to attach the master MPI process.", false );
   config.addEntry< int >( "mpi-process-to-attach", "Number of the MPI process to be attached by GDB. Set -1 for all processes.", 0 );
#endif
}

inline bool setup( const Config::ParameterContainer& parameters,
                   const String& prefix = "" )
{
#ifdef HAVE_MPI
   if( Initialized() && ! Finalized() )
   {
      const bool redirect = parameters.getParameter< bool >( "redirect-mpi-output" );
      const String outputDirectory = parameters.getParameter< String >( "redirect-mpi-output-dir" );
      if( redirect )
         MPI::setupRedirection( outputDirectory );
#ifdef HAVE_CUDA
      if( GetSize() > 1 )
      {
#if defined( MPIX_CUDA_AWARE_SUPPORT ) && MPIX_CUDA_AWARE_SUPPORT
         std::cout << "CUDA-aware MPI detected on this system ... " << std::endl;
#elif defined( MPIX_CUDA_AWARE_SUPPORT ) && !MPIX_CUDA_AWARE_SUPPORT
         std::cerr << "MPI is not CUDA-aware. Please install correct version of MPI." << std::endl;
         return false;
#else
         std::cerr << "WARNING: TNL cannot detect if you have CUDA-aware MPI. Some problems may occur." << std::endl;
#endif
      }
#endif // HAVE_CUDA
      bool gdbDebug = parameters.getParameter< bool >( "mpi-gdb-debug" );
      int processToAttach = parameters.getParameter< int >( "mpi-process-to-attach" );

      if( gdbDebug )
      {
         int rank = GetRank( MPI_COMM_WORLD );
         int pid = getpid();

         volatile int tnlMPIDebugAttached = 0;
         MPI_Send( &pid, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
         MPI_Barrier( MPI_COMM_WORLD );
         if( rank == 0 )
         {
            std::cout << "Attach GDB to MPI process(es) by entering:" << std::endl;
            for( int i = 0; i < GetSize(); i++ )
            {
               MPI_Status status;
               int recvPid;
               MPI_Recv( &recvPid, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status );

               if( i == processToAttach || processToAttach == -1 )
               {
                  std::cout << "  For MPI process " << i << ": gdb -q -ex \"attach " << recvPid << "\""
                            << " -ex \"set variable tnlMPIDebugAttached=1\""
                            << " -ex \"continue\"" << std::endl;
               }
            }
            std::cout << std::flush;
         }
         if( rank == processToAttach || processToAttach == -1 )
            while( ! tnlMPIDebugAttached );
         MPI_Barrier( MPI_COMM_WORLD );
      }
   }
#endif // HAVE_MPI
   return true;
}

} // namespace MPI
} // namespace TNL
