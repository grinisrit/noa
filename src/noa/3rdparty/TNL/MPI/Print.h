// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include <noa/3rdparty/TNL/String.h>
#include <noa/3rdparty/TNL/MPI/Utils.h>

#ifdef HAVE_MPI
#define TNL_MPI_PRINT( message )                                                                                                 \
if( ! noa::TNL::MPI::Initialized() || noa::TNL::MPI::Finalized() )                                                                         \
   std::cerr << message << std::endl;                                                                                            \
else                                                                                                                             \
{                                                                                                                                \
   if( noa::TNL::MPI::GetRank() > 0 )                                                                                                 \
   {                                                                                                                             \
      std::stringstream __tnl_mpi_print_stream_;                                                                                 \
      __tnl_mpi_print_stream_ << "Node " << noa::TNL::MPI::GetRank() << " of " << noa::TNL::MPI::GetSize() << " : "                        \
                              << message << std::endl;                                                                           \
      noa::TNL::String __tnl_mpi_print_string_( __tnl_mpi_print_stream_.str() );                                                      \
      noa::TNL::MPI::send( __tnl_mpi_print_string_, 0, std::numeric_limits< int >::max() );                                           \
   }                                                                                                                             \
   else                                                                                                                          \
   {                                                                                                                             \
      std::cerr << "Node 0 of " << noa::TNL::MPI::GetSize() << " : " << message << std::endl;                                         \
      for( int __tnl_mpi_print_j = 1; __tnl_mpi_print_j < noa::TNL::MPI::GetSize(); __tnl_mpi_print_j++ )                             \
      {                                                                                                                          \
         noa::TNL::String __tnl_mpi_print_string_;                                                                                    \
         noa::TNL::MPI::recv( __tnl_mpi_print_string_, __tnl_mpi_print_j, std::numeric_limits< int >::max() );                        \
         std::cerr << __tnl_mpi_print_string_;                                                                                   \
      }                                                                                                                          \
   }                                                                                                                             \
}
#else
#define TNL_MPI_PRINT( message )                                                                                                 \
   std::cerr << message << std::endl;
#endif

#ifdef HAVE_MPI
#define TNL_MPI_PRINT_MASTER( message )                                                                                          \
if( ! noa::TNL::MPI::Initialized() || noa::TNL::MPI::Finalized() )                                                                         \
   std::cerr << message << std::endl;                                                                                            \
else                                                                                                                             \
{                                                                                                                                \
   if( noa::TNL::MPI::GetRank() == 0 )                                                                     \
   {                                                                                                                             \
      std::cerr << "Master node : " << message << std::endl;                                                                     \
   }                                                                                                                             \
}
#else
#define TNL_MPI_PRINT_MASTER( message )                                                                                          \
   std::cerr << message << std::endl;
#endif

#ifdef HAVE_MPI
#define TNL_MPI_PRINT_COND( condition, message )                                                                                 \
if( ! noa::TNL::MPI::Initialized() || noa::TNL::MPI::Finalized() )                                                                         \
{                                                                                                                                \
   if( condition) std::cerr << message << std::endl;                                                                             \
}                                                                                                                                \
else                                                                                                                             \
{                                                                                                                                \
   if( noa::TNL::MPI::GetRank() > 0 )                                                                                                 \
   {                                                                                                                             \
      int __tnl_mpi_print_cnd = ( condition );                                                                                   \
      noa::TNL::MPI::Send( &__tnl_mpi_print_cnd, 1, 0, 0 );                                                                           \
      if( condition ) {                                                                                                          \
         std::stringstream __tnl_mpi_print_stream_;                                                                              \
         __tnl_mpi_print_stream_ << "Node " << noa::TNL::MPI::GetRank() << " of " << noa::TNL::MPI::GetSize() << " : "                     \
                                 << message << std::endl;                                                                        \
         noa::TNL::String __tnl_mpi_print_string_( __tnl_mpi_print_stream_.str() );                                                   \
         noa::TNL::MPI::send( __tnl_mpi_print_string_, 0, std::numeric_limits< int >::max() );                                        \
      }                                                                                                                          \
   }                                                                                                                             \
   else                                                                                                                          \
   {                                                                                                                             \
      if( condition )                                                                                                            \
         std::cerr << "Node 0 of " << noa::TNL::MPI::GetSize() << " : " << message << std::endl;                                      \
      for( int __tnl_mpi_print_j = 1; __tnl_mpi_print_j < noa::TNL::MPI::GetSize(); __tnl_mpi_print_j++ )                             \
         {                                                                                                                       \
            int __tnl_mpi_print_cond;                                                                                            \
            noa::TNL::MPI::Recv( &__tnl_mpi_print_cond, 1, __tnl_mpi_print_j, 0 );                                                    \
            if( __tnl_mpi_print_cond )                                                                                           \
            {                                                                                                                    \
               noa::TNL::String __tnl_mpi_print_string_;                                                                              \
               noa::TNL::MPI::recv( __tnl_mpi_print_string_, __tnl_mpi_print_j, std::numeric_limits< int >::max() );                  \
               std::cerr << __tnl_mpi_print_string_;                                                                             \
            }                                                                                                                    \
         }                                                                                                                       \
   }                                                                                                                             \
}
#else
#define TNL_MPI_PRINT_COND( condition, message )                                                                                 \
   std::cerr << message << std::endl;
#endif
