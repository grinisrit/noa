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
if( ! noaTNL::MPI::Initialized() || noaTNL::MPI::Finalized() )                                                                         \
   std::cerr << message << std::endl;                                                                                            \
else                                                                                                                             \
{                                                                                                                                \
   if( noaTNL::MPI::GetRank() > 0 )                                                                                                 \
   {                                                                                                                             \
      std::stringstream __tnl_mpi_print_stream_;                                                                                 \
      __tnl_mpi_print_stream_ << "Node " << noaTNL::MPI::GetRank() << " of " << noaTNL::MPI::GetSize() << " : "                        \
                              << message << std::endl;                                                                           \
      noaTNL::String __tnl_mpi_print_string_( __tnl_mpi_print_stream_.str() );                                                      \
      noaTNL::MPI::send( __tnl_mpi_print_string_, 0, std::numeric_limits< int >::max() );                                           \
   }                                                                                                                             \
   else                                                                                                                          \
   {                                                                                                                             \
      std::cerr << "Node 0 of " << noaTNL::MPI::GetSize() << " : " << message << std::endl;                                         \
      for( int __tnl_mpi_print_j = 1; __tnl_mpi_print_j < noaTNL::MPI::GetSize(); __tnl_mpi_print_j++ )                             \
      {                                                                                                                          \
         noaTNL::String __tnl_mpi_print_string_;                                                                                    \
         noaTNL::MPI::recv( __tnl_mpi_print_string_, __tnl_mpi_print_j, std::numeric_limits< int >::max() );                        \
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
if( ! noaTNL::MPI::Initialized() || noaTNL::MPI::Finalized() )                                                                         \
   std::cerr << message << std::endl;                                                                                            \
else                                                                                                                             \
{                                                                                                                                \
   if( noaTNL::MPI::GetRank() == 0 )                                                                     \
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
if( ! noaTNL::MPI::Initialized() || noaTNL::MPI::Finalized() )                                                                         \
{                                                                                                                                \
   if( condition) std::cerr << message << std::endl;                                                                             \
}                                                                                                                                \
else                                                                                                                             \
{                                                                                                                                \
   if( noaTNL::MPI::GetRank() > 0 )                                                                                                 \
   {                                                                                                                             \
      int __tnl_mpi_print_cnd = ( condition );                                                                                   \
      noaTNL::MPI::Send( &__tnl_mpi_print_cnd, 1, 0, 0 );                                                                           \
      if( condition ) {                                                                                                          \
         std::stringstream __tnl_mpi_print_stream_;                                                                              \
         __tnl_mpi_print_stream_ << "Node " << noaTNL::MPI::GetRank() << " of " << noaTNL::MPI::GetSize() << " : "                     \
                                 << message << std::endl;                                                                        \
         noaTNL::String __tnl_mpi_print_string_( __tnl_mpi_print_stream_.str() );                                                   \
         noaTNL::MPI::send( __tnl_mpi_print_string_, 0, std::numeric_limits< int >::max() );                                        \
      }                                                                                                                          \
   }                                                                                                                             \
   else                                                                                                                          \
   {                                                                                                                             \
      if( condition )                                                                                                            \
         std::cerr << "Node 0 of " << noaTNL::MPI::GetSize() << " : " << message << std::endl;                                      \
      for( int __tnl_mpi_print_j = 1; __tnl_mpi_print_j < noaTNL::MPI::GetSize(); __tnl_mpi_print_j++ )                             \
         {                                                                                                                       \
            int __tnl_mpi_print_cond;                                                                                            \
            noaTNL::MPI::Recv( &__tnl_mpi_print_cond, 1, __tnl_mpi_print_j, 0 );                                                    \
            if( __tnl_mpi_print_cond )                                                                                           \
            {                                                                                                                    \
               noaTNL::String __tnl_mpi_print_string_;                                                                              \
               noaTNL::MPI::recv( __tnl_mpi_print_string_, __tnl_mpi_print_j, std::numeric_limits< int >::max() );                  \
               std::cerr << __tnl_mpi_print_string_;                                                                             \
            }                                                                                                                    \
         }                                                                                                                       \
   }                                                                                                                             \
}
#else
#define TNL_MPI_PRINT_COND( condition, message )                                                                                 \
   std::cerr << message << std::endl;
#endif
