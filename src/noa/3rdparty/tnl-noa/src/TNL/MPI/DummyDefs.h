// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HAVE_MPI
using MPI_Request = int;
using MPI_Comm = int;
using MPI_Info = int;

enum MPI_Op
{
   MPI_MAX,
   MPI_MIN,
   MPI_SUM,
   MPI_PROD,
   MPI_LAND,
   MPI_BAND,
   MPI_LOR,
   MPI_BOR,
   MPI_LXOR,
   MPI_BXOR,
   MPI_MINLOC,
   MPI_MAXLOC,
};

// Comparison results
enum
{
   MPI_IDENT,
   MPI_CONGRUENT,
   MPI_SIMILAR,
   MPI_UNEQUAL
};

// MPI_Init_thread constants
enum
{
   MPI_THREAD_SINGLE,
   MPI_THREAD_FUNNELED,
   MPI_THREAD_SERIALIZED,
   MPI_THREAD_MULTIPLE
};

   // Miscellaneous constants
   #define MPI_ANY_SOURCE -1     /* match any source rank */
   #define MPI_PROC_NULL -2      /* rank of null process */
   #define MPI_ROOT -4           /* special value for intercomms */
   #define MPI_ANY_TAG -1        /* match any message tag */
   #define MPI_UNDEFINED -32766  /* undefined stuff */
   #define MPI_DIST_GRAPH 3      /* dist graph topology */
   #define MPI_CART 1            /* cartesian topology */
   #define MPI_GRAPH 2           /* graph topology */
   #define MPI_KEYVAL_INVALID -1 /* invalid key value */

   // MPI handles
   // (According to the MPI standard, they are only link-time constants (not
   // compile-time constants). OpenMPI implements them as global variables.)
   #define MPI_COMM_WORLD 1
   #define MPI_COMM_SELF MPI_COMM_WORLD
   // NULL handles
   #define MPI_GROUP_NULL 0
   #define MPI_COMM_NULL 0
   #define MPI_REQUEST_NULL 0
   #define MPI_MESSAGE_NULL 0
   #define MPI_OP_NULL 0
   #define MPI_ERRHANDLER_NULL 0
   #define MPI_INFO_NULL 0
   #define MPI_WIN_NULL 0
   #define MPI_FILE_NULL 0
   #define MPI_T_ENUM_NULL 0

#endif
