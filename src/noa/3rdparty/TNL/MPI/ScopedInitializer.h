// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include "Wrappers.h"
#include "Utils.h"

namespace TNL {
namespace MPI {

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

} // namespace MPI
} // namespace TNL
