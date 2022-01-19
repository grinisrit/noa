// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Timer.h>

namespace TNL {
namespace MPI {

inline Timer& getTimerAllreduce()
{
   static Timer t;
   return t;
}

} // namespace MPI
} // namespace TNL
