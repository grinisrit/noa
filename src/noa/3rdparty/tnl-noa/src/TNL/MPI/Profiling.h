// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>

namespace noa::TNL {
namespace MPI {

inline Timer&
getTimerAllreduce()
{
   static Timer t;
   return t;
}

}  // namespace MPI
}  // namespace noa::TNL
