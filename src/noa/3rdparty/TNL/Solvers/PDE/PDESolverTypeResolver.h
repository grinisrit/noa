// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Solvers/PDE/TimeDependentPDESolver.h>
#include <noa/3rdparty/TNL/Solvers/PDE/TimeIndependentPDESolver.h>

namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Problem,
          typename TimeStepper,
          bool TimeDependent = Problem::isTimeDependent() >
class PDESolverTypeResolver
{
};

template< typename Problem,
          typename TimeStepper >
class PDESolverTypeResolver< Problem, TimeStepper, true >
{
   public:

      using SolverType = TimeDependentPDESolver< Problem, TimeStepper >;
};

template< typename Problem,
          typename TimeStepper >
class PDESolverTypeResolver< Problem, TimeStepper, false >
{
   public:

      using SolverType = TimeIndependentPDESolver< Problem >;
};

} // namespace PDE
} // namespace Solvers
} // namespace noa::TNL
