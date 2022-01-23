// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Timer.h>
#include <noa/3rdparty/TNL/Solvers/SolverMonitor.h>
#include <ostream>

namespace noa::TNL {
namespace Solvers {

template< typename ConfigTag >
class SolverStarter
{
   public:

   SolverStarter();

   template< typename Problem >
   static bool run( const Config::ParameterContainer& parameters );

   template< typename Solver >
   bool writeEpilog( std::ostream& str, const Solver& solver );

   template< typename Problem, typename TimeStepper >
   bool runPDESolver( Problem& problem,
                      const Config::ParameterContainer& parameters );

   protected:

   int logWidth;

   Timer ioTimer, computeTimer, totalTimer;
};

} // namespace Solvers
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Solvers/SolverStarter.hpp>
