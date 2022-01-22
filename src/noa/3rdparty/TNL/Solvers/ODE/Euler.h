// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/TNL/Solvers/ODE/ExplicitSolver.h>
#include <noa/3rdparty/TNL/Config/ParameterContainer.h>

namespace noaTNL {
namespace Solvers {
namespace ODE {

template< typename Problem,
          typename SolverMonitor = IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > >
class Euler : public ExplicitSolver< Problem, SolverMonitor >
{
   public:

      using ProblemType = Problem;
      using DofVectorType = typename ProblemType::DofVectorType;
      using RealType = typename ProblemType::RealType;
      using DeviceType = typename ProblemType::DeviceType;
      using IndexType  = typename ProblemType::IndexType;
      using DofVectorPointer = Pointers::SharedPointer<  DofVectorType, DeviceType >;
      using SolverMonitorType = SolverMonitor;

      Euler();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setCFLCondition( const RealType& cfl );

      const RealType& getCFLCondition() const;

      bool solve( DofVectorPointer& u );

   protected:
      DofVectorPointer _k1;

      RealType cflCondition;
};

} // namespace ODE
} // namespace Solvers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Solvers/ODE/Euler.hpp>
