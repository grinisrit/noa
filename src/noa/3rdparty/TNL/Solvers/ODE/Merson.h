// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <math.h>
#include <noa/3rdparty/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/TNL/Solvers/ODE/ExplicitSolver.h>

namespace noaTNL {
namespace Solvers {
namespace ODE {

template< class Problem,
          typename SolverMonitor = IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > >
class Merson : public ExplicitSolver< Problem, SolverMonitor >
{
   public:

      using ProblemType = Problem;
      using DofVectorType = typename Problem::DofVectorType;
      using RealType = typename Problem::RealType;
      using DeviceType = typename Problem::DeviceType;
      using IndexType = typename Problem::IndexType;
      using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
      using SolverMonitorType = SolverMonitor;

      Merson();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setAdaptivity( const RealType& a );

      bool solve( DofVectorPointer& u );

   protected:

      void writeGrids( const DofVectorPointer& u );

      DofVectorPointer _k1, _k2, _k3, _k4, _k5, _kAux;

      /****
       * This controls the accuracy of the solver
       */
      RealType adaptivity;

      Containers::Vector< RealType, DeviceType, IndexType > openMPErrorEstimateBuffer;
};

} // namespace ODE
} // namespace Solvers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Solvers/ODE/Merson.hpp>
