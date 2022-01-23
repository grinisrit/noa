// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Timer.h>
#include <noa/3rdparty/TNL/Logger.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/TNL/Solvers/IterativeSolverMonitor.h>

namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
class ExplicitTimeStepper
{
   public:

      using ProblemType = Problem;
      using RealType = typename Problem::RealType;
      using DeviceType = typename Problem::DeviceType;
      using IndexType = typename Problem::IndexType;
      using MeshType = typename Problem::MeshType;
      using DofVectorType = typename ProblemType::DofVectorType;
      using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
      using SolverMonitorType = IterativeSolverMonitor< RealType, IndexType >;
      using OdeSolverType = OdeSolver< ExplicitTimeStepper< Problem, OdeSolver >, SolverMonitorType >;
      using OdeSolverPointer = Pointers::SharedPointer< OdeSolverType, DeviceType >;

      static_assert( ProblemType::isTimeDependent(), "The problem is not time dependent." );

      ExplicitTimeStepper();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      bool init( const MeshType& mesh );

      void setSolver( OdeSolverType& odeSolver );

      void setSolverMonitor( SolverMonitorType& solverMonitor );

      void setProblem( ProblemType& problem );

      ProblemType* getProblem() const;

      bool setTimeStep( const RealType& tau );

      bool solve( const RealType& time,
                   const RealType& stopTime,
                   DofVectorPointer& dofVector );

      const RealType& getTimeStep() const;

      void getExplicitUpdate( const RealType& time,
                           const RealType& tau,
                           DofVectorPointer& _u,
                           DofVectorPointer& _fu );

      void applyBoundaryConditions( const RealType& time,
                                 DofVectorPointer& _u );

      bool writeEpilog( Logger& logger ) const;

   protected:

      OdeSolverPointer odeSolver;

      SolverMonitorType* solverMonitor;

      Problem* problem;

      RealType timeStep;

      Timer preIterateTimer, explicitUpdaterTimer, mainTimer, postIterateTimer;

      long long int allIterations;
};

} // namespace PDE
} // namespace Solvers
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Solvers/PDE/ExplicitTimeStepper.hpp>
