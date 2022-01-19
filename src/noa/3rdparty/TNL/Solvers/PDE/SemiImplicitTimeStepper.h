// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>  // std::shared_ptr

#include <TNL/Timer.h>
#include <TNL/Logger.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Problem >
class SemiImplicitTimeStepper
{
   public:

   typedef Problem ProblemType;
   typedef typename Problem::RealType RealType;
   typedef typename Problem::DeviceType DeviceType;
   typedef typename Problem::IndexType IndexType;
   typedef typename Problem::MeshType MeshType;
   typedef typename ProblemType::DofVectorType DofVectorType;
   typedef Pointers::SharedPointer< DofVectorType, DeviceType > DofVectorPointer;
   typedef IterativeSolverMonitor< RealType, IndexType > SolverMonitorType;

   using MatrixType = typename ProblemType::MatrixType;
   using MatrixPointer = std::shared_ptr< MatrixType >;
   using LinearSolverType = Linear::LinearSolver< MatrixType >;
   using LinearSolverPointer = std::shared_ptr< LinearSolverType >;
   using PreconditionerType = typename LinearSolverType::PreconditionerType;
   using PreconditionerPointer = std::shared_ptr< PreconditionerType >;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   bool init( const MeshType& mesh );

   void setProblem( ProblemType& problem );

   ProblemType* getProblem() const;

   void setSolverMonitor( SolverMonitorType& solverMonitor );

   bool setTimeStep( const RealType& timeStep );

   const RealType& getTimeStep() const;

   bool solve( const RealType& time,
               const RealType& stopTime,
               DofVectorPointer& dofVectorPointer );

   bool writeEpilog( Logger& logger ) const;

   protected:

   // raw pointers with setters
   Problem* problem = nullptr;
   SolverMonitorType* solverMonitor = nullptr;

   // smart pointers initialized to the default-created objects
   DofVectorPointer rightHandSidePointer;

   // uninitialized smart pointers (they are initialized in the setup or init method)
   MatrixPointer matrix = nullptr;
   LinearSolverPointer linearSystemSolver = nullptr;
   PreconditionerPointer preconditioner = nullptr;

   RealType timeStep = 0.0;

   Timer preIterateTimer, linearSystemAssemblerTimer, preconditionerUpdateTimer, linearSystemSolverTimer, postIterateTimer;

   long long int allIterations = 0;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/PDE/SemiImplicitTimeStepper.hpp>
