// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Math.h>
#include <noa/3rdparty/TNL/Solvers/PDE/SemiImplicitTimeStepper.h>
#include <noa/3rdparty/TNL/Solvers/LinearSolverTypeResolver.h>

namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Problem >
void
SemiImplicitTimeStepper< Problem >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
}

template< typename Problem >
bool
SemiImplicitTimeStepper< Problem >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   // set up the linear solver
   const String& discreteSolver = parameters.getParameter< String >( prefix + "discrete-solver" );
   linearSystemSolver = getLinearSolver< MatrixType >( discreteSolver );
   if( ! linearSystemSolver )
      return false;
   if( ! linearSystemSolver->setup( parameters ) )
      return false;

   // set up the preconditioner
   const String& preconditionerName = parameters.getParameter< String >( prefix + "preconditioner" );
   preconditioner = getPreconditioner< MatrixType >( preconditionerName );
   if( preconditioner ) {
      linearSystemSolver->setPreconditioner( preconditioner );
      if( ! preconditioner->setup( parameters ) )
         return false;
   }

   return true;
}

template< typename Problem >
bool
SemiImplicitTimeStepper< Problem >::
init( const MeshType& mesh )
{
   this->matrix = std::make_shared< MatrixType >();
   if( ! this->problem->setupLinearSystem( this->matrix ) ) {
      std::cerr << "Failed to set up the linear system." << std::endl;
      return false;
   }
   if( this->matrix->getRows() == 0 || this->matrix->getColumns() == 0 )
   {
      std::cerr << "The matrix for the semi-implicit time stepping was not set correctly." << std::endl;
      if( ! this->matrix->getRows() )
         std::cerr << "The matrix dimensions are set to 0 rows." << std::endl;
      if( ! this->matrix->getColumns() )
         std::cerr << "The matrix dimensions are set to 0 columns." << std::endl;
      std::cerr << "Please check the method 'setupLinearSystem' in your solver." << std::endl;
      return false;
   }
   this->rightHandSidePointer->setSize( this->matrix->getRows() );

   this->preIterateTimer.reset();
   this->linearSystemAssemblerTimer.reset();
   this->preconditionerUpdateTimer.reset();
   this->linearSystemSolverTimer.reset();
   this->postIterateTimer.reset();

   this->allIterations = 0;
   return true;
}

template< typename Problem >
void
SemiImplicitTimeStepper< Problem >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
};

template< typename Problem >
Problem*
SemiImplicitTimeStepper< Problem >::
getProblem() const
{
    return this->problem;
};

template< typename Problem >
void
SemiImplicitTimeStepper< Problem >::
setSolverMonitor( SolverMonitorType& solverMonitor )
{
   this->solverMonitor = &solverMonitor;
   if( this->linearSystemSolver )
      this->linearSystemSolver->setSolverMonitor( solverMonitor );
}

template< typename Problem >
bool
SemiImplicitTimeStepper< Problem >::
setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0.0 )
   {
      std::cerr << "Time step for SemiImplicitTimeStepper must be positive. " << std::endl;
      return false;
   }
   this->timeStep = timeStep;
   return true;
};

template< typename Problem >
bool
SemiImplicitTimeStepper< Problem >::
solve( const RealType& time,
       const RealType& stopTime,
       DofVectorPointer& dofVector )
{
   TNL_ASSERT_TRUE( this->problem, "problem was not set" );

   // set the matrix for the linear solver
   this->linearSystemSolver->setMatrix( this->matrix );

   RealType t = time;

   // ignore very small steps at the end, most likely caused by truncation errors
   while( stopTime - t > this->timeStep * 1e-6 )
   {
      RealType currentTau = min( this->timeStep, stopTime - t );

      if( this->solverMonitor ) {
         this->solverMonitor->setTime( t );
         this->solverMonitor->setStage( "Preiteration" );
      }

      this->preIterateTimer.start();
      if( ! this->problem->preIterate( t, currentTau, dofVector ) )
      {
         std::cerr << std::endl << "Preiteration failed." << std::endl;
         return false;
      }
      this->preIterateTimer.stop();

      if( this->solverMonitor )
         this->solverMonitor->setStage( "Assembling the linear system" );

      this->linearSystemAssemblerTimer.start();
      this->problem->assemblyLinearSystem( t,
                                           currentTau,
                                           dofVector,
                                           this->matrix,
                                           this->rightHandSidePointer );
      this->linearSystemAssemblerTimer.stop();

      if( this->solverMonitor )
         this->solverMonitor->setStage( "Solving the linear system" );

      if( this->preconditioner )
      {
         this->preconditionerUpdateTimer.start();
         preconditioner->update( this->matrix );
         this->preconditionerUpdateTimer.stop();
      }

      this->linearSystemSolverTimer.start();
      if( ! this->linearSystemSolver->solve( *this->rightHandSidePointer, *dofVector ) )
      {
         std::cerr << std::endl << "The linear system solver did not converge." << std::endl;
         // save the linear system for debugging
         this->problem->saveFailedLinearSystem( *this->matrix, *dofVector, *this->rightHandSidePointer );
         return false;
      }
      this->linearSystemSolverTimer.stop();
      this->allIterations += this->linearSystemSolver->getIterations();

      if( this->solverMonitor )
         this->solverMonitor->setStage( "Postiteration" );

      this->postIterateTimer.start();
      if( ! this->problem->postIterate( t, currentTau, dofVector ) )
      {
         std::cerr << std::endl << "Postiteration failed." << std::endl;
         return false;
      }
      this->postIterateTimer.stop();

      t += currentTau;
   }
   return true;
}

template< typename Problem >
bool
SemiImplicitTimeStepper< Problem >::
writeEpilog( Logger& logger ) const
{
   logger.writeParameter< long long int >( "Iterations count:", this->allIterations );
   logger.writeParameter< const char* >( "Pre-iterate time:", "" );
   this->preIterateTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Linear system assembler time:", "" );
   this->linearSystemAssemblerTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Preconditioner update time:", "" );
   this->preconditionerUpdateTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Linear system solver time:", "" );
   this->linearSystemSolverTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Post-iterate time:", "" );
   this->postIterateTimer.writeLog( logger, 1 );
   return true;
}

} // namespace PDE
} // namespace Solvers
} // namespace noa::TNL
