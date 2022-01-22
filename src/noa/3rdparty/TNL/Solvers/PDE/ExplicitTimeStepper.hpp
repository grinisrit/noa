// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ExplicitTimeStepper.h"

namespace noaTNL {
namespace Solvers {
namespace PDE {

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
ExplicitTimeStepper< Problem, OdeSolver >::
ExplicitTimeStepper()
: problem( 0 ),
  timeStep( 0 ),
  allIterations( 0 )
{
}

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
void
ExplicitTimeStepper< Problem, OdeSolver >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
}

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
bool
ExplicitTimeStepper< Problem, OdeSolver >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   return this->odeSolver->setup( parameters, prefix );
}

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
bool
ExplicitTimeStepper< Problem, OdeSolver >::
init( const MeshType& mesh )
{
   this->explicitUpdaterTimer.reset();
   this->mainTimer.reset();
   this->preIterateTimer.reset();
   this->postIterateTimer.reset();
   return true;
}

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
void
ExplicitTimeStepper< Problem, OdeSolver >::
setSolver( typename ExplicitTimeStepper< Problem, OdeSolver >::OdeSolverType& odeSolver )
{
   this->odeSolver = &odeSolver;
};

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
void
ExplicitTimeStepper< Problem, OdeSolver >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
};

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
void
ExplicitTimeStepper< Problem, OdeSolver >::
setSolverMonitor( SolverMonitorType& solverMonitor )
{
   this->solverMonitor = &solverMonitor;
   if( this->odeSolver )
      this->odeSolver->setSolverMonitor( solverMonitor );
}

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
Problem*
ExplicitTimeStepper< Problem, OdeSolver >::
getProblem() const
{
    return this->problem;
};

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
bool
ExplicitTimeStepper< Problem, OdeSolver >::
setTimeStep( const RealType& timeStep )
{
   if( timeStep <= 0.0 )
   {
      std::cerr << "Tau for ExplicitTimeStepper must be positive. " << std::endl;
      return false;
   }
   this->timeStep = timeStep;
   return true;
};

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
bool
ExplicitTimeStepper< Problem, OdeSolver >::
solve( const RealType& time,
       const RealType& stopTime,
       DofVectorPointer& dofVector )
{
   TNL_ASSERT_TRUE( this->odeSolver, "ODE solver was not set" );
   mainTimer.start();
   this->odeSolver->setTau( this->timeStep );
   this->odeSolver->setProblem( * this );
   this->odeSolver->setTime( time );
   this->odeSolver->setStopTime( stopTime );
   if( this->odeSolver->getMinIterations() )
      this->odeSolver->setMaxTau( ( stopTime - time ) / ( typename OdeSolver< Problem, SolverMonitor >::RealType ) this->odeSolver->getMinIterations() );
   if( ! this->odeSolver->solve( dofVector ) )
      return false;
   //this->problem->setExplicitBoundaryConditions( stopTime, dofVector );
   mainTimer.stop();
   this->allIterations += this->odeSolver->getIterations();
   return true;
}

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
void
ExplicitTimeStepper< Problem, OdeSolver >::
getExplicitUpdate( const RealType& time,
                const RealType& tau,
                DofVectorPointer& u,
                DofVectorPointer& fu )
{
   if( this->solverMonitor ) {
      this->solverMonitor->setTime( time );
      this->solverMonitor->setStage( "Preiteration" );
   }

   this->preIterateTimer.start();
   if( ! this->problem->preIterate( time, tau, u ) )
   {
      std::cerr << std::endl << "Preiteration failed." << std::endl;
      return;
      //return false; // TODO: throw exception
   }
   this->preIterateTimer.stop();

   if( this->solverMonitor )
      this->solverMonitor->setStage( "Explicit update" );

   this->explicitUpdaterTimer.start();
   this->problem->applyBoundaryConditions( time, u );
   this->problem->getExplicitUpdate( time, tau, u, fu );
   this->explicitUpdaterTimer.stop();

   if( this->solverMonitor )
      this->solverMonitor->setStage( "Postiteration" );

   this->postIterateTimer.start();
   if( ! this->problem->postIterate( time, tau, u ) )
   {
      std::cerr << std::endl << "Postiteration failed." << std::endl;
      return;
      //return false; // TODO: throw exception
   }
   this->postIterateTimer.stop();
}

template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
void
ExplicitTimeStepper< Problem, OdeSolver >::
applyBoundaryConditions( const RealType& time,
                            DofVectorPointer& u )
{
   this->problem->applyBoundaryConditions( time, u );
}


template< typename Problem,
          template < typename OdeProblem, typename SolverMonitor > class OdeSolver >
bool
ExplicitTimeStepper< Problem, OdeSolver >::
writeEpilog( Logger& logger ) const
{
   logger.writeParameter< long long int >( "Iterations count:", this->allIterations );
   logger.writeParameter< const char* >( "Pre-iterate time:", "" );
   this->preIterateTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Explicit update computation:", "" );
   this->explicitUpdaterTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Explicit time stepper time:", "" );
   this->mainTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Post-iterate time:", "" );
   this->postIterateTimer.writeLog( logger, 1 );
   return true;
}

} // namespace PDE
} // namespace Solvers
} // namespace noaTNL
