// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

namespace noa::TNL {
namespace Solvers {
namespace ODE {

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   IterativeSolver< Real, Index >::configSetup( config, prefix );
   config.addEntry< bool >(
      prefix + "stop-on-steady-state", "The computation stops when steady-state solution is reached.", false );
}

template< typename Real, typename Index, typename SolverMonitor >
bool
ExplicitSolver< Real, Index, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   this->stopOnSteadyState = parameters.getParameter< bool >( "stop-on-steady-state" );
   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setTime( const RealType& time )
{
   this->time = time;
};

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getTime() const
{
   return this->time;
};

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setTau( const RealType& tau )
{
   this->tau = tau;
};

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getTau() const
{
   return this->tau;
};

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setMaxTau( const RealType& maxTau )
{
   this->maxTau = maxTau;
};

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getMaxTau() const
{
   return this->maxTau;
};

template< typename Real, typename Index, typename SolverMonitor >
const Real&
ExplicitSolver< Real, Index, SolverMonitor >::getStopTime() const
{
   return this->stopTime;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setStopTime( const RealType& stopTime )
{
   this->stopTime = stopTime;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::refreshSolverMonitor( bool force )
{
   if( this->solverMonitor ) {
      this->solverMonitor->setIterations( this->getIterations() );
      this->solverMonitor->setResidue( this->getResidue() );
      this->solverMonitor->setTimeStep( this->getTau() );
      this->solverMonitor->setTime( this->getTime() );
      this->solverMonitor->setRefreshRate( this->refreshRate );
   }
}

template< typename Real, typename Index, typename SolverMonitor >
bool
ExplicitSolver< Real, Index, SolverMonitor >::checkNextIteration()
{
   if( std::isnan( this->getResidue() ) || this->getIterations() > this->getMaxIterations()
       || ( this->getResidue() > this->getDivergenceResidue() && this->getIterations() >= this->getMinIterations() )
       || ( this->getResidue() < this->getConvergenceResidue() && this->getIterations() >= this->getMinIterations()
            && this->stopOnSteadyState ) )
      return false;
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
void
ExplicitSolver< Real, Index, SolverMonitor >::setTestingMode( bool testingMode )
{
   this->testingMode = testingMode;
}

}  // namespace ODE
}  // namespace Solvers
}  // namespace noa::TNL
