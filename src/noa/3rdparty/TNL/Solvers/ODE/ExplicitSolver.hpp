// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

namespace TNL {
namespace Solvers {
namespace ODE {

template< typename Problem, typename SolverMonitor >
ExplicitSolver< Problem, SolverMonitor >::
ExplicitSolver()
:  time( 0.0 ),
   stopTime( 0.0 ),
   tau( 0.0 ),
   maxTau( std::numeric_limits< RealType >::max() ),
   verbosity( 0 ),
   testingMode( false ),
   problem( 0 )//,
   //solverMonitor( 0 )
{
};

template< typename Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< typename Problem::RealType, typename Problem::IndexType >::configSetup( config, prefix );
}

template< typename Problem, typename SolverMonitor >
bool
ExplicitSolver< Problem, SolverMonitor >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setVerbose( parameters.getParameter< int >( "verbose" ) );
   return IterativeSolver< typename Problem::RealType, typename Problem::IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setProblem( Problem& problem )
{
   this->problem = &problem;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setTime( const RealType& time )
{
   this->time = time;
};

template< class Problem, typename SolverMonitor >
const typename Problem :: RealType&
ExplicitSolver< Problem, SolverMonitor >::
getTime() const
{
   return this->time;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setTau( const RealType& tau )
{
   this->tau = tau;
};

template< class Problem, typename SolverMonitor >
const typename Problem :: RealType&
ExplicitSolver< Problem, SolverMonitor >::
getTau() const
{
   return this->tau;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setMaxTau( const RealType& maxTau )
{
   this->maxTau = maxTau;
};


template< class Problem, typename SolverMonitor >
const typename Problem :: RealType&
ExplicitSolver< Problem, SolverMonitor >::
getMaxTau() const
{
   return this->maxTau;
};


template< class Problem, typename SolverMonitor >
typename Problem :: RealType
ExplicitSolver< Problem, SolverMonitor >::
getStopTime() const
{
    return this->stopTime;
}

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setStopTime( const RealType& stopTime )
{
    this->stopTime = stopTime;
}

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setVerbose( IndexType v )
{
   this->verbosity = v;
};

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
refreshSolverMonitor( bool force )
{
   if( this->solverMonitor )
   {
      this->solverMonitor->setIterations( this->getIterations() );
      this->solverMonitor->setResidue( this->getResidue() );
      this->solverMonitor->setTimeStep( this->getTau() );
      this->solverMonitor->setTime( this->getTime() );
      this->solverMonitor->setRefreshRate( this->refreshRate );
   }
}

template< class Problem, typename SolverMonitor >
void
ExplicitSolver< Problem, SolverMonitor >::
setTestingMode( bool testingMode )
{
   this->testingMode = testingMode;
}


} // namespace ODE
} // namespace Solvers
} // namespace TNL
