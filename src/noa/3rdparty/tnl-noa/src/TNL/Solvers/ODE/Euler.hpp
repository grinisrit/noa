// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/ODE/Euler.h>

namespace noa::TNL {
namespace Solvers {
namespace ODE {

template< typename Vector, typename SolverMonitor >
void
Euler< Vector, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   config.addEntry< double >( prefix + "euler-cfl", "Coefficient C in the Courant–Friedrichs–Lewy condition.", 0.0 );
};

template< typename Vector, typename SolverMonitor >
bool
Euler< Vector, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   ExplicitSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "euler-cfl" ) )
      this->setCourantNumber( parameters.getParameter< double >( prefix + "euler-cfl" ) );
   return true;
}

template< typename Vector, typename SolverMonitor >
void
Euler< Vector, SolverMonitor >::setCourantNumber( const RealType& c )
{
   this->courantNumber = c;
}

template< typename Vector, typename SolverMonitor >
auto
Euler< Vector, SolverMonitor >::getCourantNumber() const -> const RealType&
{
   return this->courantNumber;
}

template< typename Vector, typename SolverMonitor >
template< typename RHSFunction >
bool
Euler< Vector, SolverMonitor >::solve( VectorType& _u, RHSFunction&& rhsFunction )
{
   /////
   // First setup the supporting vector k1.
   _k1.setLike( _u );
   auto k1 = _k1.getView();
   auto u = _u.getView();
   k1 = 0.0;

   /////
   // Set necessary parameters
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 )
      return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( 1 ) {
      /////
      // Compute the RHS
      rhsFunction( time, currentTau, u, k1 );

      RealType lastResidue = this->getResidue();
      RealType maxResidue( 0.0 );
      if( this->courantNumber != 0.0 ) {
         maxResidue = max( abs( k1 ) );
         if( currentTau * maxResidue > this->courantNumber ) {
            currentTau *= 0.9;
            continue;
         }
      }
      this->setResidue( addAndReduceAbs( u, currentTau * k1, TNL::Plus(), (RealType) 0.0 )
                        / ( currentTau * (RealType) u.getSize() ) );

      /////
      // When time is close to stopTime the new residue may be inaccurate significantly.
      if( currentTau + time == this->stopTime )
         this->setResidue( lastResidue );
      time += currentTau;
      // this->problem->applyBoundaryConditions( time, _u );

      if( ! this->nextIteration() )
         return this->checkConvergence();

      /////
      // Compute the new time step.
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;

      if( this->courantNumber != 0.0 ) {
         currentTau /= 0.95;
         currentTau = min( currentTau, this->getMaxTau() );
      }
   }
   return false;  // just to avoid warnings
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace noa::TNL
