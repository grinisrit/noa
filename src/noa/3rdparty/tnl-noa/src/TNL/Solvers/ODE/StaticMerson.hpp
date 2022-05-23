// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

#include "StaticMerson.h"

namespace noa::TNL {
namespace Solvers {
namespace ODE {

template< typename Real >
void
StaticMerson< Real >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   StaticExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "merson-adaptivity",
                              "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, "
                              "zero means no adaptivity).",
                              1.0e-4 );
};

template< typename Real >
bool
StaticMerson< Real >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   StaticExplicitSolver< RealType, IndexType >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "merson-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "merson-adaptivity" ) );
   return true;
}

template< typename Real >
void __cuda_callable__
StaticMerson< Real >::setAdaptivity( const RealType& a )
{
   this->adaptivity = a;
};

template< typename Real >
__cuda_callable__
const Real&
StaticMerson< Real >::getAdaptivity() const
{
   return this->adaptivity;
};

template< typename Real >
template< typename RHSFunction, typename... Args >
bool __cuda_callable__
StaticMerson< Real >::solve( VectorType& u, RHSFunction&& rhsFunction, Args... args )
{
   if( this->getTau() == 0.0 )
      return false;

   /////
   // First setup the supporting vectors k1...k5 and kAux.
   k1 = 0.0;
   k2 = 0.0;
   k3 = 0.0;
   k4 = 0.0;
   k5 = 0.0;
   kAux = 0.0;

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
   while( this->checkNextIteration() ) {
      /////
      // Compute Runge-Kutta coefficients
      RealType tau_3 = currentTau / 3.0;

      /////
      // k1
      rhsFunction( time, currentTau, u, k1, args... );

      /////
      // k2
      kAux = u + currentTau * ( 1.0 / 3.0 * k1 );
      rhsFunction( time + tau_3, currentTau, kAux, k2, args... );

      /////
      // k3
      kAux = u + currentTau * 1.0 / 6.0 * ( k1 + k2 );
      rhsFunction( time + tau_3, currentTau, kAux, k3, args... );

      /////
      // k4
      kAux = u + currentTau * ( 0.125 * k1 + 0.375 * k3 );
      rhsFunction( time + 0.5 * currentTau, currentTau, kAux, k4, args... );

      /////
      // k5
      kAux = u + currentTau * ( 0.5 * k1 - 1.5 * k3 + 2.0 * k4 );
      rhsFunction( time + currentTau, currentTau, kAux, k5, args... );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if( adaptivity != 0.0 )
         error = currentTau / 3.0 * abs( 0.2 * k1 - 0.9 * k3 + 0.8 * k4 - 0.1 * k5 );

      if( adaptivity == 0.0 || error < adaptivity ) {
         RealType lastResidue = this->getResidue();
         time += currentTau;

         const RealType update = 1.0 / 6.0 * ( k1 + 4.0 * k4 + k5 );
         u += currentTau * update;
         this->setResidue( abs( update ) );

         /////
         // When time is close to stopTime the new residue
         // may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );

         if( ! this->nextIteration() )
            return false;
      }

      /////
      // Compute the new time step.
      if( adaptivity != 0.0 && error != 0.0 ) {
         currentTau *= 0.8 * TNL::pow( adaptivity / error, 0.2 );
         currentTau = min( currentTau, this->getMaxTau() );
      }
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
};

template< int Size_, typename Real >
void
StaticMerson< Containers::StaticVector< Size_, Real > >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   StaticExplicitSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "merson-adaptivity",
                              "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, "
                              "zero means no adaptivity).",
                              1.0e-4 );
};

template< int Size_, typename Real >
bool
StaticMerson< Containers::StaticVector< Size_, Real > >::setup( const Config::ParameterContainer& parameters,
                                                                const String& prefix )
{
   StaticExplicitSolver< RealType, IndexType >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "merson-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "merson-adaptivity" ) );
   return true;
}

template< int Size_, typename Real >
void __cuda_callable__
StaticMerson< Containers::StaticVector< Size_, Real > >::setAdaptivity( const RealType& a )
{
   this->adaptivity = a;
};

template< int Size_, typename Real >
__cuda_callable__
const Real&
StaticMerson< Containers::StaticVector< Size_, Real > >::getAdaptivity() const
{
   return this->adaptivity;
};

template< int Size_, typename Real >
template< typename RHSFunction, typename... Args >
bool __cuda_callable__
StaticMerson< Containers::StaticVector< Size_, Real > >::solve( VectorType& u, RHSFunction&& rhsFunction, Args... args )
{
   if( this->getTau() == 0.0 )
      return false;

   /////
   // First setup the supporting vectors k1...k5 and kAux.
   k1 = 0.0;
   k2 = 0.0;
   k3 = 0.0;
   k4 = 0.0;
   k5 = 0.0;
   kAux = 0.0;

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
   while( this->checkNextIteration() ) {
      /////
      // Compute Runge-Kutta coefficients
      RealType tau_3 = currentTau / 3.0;

      /////
      // k1
      rhsFunction( time, currentTau, u, k1, args... );

      /////
      // k2
      kAux = u + currentTau * ( 1.0 / 3.0 * k1 );
      rhsFunction( time + tau_3, currentTau, kAux, k2, args... );

      /////
      // k3
      kAux = u + currentTau * 1.0 / 6.0 * ( k1 + k2 );
      rhsFunction( time + tau_3, currentTau, kAux, k3, args... );

      /////
      // k4
      kAux = u + currentTau * ( 0.125 * k1 + 0.375 * k3 );
      rhsFunction( time + 0.5 * currentTau, currentTau, kAux, k4, args... );

      /////
      // k5
      kAux = u + currentTau * ( 0.5 * k1 - 1.5 * k3 + 2.0 * k4 );
      rhsFunction( time + currentTau, currentTau, kAux, k5, args... );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if( adaptivity != 0.0 )
         error = max( currentTau / 3.0 * abs( 0.2 * k1 - 0.9 * k3 + 0.8 * k4 - 0.1 * k5 ) );

      if( adaptivity == 0.0 || error < adaptivity ) {
         RealType lastResidue = this->getResidue();
         time += currentTau;

         this->setResidue( addAndReduceAbs( u, currentTau / 6.0 * ( k1 + 4.0 * k4 + k5 ), TNL::Plus(), (RealType) 0.0 )
                           / ( currentTau * (RealType) u.getSize() ) );

         /////
         // When time is close to stopTime the new residue
         // may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 )
            this->setResidue( lastResidue );

         if( ! this->nextIteration() )
            return false;
      }

      /////
      // Compute the new time step.
      if( adaptivity != 0.0 && error != 0.0 ) {
         currentTau *= 0.8 * ::pow( adaptivity / error, 0.2 );
         currentTau = min( currentTau, this->getMaxTau() );
      }
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time;  // we don't want to keep such tau
      else
         this->tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime()
          || ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace noa::TNL
