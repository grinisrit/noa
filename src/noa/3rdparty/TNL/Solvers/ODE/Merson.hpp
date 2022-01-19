// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/MPI/Wrappers.h>

#include "Merson.h"

namespace TNL {
namespace Solvers {
namespace ODE {

/****
 * In this code we do not use constants and references as we would like to.
 * OpenMP would complain that
 *
 *  error: ‘some-variable’ is predetermined ‘shared’ for ‘firstprivate’
 *
 */

template< typename Problem, typename SolverMonitor >
Merson< Problem, SolverMonitor >::Merson()
: adaptivity( 0.00001 )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->openMPErrorEstimateBuffer.setSize( std::max( 1, Devices::Host::getMaxThreadsCount() ) );
   }
};

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::configSetup( Config::ConfigDescription& config,
                                                const String& prefix )
{
   //ExplicitSolver< Problem >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "merson-adaptivity", "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, zero means no adaptivity).", 1.0e-4 );
};

template< typename Problem, typename SolverMonitor >
bool Merson< Problem, SolverMonitor >::setup( const Config::ParameterContainer& parameters,
                                         const String& prefix )
{
   ExplicitSolver< Problem, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "merson-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "merson-adaptivity" ) );
   return true;
}

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::setAdaptivity( const RealType& a )
{
   this->adaptivity = a;
};

template< typename Problem, typename SolverMonitor >
bool Merson< Problem, SolverMonitor >::solve( DofVectorPointer& _u )
{
   if( ! this->problem )
   {
      std::cerr << "No problem was set for the Merson ODE solver." << std::endl;
      return false;
   }
   if( this->getTau() == 0.0 )
   {
      std::cerr << "The time step for the Merson ODE solver is zero." << std::endl;
      return false;
   }
   /////
   // First setup the supporting meshes k1...k5 and kAux.
   _k1->setLike( *_u );
   _k2->setLike( *_u );
   _k3->setLike( *_u );
   _k4->setLike( *_u );
   _k5->setLike( *_u );
   _kAux->setLike( *_u );
   auto k1 = _k1->getView();
   auto k2 = _k2->getView();
   auto k3 = _k3->getView();
   auto k4 = _k4->getView();
   auto k5 = _k5->getView();
   auto kAux = _kAux->getView();
   auto u = _u->getView();
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
   if( currentTau == 0.0 ) return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( this->checkNextIteration() )
   {
      /////
      // Compute Runge-Kutta coefficients
      RealType tau_3 = currentTau / 3.0;

      /////
      // k1
      this->problem->getExplicitUpdate( time, currentTau, _u, _k1 );

      /////
      // k2
      kAux = u + currentTau * ( 1.0 / 3.0 * k1 );
      this->problem->applyBoundaryConditions( time + tau_3, _kAux );
      this->problem->getExplicitUpdate( time + tau_3, currentTau, _kAux, _k2 );

      /////
      // k3
      kAux = u + currentTau * 1.0 / 6.0 * ( k1 + k2 );
      this->problem->applyBoundaryConditions( time + tau_3, _kAux );
      this->problem->getExplicitUpdate( time + tau_3, currentTau, _kAux, _k3 );

      /////
      // k4
      kAux = u + currentTau * ( 0.125 * k1 + 0.375 * k3 );
      this->problem->applyBoundaryConditions( time + 0.5 * currentTau, _kAux );
      this->problem->getExplicitUpdate( time + 0.5 * currentTau, currentTau, _kAux, _k4 );

      /////
      // k5
      kAux = u + currentTau * ( 0.5 * k1 - 1.5 * k3 + 2.0 * k4 );
      this->problem->applyBoundaryConditions( time + currentTau, _kAux );
      this->problem->getExplicitUpdate( time + currentTau, currentTau, _kAux, _k5 );

      if( this->testingMode )
         writeGrids( _u );

      /////
      // Compute an error of the approximation.
      RealType error( 0.0 );
      if( adaptivity != 0.0 )
      {
         const RealType localError =
            max( currentTau / 3.0 * abs( 0.2 * k1 -0.9 * k3 + 0.8 * k4 -0.1 * k5 ) );
            MPI::Allreduce( &localError, &error, 1, MPI_MAX, MPI_COMM_WORLD );
      }

      if( adaptivity == 0.0 || error < adaptivity )
      {
         RealType lastResidue = this->getResidue();
         time += currentTau;

         this->setResidue( addAndReduceAbs( u, currentTau / 6.0 * ( k1 + 4.0 * k4 + k5 ),
            std::plus<>{}, ( RealType ) 0.0 ) / ( currentTau * ( RealType ) u.getSize() ) );

         /////
         // When time is close to stopTime the new residue
         // may be inaccurate significantly.
         if( abs( time - this->stopTime ) < 1.0e-7 ) this->setResidue( lastResidue );

         if( ! this->nextIteration() )
            return false;
      }

      /////
      // Compute the new time step.
      if( adaptivity != 0.0 && error != 0.0 )
      {
         currentTau *= 0.8 * ::pow( adaptivity / error, 0.2 );
         currentTau = min( currentTau, this->getMaxTau() );
#ifdef USE_MPI
         TNLMPI::Bcast( currentTau, 1, 0 );
#endif
      }
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time; //we don't want to keep such tau
      else this->tau = currentTau;

      /////
      // Check stop conditions.
      if( time >= this->getStopTime() ||
          ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();
};

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::writeGrids( const DofVectorPointer& u )
{
   std::cout << "Writing Merson solver grids ...";
   File( "Merson-u.tnl", std::ios_base::out ) << *u;
   File( "Merson-k1.tnl", std::ios_base::out ) << *_k1;
   File( "Merson-k2.tnl", std::ios_base::out ) << *_k2;
   File( "Merson-k3.tnl", std::ios_base::out ) << *_k3;
   File( "Merson-k4.tnl", std::ios_base::out ) << *_k4;
   File( "Merson-k5.tnl", std::ios_base::out ) << *_k5;
   std::cout << " done. PRESS A KEY." << std::endl;
   getchar();
}

} // namespace ODE
} // namespace Solvers
} // namespace TNL
