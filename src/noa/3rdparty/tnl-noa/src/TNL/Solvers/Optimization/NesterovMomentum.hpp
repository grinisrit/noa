// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Optimization/NesterovMomentum.h>

namespace noa::TNL {
namespace Solvers {
namespace Optimization {

template< typename Vector, typename SolverMonitor >
void
NesterovMomentum< Vector, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   IterativeSolver< RealType, IndexType, SolverMonitor >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "relaxation", "Relaxation parameter for the momentum method.", 1.0 );
   config.addEntry< double >( prefix + "momentum", "Momentum parameter for the momentum method.", 0.9 );
}

template< typename Vector, typename SolverMonitor >
bool
NesterovMomentum< Vector, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   this->setRelaxation( parameters.getParameter< double >( prefix + "relaxation" ) );
   this->setMomentum( parameters.getParameter< double >( prefix + "momentum" ) );
   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Vector, typename SolverMonitor >
void
NesterovMomentum< Vector, SolverMonitor >::setRelaxation( const RealType& lambda )
{
   this->relaxation = lambda;
}

template< typename Vector, typename SolverMonitor >
auto
NesterovMomentum< Vector, SolverMonitor >::getRelaxation() const -> const RealType&
{
   return this->relaxation;
}

template< typename Vector, typename SolverMonitor >
void
NesterovMomentum< Vector, SolverMonitor >::setMomentum( const RealType& beta )
{
   this->momentum = beta;
}

template< typename Vector, typename SolverMonitor >
auto
NesterovMomentum< Vector, SolverMonitor >::getMomentum() const -> const RealType&
{
   return this->momentum;
}

template< typename Vector, typename SolverMonitor >
template< typename GradientGetter >
bool
NesterovMomentum< Vector, SolverMonitor >::solve( VectorView& w, GradientGetter&& getGradient )
{
   this->gradient.setLike( w );
   this->v.setLike( w );
   this->aux.setLike( w );
   auto gradient_view = gradient.getView();
   auto w_view = w.getView();
   auto v_view = v.getView();
   auto aux_view = aux.getView();
   this->gradient = 0.0;
   this->v = 0.0;

   /////
   // Set necessary parameters
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( 1 ) {
      /////
      // Compute the gradient
      aux_view = w_view + v_view;
      getGradient( aux_view, gradient_view );
      v_view = this->momentum * v_view - this->relaxation * gradient_view;

      RealType lastResidue = this->getResidue();
      this->setResidue( Algorithms::reduce< DeviceType >( (IndexType) 0,
                                                          w_view.getSize(),
                                                          [ = ] __cuda_callable__( IndexType i ) mutable
                                                          {
                                                             w_view[ i ] += v_view[ i ];
                                                             return abs( v_view[ i ] );
                                                          },
                                                          TNL::Plus() )
                        / ( this->relaxation * (RealType) w.getSize() ) );

      if( ! this->nextIteration() )
         return this->checkConvergence();

      /////
      // Check the stop condition
      if( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() )
         return true;
   }
   return false;  // just to avoid warnings
}

}  // namespace Optimization
}  // namespace Solvers
}  // namespace noa::TNL
