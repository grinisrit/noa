// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Optimization/RMSProp.h>

namespace noa::TNL {
namespace Solvers {
namespace Optimization {

template< typename Vector, typename SolverMonitor >
void
RMSProp< Vector, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   IterativeSolver< RealType, IndexType, SolverMonitor >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "relaxation", "Relaxation parameter for the gradient descent.", 1.0 );
   config.addEntry< double >( prefix + "beta", "Momentum parameter for computing sum of squared gradients.", 0.9 );
}

template< typename Vector, typename SolverMonitor >
bool
RMSProp< Vector, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   this->setRelaxation( parameters.getParameter< double >( prefix + "relaxation" ) );
   this->beta = parameters.getParameter< double >( prefix + "beta" );
   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Vector, typename SolverMonitor >
void
RMSProp< Vector, SolverMonitor >::setRelaxation( const RealType& lambda )
{
   this->relaxation = lambda;
}

template< typename Vector, typename SolverMonitor >
auto
RMSProp< Vector, SolverMonitor >::getRelaxation() const -> const RealType&
{
   return this->relaxation;
}

template< typename Vector, typename SolverMonitor >
template< typename GradientGetter >
bool
RMSProp< Vector, SolverMonitor >::solve( VectorView& w, GradientGetter&& getGradient )
{
   this->gradient.setLike( w );
   this->a.setLike( w );
   auto gradient_view = gradient.getView();
   auto w_view = w.getView();
   this->gradient = 0.0;
   this->a = 0.0;

   /////
   // Set necessary parameters
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /////
   // Start the main loop
   while( 1 ) {
      /////
      // Compute the gradient
      getGradient( w_view, gradient_view );
      RealType lastResidue = this->getResidue();
      // a_i = beta * a_i + ( 1- beta ) * grad_i^2
      a = this->beta * a + ( 1.0 - this->beta ) * gradient_view * gradient_view;
      this->setResidue(
         addAndReduceAbs(
            w_view, -this->relaxation / sqrt( this->a + this->epsilon ) * gradient_view, TNL::Plus(), (RealType) 0.0 )
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
