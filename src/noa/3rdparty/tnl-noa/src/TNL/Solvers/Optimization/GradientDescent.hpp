// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Optimization/GradientDescent.h>

namespace noa::TNL {
namespace Solvers {
namespace Optimization {

template< typename Vector, typename SolverMonitor >
void
GradientDescent< Vector, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   IterativeSolver< RealType, IndexType, SolverMonitor >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "relaxation", "Relaxation parameter for the gradient descent.", 1.0 );
}

template< typename Vector, typename SolverMonitor >
bool
GradientDescent< Vector, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   this->setRelaxation( parameters.getParameter< double >( prefix + "relaxation" ) );
   return IterativeSolver< RealType, IndexType, SolverMonitor >::setup( parameters, prefix );
}

template< typename Vector, typename SolverMonitor >
void
GradientDescent< Vector, SolverMonitor >::setRelaxation( const RealType& lambda )
{
   this->relaxation = lambda;
}

template< typename Vector, typename SolverMonitor >
auto
GradientDescent< Vector, SolverMonitor >::getRelaxation() const -> const RealType&
{
   return this->relaxation;
}

template< typename Vector, typename SolverMonitor >
template< typename GradientGetter >
bool
GradientDescent< Vector, SolverMonitor >::solve( VectorView& w, GradientGetter&& getGradient )
{
   this->gradient.setLike( w );
   auto gradient_view = gradient.getView();
   auto w_view = w.getView();
   this->gradient = 0.0;

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
      this->setResidue( addAndReduceAbs( w_view, -this->relaxation * gradient_view, TNL::Plus(), (RealType) 0.0 )
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
