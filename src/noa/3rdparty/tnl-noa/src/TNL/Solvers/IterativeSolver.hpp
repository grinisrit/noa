// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include "IterativeSolver.h"

namespace noa::TNL {
namespace Solvers {

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   if( config.getEntry( prefix + "max-iterations" ) == nullptr )
      config.addEntry< int >( prefix + "max-iterations", "Maximal number of iterations the solver may perform.", 1000000000 );
   if( config.getEntry( prefix + "min-iterations" ) == nullptr )
      config.addEntry< int >( prefix + "min-iterations", "Minimal number of iterations the solver must perform.", 0 );

   if( config.getEntry( prefix + "convergence-residue" ) == nullptr )
      config.addEntry< double >(
         prefix + "convergence-residue", "Convergence occurs when the residue drops bellow this limit.", 1e-6 );
   if( config.getEntry( prefix + "divergence-residue" ) == nullptr )
      config.addEntry< double >( prefix + "divergence-residue",
                                 "Divergence occurs when the residue exceeds given limit.",
                                 std::numeric_limits< float >::max() );
   // TODO: setting refresh rate should be done in SolverStarter::setup (it's not a parameter of the IterativeSolver)
   if( config.getEntry( prefix + "refresh-rate" ) == nullptr )
      config.addEntry< int >( prefix + "refresh-rate", "Number of milliseconds between solver monitor refreshes.", 500 );

   if( config.getEntry( prefix + "residual-history-file" ) == nullptr )
      config.addEntry< String >(
         prefix + "residual-history-file", "Path to the file where the residual history will be saved.", "" );
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   if( parameters.checkParameter( prefix + "max-iterations" ) )
      this->setMaxIterations( parameters.getParameter< int >( prefix + "max-iterations" ) );
   if( parameters.checkParameter( prefix + "min-iterations" ) )
      this->setMinIterations( parameters.getParameter< int >( prefix + "min-iterations" ) );
   if( parameters.checkParameter( prefix + "convergence-residue" ) )
      this->setConvergenceResidue( parameters.getParameter< double >( prefix + "convergence-residue" ) );
   if( parameters.checkParameter( prefix + "divergence-residue" ) )
      this->setDivergenceResidue( parameters.getParameter< double >( prefix + "divergence-residue" ) );
   // TODO: setting refresh rate should be done in SolverStarter::setup (it's not a parameter of the IterativeSolver)
   if( parameters.checkParameter( prefix + "refresh-rate" ) )
      this->setRefreshRate( parameters.getParameter< int >( prefix + "refresh-rate" ) );
   if( parameters.checkParameter( prefix + "residual-history-file" ) ) {
      this->residualHistoryFileName = parameters.getParameter< String >( prefix + "residual-history-file" );
      if( this->residualHistoryFileName )
         this->residualHistoryFile.open( this->residualHistoryFileName.getString() );
   }
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::setMaxIterations( const Index& maxIterations )
{
   this->maxIterations = maxIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
const Index&
IterativeSolver< Real, Index, SolverMonitor >::getMaxIterations() const
{
   return this->maxIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::setMinIterations( const Index& minIterations )
{
   this->minIterations = minIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
const Index&
IterativeSolver< Real, Index, SolverMonitor >::getMinIterations() const
{
   return this->minIterations;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::resetIterations()
{
   this->currentIteration = 0;
   if( this->solverMonitor )
      this->solverMonitor->setIterations( 0 );
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::nextIteration()
{
   // this->checkNextIteration() must be called before the iteration counter is incremented
   bool result = this->checkNextIteration();
   this->currentIteration++;
   if( this->solverMonitor ) {
      this->solverMonitor->setIterations( this->getIterations() );
   }
   return result;
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::checkNextIteration()
{
   if( std::isnan( this->getResidue() ) || this->getIterations() > this->getMaxIterations()
       || ( this->getResidue() > this->getDivergenceResidue() && this->getIterations() >= this->getMinIterations() )
       || ( this->getResidue() < this->getConvergenceResidue() && this->getIterations() >= this->getMinIterations() ) )
      return false;
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
bool
IterativeSolver< Real, Index, SolverMonitor >::checkConvergence()
{
   if( std::isnan( this->getResidue() ) ) {
      std::cerr << std::endl << "The residue is NaN." << std::endl;
      return false;
   }
   if( ( this->getResidue() > this->getDivergenceResidue() && this->getIterations() > this->minIterations ) ) {
      std::cerr << std::endl
                << "The residue has exceeded allowed tolerance " << this->getDivergenceResidue() << "." << std::endl;
      return false;
   }
   if( this->getIterations() >= this->getMaxIterations() ) {
      std::cerr << std::endl
                << "The solver has exceeded maximal allowed number of iterations " << this->getMaxIterations() << "."
                << std::endl;
      return false;
   }
   if( this->getResidue() > this->getConvergenceResidue() ) {
      std::cerr << std::endl
                << "The residue ( = " << this->getResidue() << " ) is too large( > " << this->getConvergenceResidue() << " )."
                << std::endl;
      return false;
   }
   return true;
}

template< typename Real, typename Index, typename SolverMonitor >
const Index&
IterativeSolver< Real, Index, SolverMonitor >::getIterations() const
{
   return this->currentIteration;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::setConvergenceResidue( const Real& convergenceResidue )
{
   this->convergenceResidue = convergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
IterativeSolver< Real, Index, SolverMonitor >::getConvergenceResidue() const
{
   return this->convergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::setDivergenceResidue( const Real& divergenceResidue )
{
   this->divergenceResidue = divergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
IterativeSolver< Real, Index, SolverMonitor >::getDivergenceResidue() const
{
   return this->divergenceResidue;
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::setResidue( const Real& residue )
{
   this->currentResidue = residue;
   if( this->solverMonitor )
      this->solverMonitor->setResidue( this->getResidue() );
   if( this->residualHistoryFile ) {
      if( this->getIterations() == 0 )
         this->residualHistoryFile << "\n";
      this->residualHistoryFile << this->getIterations() << "\t" << std::scientific << residue << std::endl;
   }
}

template< typename Real, typename Index, typename SolverMonitor >
const Real&
IterativeSolver< Real, Index, SolverMonitor >::getResidue() const
{
   return this->currentResidue;
}

// TODO: setting refresh rate should be done in SolverStarter::setup (it's not a parameter of the IterativeSolver)
template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::setRefreshRate( const Index& refreshRate )
{
   this->refreshRate = refreshRate;
   if( this->solverMonitor )
      this->solverMonitor->setRefreshRate( this->refreshRate );
}

template< typename Real, typename Index, typename SolverMonitor >
void
IterativeSolver< Real, Index, SolverMonitor >::setSolverMonitor( SolverMonitorType& solverMonitor )
{
   this->solverMonitor = &solverMonitor;
   this->solverMonitor->setRefreshRate( this->refreshRate );
}

}  // namespace Solvers
}  // namespace noa::TNL
