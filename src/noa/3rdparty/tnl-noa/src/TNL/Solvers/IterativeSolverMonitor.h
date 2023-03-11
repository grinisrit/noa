// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/SolverMonitor.h>

namespace noa::TNL {
namespace Solvers {

/**
 * \brief Object for monitoring convergence of iterative solvers.
 *
 * \tparam Real is a type of the floating-point arithmetics.
 * \tparam Index is an indexing type.
 *
 * The following example shows how to use the iterative solver monitor for monitoring
 * convergence of linear iterative solver:
 *
 * \includelineno Solvers/Linear/IterativeLinearSolverWithMonitorExample.cpp
 *
 * The result looks as follows:
 *
 * \include IterativeLinearSolverWithMonitorExample.out
 *
 * The following example shows how to employ timer (\ref TNL::Timer) to the monitor
 * of iterative solvers:
 *
 * \includelineno Solvers/Linear/IterativeLinearSolverWithTimerExample.cpp
 *
 * The result looks as follows:
 *
 * \include IterativeLinearSolverWithTimerExample.out
 */
template< typename Real = double, typename Index = int >
class IterativeSolverMonitor : public SolverMonitor
{
public:
   /**
    * \brief A type of the floating-point arithmetics.
    */
   using RealType = Real;

   /**
    * \brief A type for indexing.
    */
   using IndexType = Index;

   /**
    * \brief Construct with no parameters.
    */
   IterativeSolverMonitor() = default;

   /**
    * \brief This method can be used for naming a stage of the monitored solver.
    *
    * The stage name can be used to differ between various stages of iterative solvers.
    *
    * \param stage is name of the solver stage.
    */
   void
   setStage( const std::string& stage );

   /**
    * \brief Set the time of the simulated evolution if it is time dependent.
    *
    * This can be used for example when solving parabolic or hyperbolic PDEs.
    *
    * \param time time of the simulated evolution.
    */
   void
   setTime( const RealType& time );

   /**
    * \brief Set the time step for time dependent iterative solvers.
    *
    * \param timeStep time step of the time dependent iterative solver.
    */
   void
   setTimeStep( const RealType& timeStep );

   /**
    * \brief Set number of the current iteration.
    *
    * \param iterations is number of the current iteration.
    */
   void
   setIterations( const IndexType& iterations );

   /**
    * \brief Set residue of the current approximation of the solution.
    *
    * \param residue is a residue of the current approximation of the solution.
    */
   void
   setResidue( const RealType& residue );

   /**
    * \brief Set up the verbosity of the monitor.
    *
    * \param verbose is the new value of the verbosity of the monitor.
    */
   void
   setVerbose( const IndexType& verbose );

   /**
    * \brief Set the number of nodes of the numerical mesh or lattice.
    *
    * This can be used to compute the number of nodes processed per one second.
    *
    * \param nodes is number of nodes of the numerical mesh or lattice.
    */
   void
   setNodesPerIteration( const IndexType& nodes );

   /**
    * \brief Causes that the monitor prints out the status of the solver.
    */
   void
   refresh() override;

protected:
   int
   getLineWidth();

   std::string stage, saved_stage;

   std::atomic_bool saved{ false };
   std::atomic_bool attributes_changed{ false };

   RealType time = 0;
   RealType saved_time = 0;
   RealType timeStep = 0;
   RealType saved_timeStep = 0;
   RealType residue = 0;
   RealType saved_residue = 0;
   RealType elapsed_time_before_refresh = 0;
   RealType last_mlups = 0;
   // TODO: Move MLUPS to LBM solver only i.e create solver monitor for LBM

   IndexType iterations = 0;
   IndexType saved_iterations = 0;
   IndexType iterations_before_refresh = 0;
   // TODO: move verbose to SolverMonitor
   IndexType verbose = 2;
   IndexType nodesPerIteration = 0;
};

}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/IterativeSolverMonitor.hpp>
