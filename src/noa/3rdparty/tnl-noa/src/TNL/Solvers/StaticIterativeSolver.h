// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

namespace noa::TNL {
namespace Solvers {

/**
 * \brief Base class for iterative solvers.
 *
 * \tparam Real is a floating point type used for computations.
 * \tparam Index is an indexing type.
 * \tparam IterativeSolverMonitor< Real, Index > is type of an object used for monitoring of the convergence.
 */
template< typename Real, typename Index >
class StaticIterativeSolver
{
public:
   /**
    * \brief Default constructor.
    */
   __cuda_callable__
   StaticIterativeSolver() = default;

   /**
    * \brief This method defines configuration entries for setup of the iterative solver.
    *
    * The following entries are defined:
    *
    * \e max-iterations - maximal number of iterations the solver \b may perform.
    *
    * \e min-iterations - minimal number of iterations the solver \b must perform.
    *
    * \e convergence-residue - convergence occurs when the residue drops \b bellow this limit.
    *
    * \e divergence-residue - divergence occurs when the residue \b exceeds given limit.
    *
    * \e refresh-rate - number of milliseconds between solver monitor refreshes.
    *
    * \e residual-history-file - path to the file where the residual history will be saved.
    *
    * \param config contains description of configuration parameters.
    * \param prefix is a prefix of particular configuration entries.
    */
   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   /**
    * \brief Method for setup of the iterative solver based on configuration parameters.
    *
    * \param parameters contains values of the define configuration entries.
    * \param prefix is a prefix of particular configuration entries.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   /**
    * \brief Sets the maximal number of iterations the solver is \b allowed to perform.
    *
    * If the number of iterations performed by the solver exceeds given limit, the divergence occurs.
    *
    * \param maxIterations maximal number of allowed iterations.
    */
   __cuda_callable__
   void
   setMaxIterations( const Index& maxIterations );

   /**
    * \brief Gets the maximal number of iterations the solver is \b allowed to perform.
    *
    * See \ref IterativeSolver::setMaxIterations.
    *
    * \return maximal number of allowed iterations.
    */
   __cuda_callable__
   const Index&
   getMaxIterations() const;

   /**
    * \brief Sets the minimal number of iterations the solver is \b supposed to do.
    *
    * \param minIterations minimal number of iterations the solver is supposed to do.
    */
   __cuda_callable__
   void
   setMinIterations( const Index& minIterations );

   /**
    * \brief Gets the minimal number of iterations the solver is \b supposed to do.
    *
    * \return minimal number of iterations the solver is supposed to do.
    */
   __cuda_callable__
   const Index&
   getMinIterations() const;

   /**
    * \brief Gets the number of iterations performed by the solver so far.
    *
    * \return number of iterations performed so far.
    */
   __cuda_callable__
   const Index&
   getIterations() const;

   /**
    * \brief Sets the threshold for the convergence.
    *
    * The convergence occurs when the residue drops \b bellow this limit.
    *
    * \param convergenceResidue is threshold for the convergence.
    */
   __cuda_callable__
   void
   setConvergenceResidue( const Real& convergenceResidue );

   /**
    * \brief Gets the the convergence threshold.
    *
    * See \ref IterativeSolver::setConvergenceResidue.
    *
    * \return the convergence threshold.
    */
   __cuda_callable__
   const Real&
   getConvergenceResidue() const;

   /**
    * \brief Sets the residue limit for the divergence criterion.
    *
    * The divergence occurs when the residue \b exceeds the limit.
    *
    * \param divergenceResidue the residue limit of the divergence.
    */
   __cuda_callable__
   void
   setDivergenceResidue( const Real& divergenceResidue );

   /**
    * \brief Gets the limit for the divergence criterion.
    *
    * See \ref IterativeSolver::setDivergenceResidue.
    *
    * \return the residue limit fo the divergence.
    */
   __cuda_callable__
   const Real&
   getDivergenceResidue() const;

   /**
    * \brief Sets the residue reached at the current iteration.
    *
    * \param residue reached at the current iteration.
    */
   __cuda_callable__
   void
   setResidue( const Real& residue );

   /**
    * \brief Gets the residue reached at the current iteration.
    *
    * \return residue reached at the current iteration.
    */
   __cuda_callable__
   const Real&
   getResidue() const;

   /**
    * \brief Sets the the number of the current iterations to zero.
    */
   __cuda_callable__
   void
   resetIterations();

   /**
    * \brief Proceeds to the next iteration.
    *
    * \return \e true if the solver is allowed to do the next iteration.
    * \return \e false if the solver is \b not allowed to do the next iteration. This may
    *    happen because the divergence occurred.
    */
   __cuda_callable__
   bool
   nextIteration();

   /**
    * \brief Checks if the solver is allowed to do the next iteration.
    *
    * \return true \e true if the solver is allowed to do the next iteration.
    * \return \e false if the solver is \b not allowed to do the next iteration. This may
    *    happen because the divergence occurred.
    */
   __cuda_callable__
   bool
   checkNextIteration();

   /**
    * \brief Checks whether the convergence occurred already.
    *
    * \return \e true if the convergence already occured.
    * \return \e false if the convergence did not occur yet.
    */
   __cuda_callable__
   bool
   checkConvergence();

protected:
   Index maxIterations = 1000000000;

   Index minIterations = 0;

   Index currentIteration = 0;

   Real convergenceResidue = 1e-6;

   // If the current residue is greater than divergenceResidue, the solver is stopped.
   Real divergenceResidue = std::numeric_limits< Real >::max();

   Real currentResidue = 0;

   Index refreshRate = 1;
};

}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/StaticIterativeSolver.hpp>
