// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/StaticIterativeSolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>

namespace noa::TNL {
namespace Solvers {
namespace ODE {

/**
 * \brief Base class for ODE solvers and explicit solvers od PDEs.
 *
 * This is a specialization for static solvers, i.e. solvers which of scalar problem
 * or small system of ODEs solution of which can be expressed by \ref TNL::Containers::StaticVector.
 * The static solvers can be created even in GPU kernels and can be combined with \ref TNL::Algorithms::ParallelFor.
 *
 * See also: \ref TNL::Solvers::ODE::StaticEuler, \ref TNL::Solvers::ODE::StaticMerson.
 *
 * \tparam Real is type of the floating-point arithmetics or static vector ( \ref TNL::Containers::StaticVector ).
 * \tparam Index is type for indexing.
 */
template< typename Real = double, typename Index = int >
class StaticExplicitSolver : public StaticIterativeSolver< Real, Index >
{
public:
   /**
    * \brief Type of the floating-point arithmetics or static vector.
    */
   using RealType = Real;

   /**
    * \brief Indexing type.
    */
   using IndexType = Index;

   /**
    * \brief Default constructor.
    */
   __cuda_callable__
   StaticExplicitSolver() = default;

   /**
    * \brief This method defines configuration entries for setup of the iterative solver.
    */
   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   /**
    * \brief Method for setup of the iterative solver based on configuration parameters.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   /**
    * \brief Settter of the current time of the evolution computed by the solver.
    */
   __cuda_callable__
   void
   setTime( const RealType& t );

   /**
    * \brief Getter of the current time of the evolution computed by the solver.
    */
   __cuda_callable__
   const RealType&
   getTime() const;

   /**
    * \brief Setter of the time where the evolution computation shall by stopped.
    */
   __cuda_callable__
   void
   setStopTime( const RealType& stopTime );

   /**
    * \brief Getter of the time where the evolution computation shall by stopped.
    */
   __cuda_callable__
   const RealType&
   getStopTime() const;

   /**
    * \brief Setter of the time step used for the computation.
    *
    * The time step can be changed by methods using adaptive choice of the time step.
    */
   __cuda_callable__
   void
   setTau( const RealType& tau );

   /**
    * \brief Getter of the time step used for the computation.
    */
   __cuda_callable__
   const RealType&
   getTau() const;

   /**
    * \brief Setter of maximal value of the time step.
    *
    * If methods uses adaptive choice of the time step, this sets the upper limit.
    */
   __cuda_callable__
   void
   setMaxTau( const RealType& maxTau );

   /**
    * \brief Getter of maximal value of the time step.
    */
   __cuda_callable__
   const RealType&
   getMaxTau() const;

   /**
    * \brief Checks if the solver is allowed to do the next iteration.
    *
    * \return true \e true if the solver is allowed to do the next iteration.
    * \return \e false if the solver is \b not allowed to do the next iteration. This may
    *    happen because the divergence occurred.
    */
   bool __cuda_callable__
   checkNextIteration();

   __cuda_callable__
   void
   setTestingMode( bool testingMode );

protected:
   /****
    * Current time of the parabolic problem.
    */
   RealType time = 0.0;

   /****
    * The method solve will stop when reaching the stopTime.
    */
   RealType stopTime;

   /****
    * Current time step.
    */
   RealType tau = 0.0;

   RealType maxTau = std::numeric_limits< RealType >::max();

   bool stopOnSteadyState = false;

   bool testingMode = false;
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/ODE/StaticExplicitSolver.hpp>
