// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/ODE/ExplicitSolver.h>

namespace noa::TNL {
namespace Solvers {
namespace ODE {

/**
 * \brief Solver of ODEs with the first order of accuracy.
 *
 * This solver is based on the [Euler method](https://en.wikipedia.org/wiki/Euler_method) for solving of
 * [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) having the
 * following form:
 *
 * \f$ \frac{d \vec u}{dt} = \vec f( t, \vec u) \text{ on } (0,T) \f$
 *
 * \f$ \vec u( 0 )  = \vec u_{ini} \f$.
 * It is supposed to be used when the unknown \f$ \vec x \in R^n \f$ is expressed by a \ref Containers::Vector.
 *
 * For problems where \f$ \vec x\f$ is represented by \ref TNL::Containers::StaticVector,
 * see \ref TNL::Solvers::ODE::StaticMerson<Containers::StaticVector<Size_,Real>>.
 * For problems where \f$ x\f$ is represented by floating-point number, see \ref TNL::Solvers::ODE::StaticMerson.
 *
 * The following example demonstrates the use the solvers:
 *
 * \includelineno Solvers/ODE/ODESolver-HeatEquationExample.h
 *
 * \tparam Vector is type of vector storing \f$ \vec x \in R^n \f$, mostly \ref TNL::Containers::Vector
 *    or \ref TNL::Containers::VectorView.
 */
template< class Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class Merson : public ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
{
public:
   /**
    * \brief Type of floating-point arithemtics.
    */
   using RealType = typename Vector::RealType;

   /**
    * \brief Device where the solver is supposed to be executed.
    */
   using DeviceType = typename Vector::DeviceType;

   /**
    * \brief Type for indexing.
    */
   using IndexType = typename Vector::IndexType;

   /**
    * \brief Type of unknown variable \f$ \vec x \f$.
    */
   using VectorType = Vector;

   /**
    * \brief Alias for type of unknown variable \f$ \vec x \f$.
    *
    * Note, \e VectorType can be \ref TNL::Containers::VectorView but
    * \e DofVectorType is always \ref TNL::Containers::Vector.
    */
   using DofVectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /**
    * \brief Type of object used for monitoring the convergence.
    *
    * Can be \ref TNL::Solvers::IterativeSolverMonitor.
    */
   using SolverMonitorType = SolverMonitor;

   /**
    * \brief Default constructor.
    */
   Merson() = default;

   /**
    * \brief Static method for setup of configuration parameters.
    *
    * \param config is the config description.
    * \param prefix is the prefix of the configuration parameters for this solver.
    */
   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   /**
    * \brief Method for setup of the explicit solver based on configuration parameters.
    *
    * \param parameters is the container for configuration parameters.
    * \param prefix is the prefix of the configuration parameters for this solver.
    * \return true if the parameters where parsed successfully.
    * \return false if the method did not succeed to read the configuration parameters.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   /**
    * \brief Setter of the parameter controlling the adaptive choice of the integration time step.
    *
    * The smaller the parammeter is the smaller the intergation time step tends to be.
    * Reasonable values for this parameters are approximately from interval \f$ [10^{-12},10^{-2}] \f$.
    * \param adaptivity new value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   void
   setAdaptivity( const RealType& adaptivity );

   /**
    * \brief Getter of the parameter controlling the adaptive choice of the integration time step.
    *
    * \returns the current value of the parameter controlling the adaptive choice of
    *    integration time step.
    */
   const RealType&
   getAdaptivity() const;

   /**
    * \brief Solve ODE given by a lambda function.
    *
    * \tparam RHSFunction is type of a lambda function representing the right-hand side of the ODE system.
    *    The definition of the lambda function reads as:
    * ```
    * auto f = [=] ( const Real& t, const Real& tau, const VectorType& u, VectorType& fu ) {...}
    * ```
    * where `t` is the current time of the evolution, `tau` is the current time step, `u` is the solution at the current time,
    * `fu` is variable/static vector into which the lambda function is suppsed to evaluate the function \f$ f(t, \vec x) \f$ at
    * the current time \f$ t \f$.
    * \param u is a variable/static vector representing the solution of the ODE system at current time.
    * \param f is the lambda function representing the right-hand side of the ODE system.
    * \param args are user define arguments which are passed to the lambda function `f`.
    * \return `true` if steady state solution has been reached, `false` otherwise.
    */
   template< typename RHSFunction >
   bool
   solve( VectorType& u, RHSFunction&& rhs );

   void
   writeGrids( const DofVectorType& u );

protected:
   DofVectorType _k1, _k2, _k3, _k4, _k5, _kAux;

   /****
    * This controls the accuracy of the solver
    */
   RealType adaptivity = 0.00001;
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/ODE/Merson.hpp>
