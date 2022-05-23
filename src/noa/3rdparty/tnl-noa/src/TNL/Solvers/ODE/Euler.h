// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/ODE/ExplicitSolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

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
 *
 * It is supposed to be used when the unknown \f$ \vec x \in R^n \f$ is expressed by a \ref Containers::Vector.
 *
 * For problems where \f$ \vec x\f$ is represented by \ref TNL::Containers::StaticVector,
 * see \ref TNL::Solvers::ODE::StaticEuler<Containers::StaticVector<Size_,Real>>.
 * For problems where \f$ x\f$ is represented by floating-point number, see \ref TNL::Solvers::ODE::StaticEuler.
 *
 * The following example demonstrates the use the solvers:
 *
 * \includelineno Solvers/ODE/ODESolver-HeatEquationExample.h
 *
 * \tparam Vector is type of vector storing \f$ \vec x \in R^n \f$, mostly \ref TNL::Containers::Vector
 *    or \ref TNL::Containers::VectorView.
 */
template< typename Vector,
          typename SolverMonitor = IterativeSolverMonitor< typename Vector::RealType, typename Vector::IndexType > >
class Euler : public ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor >
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
    * Note that \e VectorType can be \ref TNL::Containers::VectorView but
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
   Euler() = default;

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
    * \brief This method sets the Courant number in the
    * [CFL condition](https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition).
    *
    * This method sets the constant \f$ C \f$ in the Courant–Friedrichs–Lewy condition. It means that
    *
    * \f[ \Delta t = \frac{C}{\| f( t,x )\|}, \f]
    *
    * if \f$ C > 0\f$. If \f$ C = 0 \f$ the time step stays fixed.
    *
    * \param c is the Courant number.
    */
   void
   setCourantNumber( const RealType& cfl );

   /**
    * \brief Getter for the Courant number.
    *
    * \return the Courant number.
    */
   const RealType&
   getCourantNumber() const;

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
   solve( VectorType& u, RHSFunction&& f );

protected:
   DofVectorType _k1;

   RealType courantNumber = 0.0;
};

}  // namespace ODE
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/ODE/Euler.hpp>
