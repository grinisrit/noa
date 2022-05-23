// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Solvers {

/**
 * \brief Namespace for solvers of ordinary differential equations.
 *
 * Solvers in this namespace represent numerical methods for the solution of
 * the [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) of the following form:
 *
 * \f[ \frac{d \vec u}{dt} = \vec f( t, \vec u)\ \rm{ on }\ (0,T) \f]
 *
 * \f[ \vec u( 0 )  = \vec u_{ini}. \f]
 *
 * The following solvers are available:
 *
 * 1. First order of accuracy
 *      a. \ref TNL::Solvers::ODE::StaticEuler, \ref TNL::Solvers::ODE::Euler - the Euler method
 * 2. Fourth order of accuracy
 *      a. \ref TNL::Solvers::ODE::StaticMerson, \ref TNL::Solvers::ODE::Merson - the Runge-Kutta-Merson
 *          method with adaptive choice of the time step.
 *
 * The static variants of the solvers are supposed to be used when the unknown \f$ x \in R^n \f$
 * is expressed by a \ref Containers::StaticVector or it is a scalar, i.e.
 * \f$ x \in R \f$ expressed by a numeric type like `double` or `float`.
 */
namespace ODE {}  // namespace ODE
}  // namespace Solvers
}  // namespace noa::TNL
