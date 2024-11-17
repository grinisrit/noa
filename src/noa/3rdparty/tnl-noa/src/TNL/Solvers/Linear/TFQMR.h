// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/LinearSolver.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

/**
 * \brief Iterative solver of linear systems based on the Transpose-free quasi-minimal residual (TFQMR) method.
 *
 * See (Wikipedia)[https://second.wiki/wiki/algoritmo_tfqmr] for more details.
 *
 * See \ref TNL::Solvers::Linear::LinearSolver for example of showing how to use the linear solvers.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class TFQMR : public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;

public:
   /**
    * \brief Floating point type used for computations.
    */
   using RealType = typename Base::RealType;

   /**
    * \brief Device where the solver will run on and auxillary data will alloacted on.
    */
   using DeviceType = typename Base::DeviceType;

   /**
    * \brief Type for indexing.
    */
   using IndexType = typename Base::IndexType;

   /**
    * \brief Type for vector view.
    */
   using VectorViewType = typename Base::VectorViewType;

   /**
    * \brief Type for constant vector view.
    */
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   /**
    * \brief Method for solving of a linear system.
    *
    * See \ref LinearSolver::solve for more details.
    *
    * \param b vector with the right-hand side of the linear system.
    * \param x vector for the solution of the linear system.
    * \return true if the solver converged.
    * \return false if the solver did not converge.
    */
   bool
   solve( ConstVectorViewType b, VectorViewType x ) override;

protected:
   void
   setSize( const VectorViewType& x );

   typename Traits< Matrix >::VectorType d, r, w, u, v, r_ast, Au, M_tmp;
};

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/TFQMR.hpp>
