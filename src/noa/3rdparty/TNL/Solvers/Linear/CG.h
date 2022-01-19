// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Solvers/Linear/LinearSolver.h>

namespace TNL {
   namespace Solvers {
      namespace Linear {

/**
 * \brief Iterative solver of linear systems based on the conjugate gradient method.
 *
 * This solver can be used only for positive-definite linear systems.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method) for more details.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 *
 * See \ref TNL::Solvers::Linear::IterativeSolver for example of showing how to use the linear solvers.
 */
template< typename Matrix >
class CG
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
   using VectorType = typename Traits< Matrix >::VectorType;

   public:
      /**
       * \brief Floating point type used for computations.
       */
      using RealType = typename Base::RealType;

      /**
       * \brief Device where the solver will run on and auxillary data will alloacted on.
       *
       * See \ref Devices::Host or \ref Devices::Cuda.
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
      bool solve( ConstVectorViewType b, VectorViewType x ) override;

   protected:
      void setSize( const VectorViewType& x );

      VectorType r, p, Ap, z;
};

      } // namespace Linear
   } // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/Linear/CG.hpp>
