// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Solvers/Linear/LinearSolver.h>

namespace noa::TNL {
   namespace Solvers {
      namespace Linear {

/**
 * \brief Iterative solver of linear systems based on the biconjugate gradient stabilized (BICGStab) method.
 *
 * This solver can be used for nonsymmetric linear systems.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method) for more details.
 *
 * See \ref noa::TNL::Solvers::Linear::IterativeSolver for example of showing how to use the linear solvers.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class BICGStab
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
       * \brief This is method defines configuration entries for setup of the linear iterative solver.
       *
       * In addition to config entries defined by \ref IterativeSolver::configSetup, this method
       * defines the following:
       *
       * \e bicgstab-exact-residue - says whether the BiCGstab should compute the exact residue in
       *                             each step (true) or to use a cheap approximation (false).
       *
       * \param config contains description of configuration parameters.
       * \param prefix is a prefix of particular configuration entries.
       */
      static void configSetup( Config::ConfigDescription& config,
                              const String& prefix = "" );

      /**
       * \brief Method for setup of the linear iterative solver based on configuration parameters.
       *
       * \param parameters contains values of the define configuration entries.
       * \param prefix is a prefix of particular configuration entries.
       */
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" ) override;

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
      void compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b );

      void preconditioned_matvec( ConstVectorViewType src, VectorViewType dst );

      void setSize( const VectorViewType& x );

      bool exact_residue = false;

      VectorType r, r_ast, p, s, Ap, As, M_tmp;
};

      } // namespace Linear
   } // namespace Solvers
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Solvers/Linear/BICGStab.hpp>
