// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/*
 * TODO: further variations to explore:
 *
 * [5] Gerard L. G. Sleijpen and Henk A. van der Vorst, "Reliable updated
 *     residuals in hybrid Bi-CG methods", Computing 56 (2), 141-163 (1996).
 * [6] Gerard L. G. Sleijpen and Henk A. van der Vorst, "Maintaining convergence
 *     properties of BiCGstab methods in finite precision arithmetic", Numerical
 *     Algorithms 10, 203-223 (1995).
 */

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/LinearSolver.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

/**
 * \brief Iterative solver of linear systems based on the BICGStab(l) method.
 *
 * BICGStabL implements an iterative solver for non-symmetric linear systems,
 * using the BiCGstab(l) algorithm described in [1] and [2]. It is a
 * generalization of the stabilized biconjugate-gradient (BiCGstab) algorithm
 * proposed by van der Vorst [3]. BiCGstab(1) is equivalent to BiCGstab, and
 * BiCGstab(2) is a slightly more efficient version of the BiCGstab2 algorithm
 * by Gutknecht [4], while BiCGstab(l>2) is a further generalization.
 *
 * [1] Gerard L. G. Sleijpen and Diederik R. Fokkema, "BiCGstab(l) for linear
 *     equations involving unsymmetric matrices with complex spectrum",
 *     Electronic Trans. on Numerical Analysis 1, 11-32 (1993).
 *
 * [2] Gerard L. G. Sleijpen, Henk A. van der Vorst, and Diederik R. Fokkema,
 *     "BiCGstab(l) and other Hybrid Bi-CG Methods", Numerical Algorithms 7,
 *     75-109 (1994).
 *
 * [3] Henk A. van der Vorst, "Bi-CGSTAB: A fast and smoothly converging variant
 *     of Bi-CG for the solution of nonsymmetric linear systems, SIAM Journal on
 *     scientific and Statistical Computing 13.2, 631-644 (1992).
 *
 * [4] Martin H. Gutknecht, "Variants of BiCGStab for matrices with complex
 *     spectrum", IPS Research Report No. 91-14 (1991).
 *
 * See \ref TNL::Solvers::Linear::IterativeSolver for example of showing how to use the linear solvers.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class BICGStabL : public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;

   // compatibility shortcut
   using Traits = Linear::Traits< Matrix >;

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
    * \brief This is method defines configuration entries for setup of the linear iterative solver.
    *
    * In addition to config entries defined by \ref IterativeSolver::configSetup, this method
    * defines the following:
    *
    * \e bicgstab-ell - number of Bi-CG iterations before the MR part starts.
    *
    * \e bicgstab-exact-residue - says whether the BiCGstab should compute the exact residue in
    *                             each step (true) or to use a cheap approximation (false).
    *
    * \param config contains description of configuration parameters.
    * \param prefix is a prefix of particular configuration entries.
    */
   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" );

   /**
    * \brief Method for setup of the linear iterative solver based on configuration parameters.
    *
    * \param parameters contains values of the define configuration entries.
    * \param prefix is a prefix of particular configuration entries.
    */
   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" ) override;

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
   using VectorType = typename Traits::VectorType;
   using DeviceVector = Containers::Vector< RealType, DeviceType, IndexType >;
   using HostVector = Containers::Vector< RealType, Devices::Host, IndexType >;

   void
   compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b );

   void
   preconditioned_matvec( ConstVectorViewType src, VectorViewType dst );

   void
   setSize( const VectorViewType& x );

   int ell = 1;

   bool exact_residue = false;

   // matrices (in column-major format)
   DeviceVector R, U;
   // single vectors (distributed)
   VectorType r_ast, M_tmp, res_tmp;
   // host-only storage
   HostVector T, sigma, g_0, g_1, g_2;

   IndexType size = 0;
   IndexType ldSize = 0;
};

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/BICGStabL.hpp>
