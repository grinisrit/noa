// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/LinearSolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrix.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

/**
 * \brief Iterative solver of linear systems based on the IDR(s) method.
 *
 * IDRs implements an iterative solver for non-symmetric linear systems,
 * using the Induced Dimension Reduction algorithm, denoted as IDR(s),
 * according to the description in [1] and [2].
 *
 * \note Unlike other iterative methods in TNL, IDR(s) is implemented with
 * right-preconditioning rather than left-preconditioning.
 *
 * [1] Peter Sonneveld and Martin Van Gijzen. "IDR(s): A Family of Simple and
 *     Fast Algorithms for Solving Large Nonsymmetric Systems of Linear
 *     Equations", SIAM Journal on Scientific Computing 31.2 (2009): 1035-1062.
 * [2] Martin Van Gijzen and Peter Sonneveld. "Algorithm 913: An elegant
 *     IDR(s) variant that efficiently exploits biorthogonality properties."
 *     ACM Transactions on Mathematical Software (TOMS) 38.1 (2011): 1-19.
 *
 * See \ref TNL::Solvers::Linear::LinearSolver for example of showing how to use the linear solvers.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class IDRs : public LinearSolver< Matrix >
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

   //! Sets the dimension of the shadow space.
   void
   setShadowSpaceDimension( int s );

   //! Enables or disables the residual smoothing procedure.
   void
   setResidualSmoothing( bool smoothing );

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
   using HostMatrix = Matrices::DenseMatrix< RealType, Devices::Sequential, IndexType >;

   void
   psolve( ConstVectorViewType src, VectorViewType dst );

   void
   matvec( ConstVectorViewType src, VectorViewType dst );

   void
   setSize( const VectorViewType& x );

   // shadow space dimension
   int s = 4;

   // residual smoothing
   bool smoothing = false;

   // matrices (in column-major format)
   DeviceVector P, G, U;
   // single vectors (distributed)
   VectorType r, v, t, x_s, r_s;
   // host-only storage
   HostMatrix M;
   HostVector f;

   IndexType size = 0;
   IndexType sizeWithGhosts = 0;
};

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/IDRs.hpp>
