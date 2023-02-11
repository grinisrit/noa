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
 * \brief Iterative solver of linear systems based on the Successive-overrelaxation (SOR) or Gauss-Seidel method.
 *
 * See (Wikipedia)[https://en.wikipedia.org/wiki/Successive_over-relaxation] for more details.
 *
 * See \ref TNL::Solvers::Linear::LinearSolver for example of showing how to use the linear solvers.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 */
template< typename Matrix >
class SOR : public LinearSolver< Matrix >
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
    * \e sor-omega - relaxation parameter of the weighted/damped Jacobi method.
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
    * \brief Setter of the relaxation parameter.
    *
    * \param omega the relaxation parameter. It is 1 by default.
    */
   void
   setOmega( const RealType& omega );

   /**
    * \brief Getter of the relaxation parameter.
    *
    * \return value of the relaxation parameter.
    */
   const RealType&
   getOmega() const;

   /**
    * \brief Set the period for a recomputation of the residue.
    *
    * \param period number of iterations between subsequent recomputations of the residue.
    */
   void
   setResiduePeriod( IndexType period );

   /**
    * \brief Get the period for a recomputation of the residue.
    *
    * \return number of iterations between subsequent recomputations of the residue.
    */
   IndexType
   getResiduePerid() const;

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
   RealType omega = 1.0;

   IndexType residuePeriod = 4;

   VectorType diagonal;

public:  // because nvcc does not accept lambda functions within private or protected methods
   void
   performIteration( const ConstVectorViewType& b, const ConstVectorViewType& diagonalView, VectorViewType& x ) const;
};

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/SOR.hpp>
