// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <type_traits>  // std::add_const_t
#include <memory>  // std::shared_ptr

#include <TNL/Solvers/IterativeSolver.h>
#include <TNL/Solvers/Linear/Preconditioners/Preconditioner.h>
#include <TNL/Solvers/Linear/Utils/Traits.h>

namespace TNL {
   namespace Solvers {
      namespace Linear {

/**
 * \brief Base class for iterative solvers of systems of linear equations.
 *
 * To use the linear solver, one needs to first set the matrix of the linear system
 * by means of the method \ref LinearSolver::setMatrix. Afterward, one may call
 * the method \ref LinearSolver::solve which accepts the right-hand side vector \e b
 * and a vector \e x to which the solution will be stored. One may also use appropriate
 * preconditioner to speed-up the convergence - see the method
 * \ref LinearSolver::setPreconditioner.
 *
 * \tparam Matrix is type of matrix representing the linear system.
 *
 * The following example demonstrates the use iterative linear solvers:
 *
 * \includelineno Solvers/Linear/IterativeLinearSolverExample.cpp
 *
 * The result looks as follows:
 *
 * \include IterativeLinearSolverExample.out
 *
 * See also \ref TNL::Solvers::IterativeSolverMonitor for monitoring of iterative solvers.
 */
template< typename Matrix >
class LinearSolver
: public IterativeSolver< typename Matrix::RealType, typename Matrix::IndexType >
{
   public:

      /**
       * \brief Floating point type used for computations.
       */
      using RealType = typename Matrix::RealType;

      /**
       * \brief Device where the solver will run on and auxillary data will alloacted on.
       *
       * See \ref Devices::Host or \ref Devices::Cuda.
       */
      using DeviceType = typename Matrix::DeviceType;

      /**
       * \brief Type for indexing.
       */
      using IndexType = typename Matrix::IndexType;

      /**
       * \brief Type for vector view.
       */
      using VectorViewType = typename Traits< Matrix >::VectorViewType;

      /**
       * \brief Type for constant vector view.
       */
      using ConstVectorViewType = typename Traits< Matrix >::ConstVectorViewType;

      /**
       * \brief Type of the matrix representing the linear system.
       */
      using MatrixType = Matrix;

      /**
       * \brief Type of shared pointer to the matrix.
       */
      using MatrixPointer = std::shared_ptr< std::add_const_t< MatrixType > >;

      /**
       * \brief Type of preconditioner.
       */
      using PreconditionerType = Preconditioners::Preconditioner< MatrixType >;

      /**
       * \brief Type of shared pointer to the preconditioner.
       */
      using PreconditionerPointer = std::shared_ptr< std::add_const_t< PreconditionerType > >;

      /**
       * \brief This method defines configuration entries for setup of the linear iterative solver.
       *
       * See \ref IterativeSolver::configSetup.
       *
       * \param config contains description of configuration parameters.
       * \param prefix is a prefix of particular configuration entries.
       */
      static void configSetup( Config::ConfigDescription& config,
                              const String& prefix = "" )
      {
         IterativeSolver< RealType, IndexType >::configSetup( config, prefix );
      }

      /**
       * \brief Method for setup of the linear iterative solver based on configuration parameters.
       *
       * \param parameters contains values of the define configuration entries.
       * \param prefix is a prefix of particular configuration entries.
       */
      virtual bool setup( const Config::ParameterContainer& parameters,
                          const String& prefix = "" )
      {
         return IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
      }

      /**
       * \brief Set the matrix of the linear system.
       *
       * \param matrix is a shared pointer to the matrix of the linear system
       */
      void setMatrix( const MatrixPointer& matrix )
      {
         this->matrix = matrix;
      }

      /**
       * \brief Set the preconditioner.
       *
       * \param preconditioner is a shared pointer to preconditioner.
       */
      void setPreconditioner( const PreconditionerPointer& preconditioner )
      {
         this->preconditioner = preconditioner;
      }

      /**
       * \brief Method for solving a linear system.
       *
       * The linear system is defined by the matrix given by the method \ref LinearSolver::setMatrix and
       * by the right-hand side vector represented by the vector \e b. The result is stored in the
       * vector \e b. The solver can be accelerated with appropriate preconditioner set by the methods
       * \ref LinearSolver::setPreconditioner.
       *
       * \param b vector with the right-hand side of the linear system.
       * \param x vector for the solution of the linear system.
       * \return true if the solver converged.
       * \return false if the solver did not converge.
       *
       * \par Example
       * \include Solvers/Linear/IterativeLinearSolverExample.cpp
       * \par Output
       * \include IterativeLinearSolverExample.out
       */
      virtual bool solve( ConstVectorViewType b, VectorViewType x ) = 0;

      /**
       * \brief Default destructor.
       */
      virtual ~LinearSolver() {}

   protected:
      MatrixPointer matrix = nullptr;
      PreconditionerPointer preconditioner = nullptr;
};
      } // namespace Linear
   } // namespace Solvers
} // namespace TNL
