// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <type_traits>  // std::add_const_t
#include <memory>  // std::shared_ptr

#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Solvers/Linear/Utils/Traits.h>

namespace noa::TNL {
   namespace Solvers {
      namespace Linear {
         namespace Preconditioners {

/**
 * \brief Base class for preconditioners of of iterative solvers of linear systems.
 *
 * \tparam Matrix is type of matrix describing the linear system.
 *
 * The following example shows how to setup an iterative solver of linear systems with
 * preconditioning:
 *
 * \includelineno Solvers/Linear/IterativeLinearSolverWithPreconditionerExample.cpp
 *
 * The result looks as follows:
 *
 * \include IterativeLinearSolverWithPreconditionerExample.out
 */
template< typename Matrix >
class Preconditioner
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
       * \brief This method defines configuration entries for setup of the preconditioner of linear iterative solver.
       *
       * \param config contains description of configuration parameters.
       * \param prefix is a prefix of particular configuration entries.
       */
      static void configSetup( Config::ConfigDescription& config,
                              const String& prefix = "" ) {}

      /**
       * \brief Method for setup of the preconditioner of linear iterative solver based on configuration parameters.
       *
       * \param parameters contains values of the define configuration entries.
       * \param prefix is a prefix of particular configuration entries.
       */
      virtual bool setup( const Config::ParameterContainer& parameters,
                        const String& prefix = "" )
      {
         return true;
      }

      /**
       * \brief This method updates the preconditioner with respect to given matrix.
       *
       * \param matrixPointer smart pointer (\ref std::shared_ptr) to matrix the preconditioner is related to.
       */
      virtual void update( const MatrixPointer& matrixPointer )
      {}

      /**
       * \brief This method applies the preconditioner.
       *
       * \param b is the input vector the preconditioner is applied on.
       * \param x is the result of the preconditioning.
       */
      virtual void solve( ConstVectorViewType b, VectorViewType x ) const
      {
         throw std::logic_error("The solve() method of a dummy preconditioner should not be called.");
      }

      /**
       * \brief Destructor of the preconditioner.
       */
      virtual ~Preconditioner() {}
};

         } // namespace Preconditioners
      } // namespace Linear
   } // namespace Solvers
} // namespace noa::TNL
