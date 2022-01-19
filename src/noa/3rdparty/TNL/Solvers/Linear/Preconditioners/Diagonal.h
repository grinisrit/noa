// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Preconditioner.h"


namespace TNL {
   namespace Solvers {
      namespace Linear {
         namespace Preconditioners {

/**
 * \brief Diagonal (Jacobi) preconditioner for iterative solvers of linear systems.
 *
 * See [detailed description]([Netlib](http://netlib.org/linalg/html_templates/node55.html)).
 *
 * See \ref TNL::Solvers::Linear::Preconditioners::Preconditioner for example of setup with a linear solver.
 *
 * \tparam Matrix is type of the matrix describing the linear system.
 */
template< typename Matrix >
class Diagonal
: public Preconditioner< Matrix >
{
   public:

      /**
       * \brief Floating point type used for computations.
       */
      using RealType = typename Matrix::RealType;

      /**
       * \brief Device where the preconditioner will run on and auxillary data will alloacted on.
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
      using typename Preconditioner< Matrix >::VectorViewType;

      /**
       * \brief Type for constant vector view.
       */
      using typename Preconditioner< Matrix >::ConstVectorViewType;

      /**
       * \brief Type of the matrix representing the linear system.
       */
      using MatrixType = Matrix;

      /**
       * \brief Type of shared pointer to the matrix.
       */
      using typename Preconditioner< Matrix >::MatrixPointer;

      /**
       * \brief This method updates the preconditioner with respect to given matrix.
       *
       * \param matrixPointer smart pointer (\ref std::shared_ptr) to matrix the preconditioner is related to.
       */
      virtual void update( const MatrixPointer& matrixPointer ) override;

      /**
       * \brief This method applies the preconditioner.
       *
       * \param b is the input vector the preconditioner is applied on.
       * \param x is the result of the preconditioning.
       */
      virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

   protected:

      using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

      VectorType diagonal;
};

/**
 * \brief Specialization of the diagonal preconditioner for distributed matrices.
 *
 * See \ref TNL::Solvers::Linear::Preconditioners::Diagonal
 *
 * \tparam Matrix is a type of matrix describing the linear system.
 */
template< typename Matrix >
class Diagonal< Matrices::DistributedMatrix< Matrix > >
: public Preconditioner< Matrices::DistributedMatrix< Matrix > >
{
   public:

      /**
       * \brief Type of the matrix representing the linear system.
       */
      using MatrixType = Matrices::DistributedMatrix< Matrix >;

      /**
       * \brief Floating point type used for computations.
       */
      using RealType = typename MatrixType::RealType;

      /**
       * \brief Device where the solver will run on and auxillary data will alloacted on.
       *
       * See \ref Devices::Host or \ref Devices::Cuda.
       */
      using DeviceType = typename MatrixType::DeviceType;

      /**
       * \brief Type for indexing.
       */
      using IndexType = typename MatrixType::IndexType;

      /**
       * \brief Type for vector view.
       */
      using typename Preconditioner< MatrixType >::VectorViewType;

      /**
       * \brief Type for vector constant view.
       */
      using typename Preconditioner< MatrixType >::ConstVectorViewType;

      /**
       * \brief Type of shared pointer to the matrix.
       */
      using typename Preconditioner< MatrixType >::MatrixPointer;

      /**
       * \brief This method updates the preconditioner with respect to given matrix.
       *
       * \param matrixPointer smart pointer (\ref std::shared_ptr) to matrix the preconditioner is related to.
       */
      virtual void update( const MatrixPointer& matrixPointer ) override;

      /**
       * \brief This method applies the preconditioner.
       *
       * \param b is the input vector the preconditioner is applied on.
       * \param x is the result of the preconditioning.
       */
      virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

   protected:
      using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
      using LocalViewType = Containers::VectorView< RealType, DeviceType, IndexType >;
      using ConstLocalViewType = Containers::VectorView< std::add_const_t< RealType >, DeviceType, IndexType >;

      VectorType diagonal;
};

         } // namespace Preconditioners
      } // namespace Linear
   } // namespace Solvers
} // namespace TNL

#include "Diagonal.hpp"
