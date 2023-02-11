// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Preconditioner.h"

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/UniquePointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

// implementation template
template< typename Matrix, typename Real, typename Device, typename Index >
class ILU0_impl;

/**
 * \brief Implementation of a preconditioner based on Incomplete LU.
 *
 * See [detailed description](https://en.wikipedia.org/wiki/Incomplete_LU_factorization).
 *
 * \tparam Matrix is type of the matrix describing the linear system.
 */
template< typename Matrix >
class ILU0 : public ILU0_impl< Matrix, typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >
{};

/**
 * \brief Implementation of a preconditioner based in Incomplete LU - specialization for CPU.
 *
 * See [detailed description](https://en.wikipedia.org/wiki/Incomplete_LU_factorization).
 *
 * See \ref TNL::Solvers::Linear::Preconditioners::Preconditioner for example of setup with a linear solver.
 *
 * \tparam Matrix is type of the matrix describing the linear system.
 */
template< typename Matrix, typename Real, typename Index >
class ILU0_impl< Matrix, Real, Devices::Host, Index > : public Preconditioner< Matrix >
{
public:
   /**
    * \brief Floating point type used for computations.
    */
   using RealType = Real;

   /**
    * \brief Device where the preconditioner will run on and auxillary data will alloacted on.
    */
   using DeviceType = Devices::Host;

   /**
    * \brief Type for indexing.
    */
   using IndexType = Index;

   /**
    * \brief Type for vector view.
    */
   using typename Preconditioner< Matrix >::VectorViewType;

   /**
    * \brief Type for constant vector view.
    */
   using typename Preconditioner< Matrix >::ConstVectorViewType;

   /**
    * \brief Type of shared pointer to the matrix.
    */
   using typename Preconditioner< Matrix >::MatrixPointer;

   /**
    * \brief This method updates the preconditioner with respect to given matrix.
    *
    * \param matrixPointer smart pointer (\ref std::shared_ptr) to matrix the preconditioner is related to.
    */
   void
   update( const MatrixPointer& matrixPointer ) override;

   /**
    * \brief This method applies the preconditioner.
    *
    * \param b is the input vector the preconditioner is applied on.
    * \param x is the result of the preconditioning.
    */
   void
   solve( ConstVectorViewType b, VectorViewType x ) const override;

protected:
   // The factors L and U are stored separately and the rows of U are reversed.
   using CSR =
      Matrices::SparseMatrix< RealType, DeviceType, IndexType, Matrices::GeneralMatrix, Algorithms::Segments::CSRScalar >;
   CSR L, U;

   // Specialized methods to distinguish between normal and distributed matrices
   // in the implementation.
   template< typename M >
   static IndexType
   getMinColumn( const M& m )
   {
      return 0;
   }

   template< typename M >
   static IndexType
   getMinColumn( const Matrices::DistributedMatrix< M >& m )
   {
      if( m.getRows() == m.getColumns() )
         // square matrix, assume global column indices
         return m.getLocalRowRange().getBegin();
      else
         // non-square matrix, assume ghost indexing
         return 0;
   }
};

template< typename Matrix, typename Real, typename Index >
class ILU0_impl< Matrix, Real, Devices::Sequential, Index > : public ILU0_impl< Matrix, Real, Devices::Host, Index >
{};

template< typename Matrix, typename Real, typename Index >
class ILU0_impl< Matrix, Real, Devices::Cuda, Index > : public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Cuda;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   void
   update( const MatrixPointer& matrixPointer ) override
   {
      throw Exceptions::NotImplementedError( "ILU0 is not implemented for CUDA" );
   }

   void
   solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw Exceptions::NotImplementedError( "ILU0 is not implemented for CUDA" );
   }
};

}  // namespace Preconditioners
}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL

#include "ILU0.hpp"
