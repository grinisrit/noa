// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include "Preconditioner.h"

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/TNL/Pointers/UniquePointer.h>
#include <noa/3rdparty/TNL/Exceptions/NotImplementedError.h>

#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
#include <cusparse.h>
#endif

namespace noaTNL {
   namespace Solvers {
      namespace Linear {
      namespace Preconditioners {

// implementation template
template< typename Matrix, typename Real, typename Device, typename Index >
class ILU0_impl
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for the matrix type " + getType< Matrix >());
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for the matrix type " + getType< Matrix >());
   }
};

/**
 * \brief Implementation of a preconditioner based on Incomplete LU.
 *
 * See [detailed description](https://en.wikipedia.org/wiki/Incomplete_LU_factorization).
 *
 * \tparam Matrix is type of the matrix describing the linear system.
 */
template< typename Matrix >
class ILU0
: public ILU0_impl< Matrix, typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >
{};

/**
 * \brief Implementation of a preconditioner based in Incomplete LU - specialization for CPU.
 *
 * See [detailed description](https://en.wikipedia.org/wiki/Incomplete_LU_factorization).
 *
 * See \ref noaTNL::Solvers::Linear::Preconditioners::Preconditioner for example of setup with a linear solver.
 *
 * \tparam Matrix is type of the matrix describing the linear system.
 */
template< typename Matrix, typename Real, typename Index >
class ILU0_impl< Matrix, Real, Devices::Host, Index >
: public Preconditioner< Matrix >
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
      virtual void update( const MatrixPointer& matrixPointer ) override;

      /**
       * \brief This method applies the preconditioner.
       *
       * \param b is the input vector the preconditioner is applied on.
       * \param x is the result of the preconditioning.
       */
      virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

   protected:
      // The factors L and U are stored separately and the rows of U are reversed.
      using CSR = Matrices::SparseMatrix< RealType, DeviceType, IndexType, Matrices::GeneralMatrix, Algorithms::Segments::CSRScalar >;
      CSR L, U;

      // Specialized methods to distinguish between normal and distributed matrices
      // in the implementation.
      template< typename M >
      static IndexType getMinColumn( const M& m )
      {
         return 0;
      }

      template< typename M >
      static IndexType getMinColumn( const Matrices::DistributedMatrix< M >& m )
      {
         if( m.getRows() == m.getColumns() )
            // square matrix, assume global column indices
            return m.getLocalRowRange().getBegin();
         else
            // non-square matrix, assume ghost indexing
            return 0;
      }
};

template< typename Matrix >
class ILU0_impl< Matrix, double, Devices::Cuda, int >
: public Preconditioner< Matrix >
{
public:
   using RealType = double;
   using DeviceType = Devices::Cuda;
   using IndexType = int;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   ILU0_impl()
   {
#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
      cusparseCreate( &handle );
#endif
   }

   virtual void update( const MatrixPointer& matrixPointer ) override;

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

   ~ILU0_impl()
   {
#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
      resetMatrices();
      cusparseDestroy( handle );
#endif
   }

   // must be public because nvcc does not allow extended lambdas in private or protected regions
   void allocate_LU();
   void copy_triangular_factors();
protected:

#if defined(HAVE_CUDA) && defined(HAVE_CUSPARSE)
   using CSR = Matrices::SparseMatrix< RealType, DeviceType, IndexType, Matrices::GeneralMatrix, Algorithms::Segments::CSRScalar >;
   Pointers::UniquePointer< CSR > A, L, U;
   Containers::Vector< RealType, DeviceType, IndexType > y;

   cusparseHandle_t handle;

   cusparseMatDescr_t descr_A = 0;
   cusparseMatDescr_t descr_L = 0;
   cusparseMatDescr_t descr_U = 0;
   csrilu02Info_t     info_A  = 0;
   csrsv2Info_t       info_L  = 0;
   csrsv2Info_t       info_U  = 0;

   const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
//   const cusparseSolvePolicy_t policy_A = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
//   const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
//   const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
   const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
   const cusparseOperation_t trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;

   Containers::Array< char, DeviceType, int > pBuffer;

   // scaling factor for triangular solves
   const double alpha = 1.0;

   void resetMatrices()
   {
      if( descr_A ) {
         cusparseDestroyMatDescr( descr_A );
         descr_A = 0;
      }
      if( descr_L ) {
         cusparseDestroyMatDescr( descr_L );
         descr_L = 0;
      }
      if( descr_U ) {
         cusparseDestroyMatDescr( descr_U );
         descr_U = 0;
      }
      if( info_A ) {
         cusparseDestroyCsrilu02Info( info_A );
         info_A = 0;
      }
      if( info_L ) {
         cusparseDestroyCsrsv2Info( info_L );
         info_L = 0;
      }
      if( info_U ) {
         cusparseDestroyCsrsv2Info( info_U );
         info_U = 0;
      }
      pBuffer.reset();
   }
#endif
};

template< typename Matrix >
class ILU0_impl< Matrices::DistributedMatrix< Matrix >, double, Devices::Cuda, int >
: public Preconditioner< Matrices::DistributedMatrix< Matrix > >
{
   using MatrixType = Matrices::DistributedMatrix< Matrix >;
public:
   using RealType = double;
   using DeviceType = Devices::Cuda;
   using IndexType = int;
   using typename Preconditioner< MatrixType >::VectorViewType;
   using typename Preconditioner< MatrixType >::ConstVectorViewType;
   using typename Preconditioner< MatrixType >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for CUDA and distributed matrices.");
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw Exceptions::NotImplementedError("ILU0 is not implemented yet for CUDA and distributed matrices.");
   }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace noaTNL

#include "ILU0.hpp"
