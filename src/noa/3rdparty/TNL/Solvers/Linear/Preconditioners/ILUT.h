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

namespace noaTNL {
namespace Solvers {
namespace Linear {
namespace Preconditioners {

// implementation template
template< typename Matrix, typename Real, typename Device, typename Index >
class ILUT_impl
{};

/**
 * \brief Implementation of a preconditioner based on Incomplete LU with thresholding.
 *
 * See [detailed description](https://www-users.cse.umn.edu/~saad/PDF/umsi-92-38.pdf)
 *
 * See \ref noaTNL::Solvers::Linear::Preconditioners::Preconditioner for example of setup with a linear solver.
 *
 * \tparam Matrix is type of the matrix describing the linear system.
 */
template< typename Matrix >
class ILUT
: public ILUT_impl< Matrix, typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >
{
   using Base = ILUT_impl< Matrix, typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   public:

      /**
       * \brief Floating point type used for computations.
       */
      using RealType = typename Matrix::RealType;

      /**
       * \brief Device where the preconditioner will run on and auxillary data will alloacted on.
       */
      using DeviceType = Devices::Host;

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
       * \brief Type of shared pointer to the matrix.
       */
      using typename Preconditioner< Matrix >::MatrixPointer;

      /**
       * \brief This method defines configuration entries for setup of the preconditioner of linear iterative solver.
       *
       * \param config contains description of configuration parameters.
       * \param prefix is a prefix of particular configuration entries.
       */
      static void configSetup( Config::ConfigDescription& config,
                              const String& prefix = "" )
      {
         config.addEntry< int >( prefix + "ilut-p", "Number of additional non-zero entries to allocate on each row of the factors L and U.", 0 );
         config.addEntry< double >( prefix + "ilut-threshold", "Threshold for droppping small entries.", 1e-4 );
      }

      /**
       * \brief This method updates the preconditioner with respect to given matrix.
       *
       * \param matrixPointer smart pointer (\ref std::shared_ptr) to matrix the preconditioner is related to.
       */
      using Base::update;

      /**
       * \brief This method applies the preconditioner.
       *
       * \param b is the input vector the preconditioner is applied on.
       * \param x is the result of the preconditioning.
       */
      using Base::solve;
};

template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::Host, Index >
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Host;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override;

   virtual void update( const MatrixPointer& matrixPointer ) override;

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override;

protected:
   Index p = 0;
   Real tau = 1e-4;

   // The factors L and U are stored separately and the rows of U are reversed.
   Matrices::SparseMatrix< RealType, DeviceType, IndexType, Matrices::GeneralMatrix, Algorithms::Segments::CSRDefault > L, U;

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

template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::Sequential, Index >
: public ILUT_impl< Matrix, Real, Devices::Host, Index >
{};


template< typename Matrix, typename Real, typename Index >
class ILUT_impl< Matrix, Real, Devices::Cuda, Index >
: public Preconditioner< Matrix >
{
public:
   using RealType = Real;
   using DeviceType = Devices::Cuda;
   using IndexType = Index;
   using typename Preconditioner< Matrix >::VectorViewType;
   using typename Preconditioner< Matrix >::ConstVectorViewType;
   using typename Preconditioner< Matrix >::MatrixPointer;

   virtual void update( const MatrixPointer& matrixPointer ) override
   {
      throw std::runtime_error("Not Iplemented yet for CUDA");
   }

   virtual void solve( ConstVectorViewType b, VectorViewType x ) const override
   {
      throw std::runtime_error("Not Iplemented yet for CUDA");
   }
};

} // namespace Preconditioners
} // namespace Linear
} // namespace Solvers
} // namespace noaTNL

#include "ILUT.hpp"
