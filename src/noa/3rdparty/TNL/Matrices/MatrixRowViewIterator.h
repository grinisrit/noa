// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Matrices/SparseMatrixElement.h>

namespace TNL {
namespace Matrices {

template< typename RowView >
class MatrixRowViewIterator
{

   public:

      /**
       * \brief Type of SparseMatrixRowView
       */
      using RowViewType = RowView;

      /**
       * \brief The type of matrix elements.
       */
      using RealType = typename RowViewType::RealType;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = typename RowViewType::IndexType;

      /**
       * \brief The type of related matrix element.
       */
      using MatrixElementType = typename RowView::MatrixElementType;

      /**
       * \brief Tells whether the parent matrix is a binary matrix.
       * @return `true` if the matrix is binary.
       */
      static constexpr bool isBinary() { return RowViewType::isBinary(); };

      __cuda_callable__
      MatrixRowViewIterator( RowViewType& rowView,
                                   const IndexType& localIdx );

      /**
       * \brief Comparison of two matrix row iterators.
       *
       * \param other is another matrix row iterator.
       * \return \e true if both iterators points at the same point of the same matrix, \e false otherwise.
       */
      __cuda_callable__
      bool operator==( const MatrixRowViewIterator& other ) const;

      /**
       * \brief Comparison of two matrix row iterators.
       *
       * \param other is another matrix row iterator.
       * \return \e false if both iterators points at the same point of the same matrix, \e true otherwise.
       */
      __cuda_callable__
      bool operator!=( const MatrixRowViewIterator& other ) const;

      /**
       * \brief Increment operator.
       */
      __cuda_callable__
      MatrixRowViewIterator& operator++();

      /**
       * \brief Decrement operetor.
       */
      __cuda_callable__
      MatrixRowViewIterator& operator--();

      /**
       * \brief Dereference operator.
       */
      __cuda_callable__
      MatrixElementType operator*();

      /**
       * \brief Dereference operator for constant instances.
       */
      __cuda_callable__
      const MatrixElementType operator*() const;

   protected:

      RowViewType& rowView;

      IndexType localIdx = 0;
};


   } // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/MatrixRowViewIterator.hpp>
