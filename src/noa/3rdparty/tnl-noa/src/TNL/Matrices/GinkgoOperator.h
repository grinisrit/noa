// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/VectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>

#include <ginkgo/ginkgo.hpp>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief Creates a Ginkgo Csr matrix view from a TNL CSR matrix.
 *
 * \ingroup Ginkgo
 */
template< typename Matrix >
auto
getGinkgoMatrixCsrView( std::shared_ptr< const gko::Executor > exec, Matrix& matrix )
   -> std::unique_ptr< gko::matrix::Csr< typename Matrix::RealType, typename Matrix::IndexType > >
{
   return gko::matrix::Csr< typename Matrix::RealType, typename Matrix::IndexType >::create(
      exec,
      gko::dim< 2 >{ static_cast< std::size_t >( matrix.getRows() ), static_cast< std::size_t >( matrix.getColumns() ) },
      gko::make_array_view( exec, matrix.getNonzeroElementsCount(), matrix.getValues().getData() ),
      gko::make_array_view( exec, matrix.getNonzeroElementsCount(), matrix.getColumnIndexes().getData() ),
      gko::make_array_view( exec, matrix.getRows() + 1, matrix.getSegments().getOffsets().getData() ) );
}

/**
 * \brief Converts any TNL sparse matrix to a Ginkgo Csr matrix.
 *
 * \ingroup Ginkgo
 */
template< typename Matrix >
auto
getGinkgoMatrixCsr( std::shared_ptr< const gko::Executor > exec, Matrix& matrix )
   -> std::unique_ptr< gko::matrix::Csr< typename Matrix::RealType, typename Matrix::IndexType > >
{
   // convert to CSR in TNL
   using CSR = SparseMatrix< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   CSR matrix_csr;
   matrix_csr = matrix;

   // extract views
   auto values = gko::make_array_view( exec, matrix_csr.getNonzeroElementsCount(), matrix_csr.getValues().getData() );
   auto column_indexes =
      gko::make_array_view( exec, matrix_csr.getNonzeroElementsCount(), matrix_csr.getColumnIndexes().getData() );
   auto row_offsets = gko::make_array_view( exec, matrix_csr.getRows() + 1, matrix_csr.getSegments().getOffsets().getData() );

   // create gko::matrix::Csr by cloning the arrays
   return gko::matrix::Csr< typename Matrix::RealType, typename Matrix::IndexType >::create(
      exec,
      gko::dim< 2 >{ static_cast< std::size_t >( matrix.getRows() ), static_cast< std::size_t >( matrix.getColumns() ) },
      gko::array< typename Matrix::RealType >( values ),
      gko::array< typename Matrix::IndexType >( column_indexes ),
      gko::array< typename Matrix::IndexType >( row_offsets ) );
}

/**
 * \brief Wraps a general TNL matrix as a Ginkgo \e LinOp.
 *
 * \note This \e LinOp cannot be used with preconditioners that require a CSR
 * matrix, see https://github.com/ginkgo-project/ginkgo/issues/1094. If you
 * have a CSR matrix in TNL already, use \ref getGinkgoMatrixCsrView. If you
 * have a TNL matrix in a different format, use \ref getGinkgoMatrixCsr to do
 * the conversion.
 *
 * \ingroup Ginkgo
 */
template< typename Matrix >
class GinkgoOperator : public gko::EnableLinOp< GinkgoOperator< Matrix > >,
                       public gko::EnableCreateMethod< GinkgoOperator< Matrix > >
{
public:
   // dummy constructor to make Ginkgo happy
   GinkgoOperator( std::shared_ptr< const gko::Executor > exec )
   : gko::EnableLinOp< GinkgoOperator< Matrix > >( exec, gko::dim< 2 >{ 0, 0 } ),
     gko::EnableCreateMethod< GinkgoOperator< Matrix > >()
   {}

   GinkgoOperator( std::shared_ptr< const gko::Executor > exec, const Matrix& matrix )
   : gko::EnableLinOp< GinkgoOperator< Matrix > >(
      exec,
      gko::dim< 2 >{ static_cast< std::size_t >( matrix.getRows() ), static_cast< std::size_t >( matrix.getColumns() ) } ),
     gko::EnableCreateMethod< GinkgoOperator< Matrix > >()
   {
      wrapped_matrix = &matrix;
   }

protected:
   using ValueType = typename Matrix::RealType;
   using ViewType = Containers::VectorView< ValueType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   using ConstViewType = typename ViewType::ConstViewType;

   //! \brief Applies the operator on a vector: computes the expression
   //! `x = op(b)`
   void
   apply_impl( const gko::LinOp* b, gko::LinOp* x ) const override
   {
      // cast to Dense
      const auto b_vec = gko::as< const gko::matrix::Dense< ValueType > >( b );
      auto x_vec = gko::as< gko::matrix::Dense< ValueType > >( x );

      // create TNL views
      ConstViewType b_view( b_vec->get_const_values(), b_vec->get_num_stored_elements() );
      ViewType x_view( x_vec->get_values(), x_vec->get_num_stored_elements() );

      wrapped_matrix->vectorProduct( b_view, x_view );
   }

   //! \brief Applies the operator on a vector: computes the expression
   //! `x = alpha * op(b) + beta * x`
   void
   apply_impl( const gko::LinOp* alpha, const gko::LinOp* b, const gko::LinOp* beta, gko::LinOp* x ) const override
   {
      // cast to Dense
      const auto b_vec = gko::as< const gko::matrix::Dense< ValueType > >( b );
      auto x_vec = gko::as< gko::matrix::Dense< ValueType > >( x );

      // create TNL views
      ConstViewType b_view( b_vec->get_const_values(), b_vec->get_num_stored_elements() );
      ViewType x_view( x_vec->get_values(), x_vec->get_num_stored_elements() );

      // Check that alpha and beta are Dense<ValueType> of size (1,1):
      if( alpha->get_size()[ 0 ] > 1 || alpha->get_size()[ 1 ] > 1 )
         throw gko::BadDimension( __FILE__,
                                  __LINE__,
                                  __func__,
                                  "alpha",
                                  alpha->get_size()[ 0 ],
                                  alpha->get_size()[ 1 ],
                                  "Expected an object of size [1 x 1] for scaling "
                                  " in this operator's apply_impl" );
      if( beta->get_size()[ 0 ] > 1 || beta->get_size()[ 1 ] > 1 )
         throw gko::BadDimension( __FILE__,
                                  __LINE__,
                                  __func__,
                                  "beta",
                                  beta->get_size()[ 0 ],
                                  beta->get_size()[ 1 ],
                                  "Expected an object of size [1 x 1] for scaling "
                                  " in this operator's apply_impl" );

      ValueType alpha_f;
      ValueType beta_f;

      if( alpha->get_executor() == alpha->get_executor()->get_master() )
         // Access value directly
         alpha_f = gko::as< gko::matrix::Dense< ValueType > >( alpha )->at( 0, 0 );
      else
         // Copy from device to host
         this->get_executor()->get_master().get()->copy_from(
            this->get_executor().get(), 1, gko::as< gko::matrix::Dense< ValueType > >( alpha )->get_const_values(), &alpha_f );

      if( beta->get_executor() == beta->get_executor()->get_master() )
         // Access value directly
         beta_f = gko::as< gko::matrix::Dense< ValueType > >( beta )->at( 0, 0 );
      else
         // Copy from device to host
         this->get_executor()->get_master().get()->copy_from(
            this->get_executor().get(), 1, gko::as< gko::matrix::Dense< ValueType > >( beta )->get_const_values(), &beta_f );

      wrapped_matrix->vectorProduct( b_view, x_view, alpha_f, beta_f );
   }

private:
   const Matrix* wrapped_matrix;
};

}  // namespace Matrices
}  // namespace noa::TNL
