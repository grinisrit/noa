// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include <noa/3rdparty/tnl-noa/src/TNL/Hypre.h>

   #include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief Wrapper for Hypre's sequential CSR matrix.
 *
 * Links to upstream sources:
 * - https://github.com/hypre-space/hypre/blob/master/src/seq_mv/csr_matrix.h
 * - https://github.com/hypre-space/hypre/blob/master/src/seq_mv/csr_matrix.c
 * - https://github.com/hypre-space/hypre/blob/master/src/seq_mv/seq_mv.h (catch-all interface)
 *
 * \ingroup Hypre
 */
class HypreCSRMatrix
{
public:
   using RealType = HYPRE_Real;
   using ValueType = RealType;
   using DeviceType = HYPRE_Device;
   using IndexType = HYPRE_Int;

   using MatrixType = SparseMatrix< RealType, DeviceType, IndexType, GeneralMatrix, Algorithms::Segments::CSRDefault >;
   using ViewType = typename MatrixType::ViewType;
   using ConstViewType = typename MatrixType::ConstViewType;

   using ValuesViewType = Containers::VectorView< RealType, DeviceType, IndexType >;
   using ConstValuesViewType = typename ValuesViewType::ConstViewType;
   using ColumnIndexesVectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
   using ColumnIndexesViewType = typename ColumnIndexesVectorType::ViewType;
   using ConstColumnIndexesViewType = typename ColumnIndexesVectorType::ConstViewType;
   using SegmentsViewType = Algorithms::Segments::CSRViewDefault< DeviceType, IndexType >;
   using ConstSegmentsViewType = Algorithms::Segments::CSRViewDefault< DeviceType, std::add_const_t< IndexType > >;

   HypreCSRMatrix() = default;

   // TODO: behavior should depend on "owns_data" (shallow vs deep copy)
   HypreCSRMatrix( const HypreCSRMatrix& other ) = delete;

   HypreCSRMatrix( HypreCSRMatrix&& other ) noexcept : m( other.m ), owns_handle( other.owns_handle )
   {
      other.m = nullptr;
   }

   // TODO should do a deep copy
   HypreCSRMatrix&
   operator=( const HypreCSRMatrix& other ) = delete;

   HypreCSRMatrix&
   operator=( HypreCSRMatrix&& other ) noexcept
   {
      m = other.m;
      other.m = nullptr;
      owns_handle = other.owns_handle;
      return *this;
   }

   HypreCSRMatrix( IndexType rows,
                   IndexType columns,
                   ValuesViewType values,
                   ColumnIndexesViewType columnIndexes,
                   ColumnIndexesViewType rowOffsets )
   {
      bind( rows, columns, std::move( values ), std::move( columnIndexes ), std::move( rowOffsets ) );
   }

   HypreCSRMatrix( ViewType view )
   {
      bind( std::move( view ) );
   }

   /**
    * \brief Convert Hypre's format to \e HypreCSRMatrix
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the matrix should take ownership of
    * the handle, i.e. whether to call \e hypre_CSRMatrixDestroy when it does
    * not need it anymore.
    */
   explicit HypreCSRMatrix( hypre_CSRMatrix* handle, bool take_ownership = true )
   {
      bind( handle, take_ownership );
   }

   operator const hypre_CSRMatrix*() const noexcept
   {
      return m;
   }

   operator hypre_CSRMatrix*() noexcept
   {
      return m;
   }

   // HYPRE_CSRMatrix is "equivalent" to pointer to hypre_CSRMatrix, but requires
   // ugly C-style cast on the pointer (which is done even in Hypre itself)
   // https://github.com/hypre-space/hypre/blob/master/src/seq_mv/HYPRE_csr_matrix.c
   operator HYPRE_CSRMatrix() noexcept
   {
      return (HYPRE_CSRMatrix) m;
   }

   ~HypreCSRMatrix()
   {
      reset();
   }

   IndexType
   getRows() const
   {
      if( m == nullptr )
         return 0;
      return hypre_CSRMatrixNumRows( m );
   }

   IndexType
   getColumns() const
   {
      if( m == nullptr )
         return 0;
      return hypre_CSRMatrixNumCols( m );
   }

   IndexType
   getNonzeroElementsCount() const
   {
      if( m == nullptr )
         return 0;
      return hypre_CSRMatrixNumNonzeros( m );
   }

   ConstValuesViewType
   getValues() const
   {
      if( m == nullptr )
         return {};
      return { hypre_CSRMatrixData( m ), hypre_CSRMatrixNumNonzeros( m ) };
   }

   ValuesViewType
   getValues()
   {
      if( m == nullptr )
         return {};
      return { hypre_CSRMatrixData( m ), hypre_CSRMatrixNumNonzeros( m ) };
   }

   ConstColumnIndexesViewType
   getColumnIndexes() const
   {
      if( m == nullptr )
         return {};
      static_assert( std::is_same< HYPRE_Int, HYPRE_BigInt >::value,
                     "The J array cannot be accessed via this method when HYPRE_Int and HYPRE_BigInt are different types." );
      if( hypre_CSRMatrixBigJ( m ) != nullptr )
         return { hypre_CSRMatrixBigJ( m ), hypre_CSRMatrixNumNonzeros( m ) };
      return { hypre_CSRMatrixJ( m ), hypre_CSRMatrixNumNonzeros( m ) };
   }

   ColumnIndexesViewType
   getColumnIndexes()
   {
      if( m == nullptr )
         return {};
      static_assert( std::is_same< HYPRE_Int, HYPRE_BigInt >::value,
                     "The J array cannot be accessed via this method when HYPRE_Int and HYPRE_BigInt are different types." );
      if( hypre_CSRMatrixBigJ( m ) != nullptr )
         return { hypre_CSRMatrixBigJ( m ), hypre_CSRMatrixNumNonzeros( m ) };
      return { hypre_CSRMatrixJ( m ), hypre_CSRMatrixNumNonzeros( m ) };
   }

   ConstColumnIndexesViewType
   getRowOffsets() const
   {
      if( m == nullptr )
         return {};
      if( hypre_CSRMatrixI( m ) == nullptr )
         return {};
      return { hypre_CSRMatrixI( m ), hypre_CSRMatrixNumRows( m ) + 1 };
   }

   ColumnIndexesViewType
   getRowOffsets()
   {
      if( m == nullptr )
         return {};
      if( hypre_CSRMatrixI( m ) == nullptr )
         return {};
      return { hypre_CSRMatrixI( m ), hypre_CSRMatrixNumRows( m ) + 1 };
   }

   ConstSegmentsViewType
   getSegments() const
   {
      if( m == nullptr )
         return {};
      return { getRowOffsets(), typename ConstSegmentsViewType::KernelType{} };
   }

   SegmentsViewType
   getSegments()
   {
      if( m == nullptr )
         return {};
      return { getRowOffsets(), typename SegmentsViewType::KernelType{} };
   }

   ConstViewType
   getConstView() const
   {
      return { getRows(), getColumns(), getValues(), getColumnIndexes(), getSegments() };
   }

   ViewType
   getView()
   {
      return { getRows(), getColumns(), getValues(), getColumnIndexes(), getSegments() };
   }

   /**
    * \brief Drop previously set data (deallocate if the matrix was the owner)
    * and bind to the given data (i.e., the matrix does not become the owner).
    */
   void
   bind( IndexType rows,
         IndexType columns,
         ValuesViewType values,
         ColumnIndexesViewType columnIndexes,
         ColumnIndexesViewType rowOffsets )
   {
      TNL_ASSERT_EQ( rowOffsets.getSize(), rows + 1, "wrong size of rowOffsets" );
      TNL_ASSERT_EQ( values.getSize(), rowOffsets.getElement( rows ), "wrong size of values" );
      TNL_ASSERT_EQ( columnIndexes.getSize(), rowOffsets.getElement( rows ), "wrong size of columnIndexes" );

      // drop/deallocate the current data
      reset();

      // create handle for the matrix
      m = hypre_CSRMatrixCreate( rows, columns, rowOffsets.getElement( rows ) );
      hypre_CSRMatrixMemoryLocation( m ) = getHypreMemoryLocation();

      // set view data
      hypre_CSRMatrixOwnsData( m ) = 0;
      hypre_CSRMatrixData( m ) = values.getData();
      hypre_CSRMatrixJ( m ) = columnIndexes.getData();
      hypre_CSRMatrixI( m ) = rowOffsets.getData();
   }

   void
   bind( ViewType view )
   {
      bind( view.getRows(), view.getColumns(), view.getValues(), view.getColumnIndexes(), view.getSegments().getOffsets() );
   }

   void
   bind( MatrixType& matrix )
   {
      bind( matrix.getView() );
   }

   void
   bind( HypreCSRMatrix& matrix )
   {
      bind( matrix.getRows(), matrix.getColumns(), matrix.getValues(), matrix.getColumnIndexes(), matrix.getRowOffsets() );
   }

   /**
    * \brief Convert Hypre's format to \e HypreCSRMatrix
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the matrix should take ownership of
    * the handle, i.e. whether to call \e hypre_CSRMatrixDestroy when it does
    * not need it anymore.
    */
   void
   bind( hypre_CSRMatrix* handle, bool take_ownership = true )
   {
      // drop/deallocate the current data
      reset();

      // set the handle and ownership flag
      m = handle;
      owns_handle = take_ownership;
   }

   //! \brief Reset the matrix to empty state.
   void
   reset()
   {
      if( owns_handle && m != nullptr ) {
         // FIXME: workaround for https://github.com/hypre-space/hypre/issues/621
         if( ! hypre_CSRMatrixOwnsData( m ) )
            // prevent deallocation of the "I" array when the handle does not own it
            hypre_CSRMatrixI( m ) = nullptr;

         hypre_CSRMatrixDestroy( m );
         m = nullptr;
      }
      else
         m = nullptr;
      owns_handle = true;
   }

   /**
    * \brief Set the new matrix dimensions.
    *
    * - if the matrix previously owned data, they are deallocated
    * - new size is set
    * - the matrix is initialized with \e hypre_CSRMatrixInitialize
    *   (i.e., data are allocated)
    */
   void
   setDimensions( IndexType rows, IndexType cols )
   {
      reset();
      m = hypre_CSRMatrixCreate( rows, cols, 0 );
      hypre_CSRMatrixMemoryLocation( m ) = getHypreMemoryLocation();
      hypre_CSRMatrixInitialize( m );
   }

   template< typename RowsCapacitiesVector >
   void
   setRowCapacities( const RowsCapacitiesVector& rowCapacities )
   {
      TNL_ASSERT_EQ(
         rowCapacities.getSize(), this->getRows(), "Number of matrix rows does match the rowCapacities vector size." );

      const IndexType nonzeros = TNL::sum( rowCapacities );
      hypre_CSRMatrixResize( m, getRows(), getColumns(), nonzeros );

      // initialize row pointers
      auto rowOffsets = getRowOffsets();
      TNL_ASSERT_EQ(
         rowOffsets.getSize(), rowCapacities.getSize() + 1, "The size of the rowOffsets vector does not match rowCapacities" );
      // GOTCHA: when rowCapacities.getSize() == 0, getView returns a full view with size == 1
      if( rowCapacities.getSize() > 0 ) {
         auto view = rowOffsets.getView( 0, rowCapacities.getSize() );
         view = rowCapacities;
      }
      rowOffsets.setElement( rowCapacities.getSize(), 0 );
      Algorithms::inplaceExclusiveScan( rowOffsets );

      // reset column indices with the padding index
      getColumnIndexes().setValue( -1 );
   }

   /**
    * \brief Reorders the column and data arrays of a square matrix, such that
    * the first entry in each row is the diagonal one.
    *
    * Note that the \e hypre_CSRMatrixReorder function only swaps the diagonal
    * and first entries in the row, but this function also shifts the
    * subdiagonal entries to ensure that the column indices (except for the
    * diagonal one) remain in the original order.
    */
   void
   reorderDiagonalEntries()
   {
      // operation does not make sense for non-square matrices
      if( getRows() != getColumns() )
         return;

      getView().forAllRows(
         [] __cuda_callable__( typename ViewType::RowView & row ) mutable
         {
            const IndexType j_diag = row.getRowIndex();
            IndexType c_diag = 0;
            RealType v_diag;

            // find the diagonal entry
            for( IndexType c = 0; c < row.getSize(); c++ )
               if( row.getColumnIndex( c ) == j_diag ) {
                  c_diag = c;
                  v_diag = row.getValue( c );
                  break;
               }

            if( c_diag > 0 ) {
               // shift the subdiagonal elements
               for( IndexType c = c_diag; c > 0; c-- )
                  row.setElement( c, row.getColumnIndex( c - 1 ), row.getValue( c - 1 ) );

               // write the diagonal entry to the first position
               row.setElement( 0, j_diag, v_diag );
            }
         } );
   }

protected:
   hypre_CSRMatrix* m = nullptr;
   bool owns_handle = true;
};

}  // namespace Matrices
}  // namespace noa::TNL

#endif  // HAVE_HYPRE
