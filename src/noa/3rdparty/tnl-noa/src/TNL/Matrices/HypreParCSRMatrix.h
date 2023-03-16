// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include <memory>  // std::unique_ptr

   #include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/MPI/Wrappers.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Containers/Subrange.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Containers/HypreVector.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Matrices/HypreCSRMatrix.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Matrices/HypreGenerateDiagAndOffd.h>

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
class HypreParCSRMatrix
{
public:
   using RealType = HYPRE_Real;
   using ValueType = RealType;
   using DeviceType = HYPRE_Device;
   using IndexType = HYPRE_Int;
   using LocalRangeType = Containers::Subrange< IndexType >;

   using MatrixType = HypreCSRMatrix;

   using ValuesViewType = Containers::VectorView< RealType, DeviceType, IndexType >;
   using ConstValuesViewType = typename ValuesViewType::ConstViewType;
   using ColumnIndexesVectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
   using ColumnIndexesViewType = typename ColumnIndexesVectorType::ViewType;
   using ConstColumnIndexesViewType = typename ColumnIndexesVectorType::ConstViewType;
   using SegmentsViewType = Algorithms::Segments::CSRViewDefault< DeviceType, IndexType >;
   using ConstSegmentsViewType = Algorithms::Segments::CSRViewDefault< DeviceType, std::add_const_t< IndexType > >;

   HypreParCSRMatrix() = default;

   // TODO: behavior should depend on "owns_data" (shallow vs deep copy)
   HypreParCSRMatrix( const HypreParCSRMatrix& other ) = delete;

   HypreParCSRMatrix( HypreParCSRMatrix&& other ) noexcept : m( other.m )
   {
      other.m = nullptr;
   }

   // TODO should do a deep copy
   HypreParCSRMatrix&
   operator=( const HypreParCSRMatrix& other ) = delete;

   HypreParCSRMatrix&
   operator=( HypreParCSRMatrix&& other ) noexcept
   {
      m = other.m;
      other.m = nullptr;
      return *this;
   }

   /**
    * \brief Convert Hypre's format to \e HypreParCSRMatrix
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the matrix should take ownership of
    * the handle, i.e. whether to call \e hypre_CSRMatrixDestroy when it does
    * not need it anymore.
    */
   explicit HypreParCSRMatrix( hypre_ParCSRMatrix* handle, bool take_ownership = true )
   {
      bind( handle, take_ownership );
   }

   /**
    * \brief Wrap a global \e hypre_CSRMatrix into \e HypreParCSRMatrix
    *
    * Each rank will get its own independent \e HypreParCSRMatrix with the
    * `MPI_COMM_SELF` communicator. The data is not copied, but
    * \e HypreParCSRMatrix keeps a non-owning reference.
    */
   static HypreParCSRMatrix
   wrapCSRMatrix( hypre_CSRMatrix* matrix )
   {
      TNL_ASSERT_TRUE( matrix, "invalid input" );
      // check the memory location of the input
      TNL_ASSERT_EQ( hypre_CSRMatrixMemoryLocation( matrix ),
                     getHypreMemoryLocation(),
                     "memory location of the input Hypre matrix does not match" );

      const IndexType global_num_rows = hypre_CSRMatrixNumRows( matrix );
      const IndexType global_num_cols = hypre_CSRMatrixNumCols( matrix );
      IndexType row_starts[ 2 ];
      row_starts[ 0 ] = 0;
      row_starts[ 1 ] = global_num_rows;
      IndexType col_starts[ 2 ];
      col_starts[ 0 ] = 0;
      col_starts[ 1 ] = global_num_cols;

      // create the handle
      hypre_ParCSRMatrix* A =
         hypre_ParCSRMatrixCreate( MPI_COMM_SELF, global_num_rows, global_num_cols, row_starts, col_starts, 0, 0, 0 );

      // bind the diag block
      hypre_CSRMatrixDestroy( hypre_ParCSRMatrixDiag( A ) );
      hypre_ParCSRMatrixDiag( A ) = matrix;

      // set the memory location of the (empty) offd matrix
      hypre_CSRMatrixMemoryLocation( hypre_ParCSRMatrixOffd( A ) ) = getHypreMemoryLocation();

      // make sure that the first entry in each row of a square diagonal block is the diagonal element
      HypreCSRMatrix diag_view( hypre_ParCSRMatrixDiag( A ), false );
      diag_view.reorderDiagonalEntries();

      // set additional attributes
      hypre_ParCSRMatrixNumNonzeros( A ) = hypre_CSRMatrixNumNonzeros( matrix );

      // initialize auxiliary substructures
      hypre_CSRMatrixSetRownnz( hypre_ParCSRMatrixDiag( A ) );
      hypre_MatvecCommPkgCreate( A );

      // check the memory location of the result
      TNL_ASSERT_EQ( hypre_ParCSRMatrixMemoryLocation( A ),
                     getHypreMemoryLocation(),
                     "memory location of the output Hypre matrix does not match" );

      // set the diag owner flag
      HypreParCSRMatrix result( A );
      result.owns_diag = false;
      return result;
   }

   /**
    * \brief Constructs a \e ParCSRMatrix distributed across the processors in
    * \e communicator from a \e CSRMatrix on rank 0.
    *
    * \note This function can be used only for matrices allocated on the host.
    *
    * \param communicator MPI communicator to associate with the matrix.
    * \param global_row_starts Array of `nproc + 1` elements, where `nproc` is
    *                          the number of ranks in the \e communicator.
    * \param global_col_starts Array of `nproc + 1` elements, where `nproc` is
    *                          the number of ranks in the \e communicator.
    * \param matrix Matrix allocated on the master rank to be distributed.
    */
   static HypreParCSRMatrix
   fromMasterRank( MPI_Comm communicator, IndexType* global_row_starts, IndexType* global_col_starts, hypre_CSRMatrix* matrix )
   {
      TNL_ASSERT_TRUE( global_row_starts, "invalid input" );
      TNL_ASSERT_TRUE( global_col_starts, "invalid input" );
      TNL_ASSERT_TRUE( matrix, "invalid input" );
      // check the memory location of the input
      TNL_ASSERT_EQ( hypre_CSRMatrixMemoryLocation( matrix ),
                     getHypreMemoryLocation(),
                     "memory location of the input Hypre matrix does not match" );

      // NOTE: this call creates a matrix on host even when device support is
      // enabled in Hypre
      hypre_ParCSRMatrix* A = hypre_CSRMatrixToParCSRMatrix( communicator, matrix, global_row_starts, global_col_starts );

      // make sure that the first entry in each row of a square diagonal block is the diagonal element
      HypreCSRMatrix diag_view( hypre_ParCSRMatrixDiag( A ), false );
      diag_view.reorderDiagonalEntries();

      // initialize auxiliary substructures
      hypre_CSRMatrixSetRownnz( hypre_ParCSRMatrixDiag( A ) );
      hypre_ParCSRMatrixSetNumNonzeros( A );
      hypre_MatvecCommPkgCreate( A );

      // check the memory location of the result
      // FIXME: wtf, we get HYPRE_MEMORY_DEVICE even when Hypre is compiled without CUDA
      // TNL_ASSERT_EQ( hypre_ParCSRMatrixMemoryLocation( A ),
      //                getHypreMemoryLocation(),
      //                "memory location of the output Hypre matrix does not match" );

      return HypreParCSRMatrix( A );
   }

   /**
    * \brief Constructs a \e ParCSRMatrix distributed across the processors in
    * \e communicator from a \e CSRMatrix on rank 0.
    *
    * \note This function can be used only for matrices allocated on the host.
    *
    * \param matrix Matrix allocated on the master rank to be distributed.
    * \param x The values of the vector are unused, but its distribution is
    *          used for the distribution of the matrix **columns**.
    * \param b The values of the vector are unused, but its distribution is
    *          used for the distribution of the matrix **rows**.
    */
   static HypreParCSRMatrix
   fromMasterRank( hypre_CSRMatrix* matrix, hypre_ParVector* x, hypre_ParVector* b )
   {
      const MPI_Comm communicator = hypre_ParVectorComm( x );
      if( communicator == MPI_COMM_NULL )
         return {};

      // construct global partitioning info on rank 0
      // (the result on other ranks is not used)
      const int nproc = MPI::GetSize( communicator );
      std::unique_ptr< IndexType[] > sendbuf{ new IndexType[ nproc ] };

      std::unique_ptr< IndexType[] > global_row_starts{ new IndexType[ nproc + 1 ] };
      for( int i = 0; i < nproc; i++ )
         sendbuf[ i ] = hypre_ParVectorFirstIndex( b );
      MPI::Alltoall( sendbuf.get(), 1, global_row_starts.get(), 1, communicator );
      global_row_starts[ nproc ] = hypre_ParVectorGlobalSize( b );

      // if the matrix is square, global_col_starts should be equal to global_row_starts
      // (otherwise Hypre puts everything into the offd block which would violate the
      // AssumedPartition premise)
      if( hypre_CSRMatrixNumRows( matrix ) == hypre_CSRMatrixNumCols( matrix ) ) {
         return fromMasterRank( communicator, global_row_starts.get(), global_row_starts.get(), matrix );
      }

      // NOTE: this was not tested...
      std::unique_ptr< IndexType[] > global_col_starts{ new IndexType[ nproc + 1 ] };
      for( int i = 0; i < nproc; i++ )
         sendbuf[ i ] = hypre_ParVectorFirstIndex( x );
      MPI::Alltoall( sendbuf.get(), 1, global_col_starts.get(), 1, communicator );
      global_col_starts[ nproc ] = hypre_ParVectorGlobalSize( x );

      return fromMasterRank( communicator, global_row_starts.get(), global_col_starts.get(), matrix );
   }

   /**
    * \brief Constructs a \e ParCSRMatrix from local blocks distributed across
    * the processors in \e communicator.
    *
    * \param communicator MPI communicator to associate with the matrix.
    * \param global_num_rows Global number of rows of the distributed matrix.
    * \param global_num_cols Global number of columns of the distributed matrix.
    * \param local_row_range The range `[begin, end)` of rows owned by the
    *                        calling rank.
    * \param local_col_range The range `[begin, end)` of columns owned by the
    *                        calling rank. For square matrices it should be
    *                        equal to \e local_row_range.
    * \param local_A The local matrix block owned by the calling rank. The
    *        number of rows must match the size of \e local_row_range and the
    *        column indices must span the whole `[0, global_cols)` range.
    */
   static HypreParCSRMatrix
   fromLocalBlocks( MPI_Comm communicator,
                    IndexType global_num_rows,
                    IndexType global_num_cols,
                    LocalRangeType local_row_range,
                    LocalRangeType local_col_range,
                    hypre_CSRMatrix* local_A )
   {
      TNL_ASSERT_TRUE( local_A, "invalid input" );
      // check the memory location of the input
      TNL_ASSERT_EQ( hypre_CSRMatrixMemoryLocation( local_A ),
                     getHypreMemoryLocation(),
                     "memory location of the input Hypre matrix does not match" );

      IndexType row_starts[ 2 ];
      row_starts[ 0 ] = local_row_range.getBegin();
      row_starts[ 1 ] = local_row_range.getEnd();
      IndexType col_starts[ 2 ];
      col_starts[ 0 ] = local_col_range.getBegin();
      col_starts[ 1 ] = local_col_range.getEnd();

      hypre_ParCSRMatrix* A =
         hypre_ParCSRMatrixCreate( communicator, global_num_rows, global_num_cols, row_starts, col_starts, 0, 0, 0 );

      // set the memory location of the (empty) diag and offd matrices
      // (hypre_ParCSRMatrixMemoryLocation is read-only)
      hypre_CSRMatrixMemoryLocation( hypre_ParCSRMatrixDiag( A ) ) = getHypreMemoryLocation();
      hypre_CSRMatrixMemoryLocation( hypre_ParCSRMatrixOffd( A ) ) = getHypreMemoryLocation();

      const IndexType first_col_diag = hypre_ParCSRMatrixFirstColDiag( A );
      const IndexType last_col_diag = hypre_ParCSRMatrixLastColDiag( A );

      // split local_A into the diagonal and off-diagonal blocks
      detail::GenerateDiagAndOffd( local_A, A, first_col_diag, last_col_diag );

      // make sure that the first entry in each row of a square diagonal block is the diagonal element
      HypreCSRMatrix diag_view( hypre_ParCSRMatrixDiag( A ), false );
      diag_view.reorderDiagonalEntries();

      // initialize auxiliary substructures
      hypre_CSRMatrixSetRownnz( hypre_ParCSRMatrixDiag( A ) );
      hypre_ParCSRMatrixSetNumNonzeros( A );
      hypre_MatvecCommPkgCreate( A );

      // check the memory location of the result
      TNL_ASSERT_EQ( hypre_ParCSRMatrixMemoryLocation( A ),
                     getHypreMemoryLocation(),
                     "memory location of the output Hypre matrix does not match" );

      return HypreParCSRMatrix( A );
   }

   operator const hypre_ParCSRMatrix*() const noexcept
   {
      return m;
   }

   operator hypre_ParCSRMatrix*() noexcept
   {
      return m;
   }

   // HYPRE_ParCSRMatrix is "equivalent" to pointer to hypre_ParCSRMatrix, but requires
   // ugly C-style cast on the pointer (which is done even in Hypre itself)
   // https://github.com/hypre-space/hypre/blob/master/src/parcsr_mv/HYPRE_parcsr_matrix.c
   operator HYPRE_ParCSRMatrix() const noexcept
   {
      return (HYPRE_ParCSRMatrix) m;
   }

   ~HypreParCSRMatrix()
   {
      reset();
   }

   LocalRangeType
   getLocalRowRange() const
   {
      if( m == nullptr )
         return {};
      return { hypre_ParCSRMatrixRowStarts( m )[ 0 ], hypre_ParCSRMatrixRowStarts( m )[ 1 ] };
   }

   LocalRangeType
   getLocalColumnRange() const
   {
      if( m == nullptr )
         return {};
      return { hypre_ParCSRMatrixColStarts( m )[ 0 ], hypre_ParCSRMatrixColStarts( m )[ 1 ] };
   }

   IndexType
   getRows() const
   {
      if( m == nullptr )
         return 0;
      return hypre_ParCSRMatrixGlobalNumRows( m );
   }

   IndexType
   getColumns() const
   {
      if( m == nullptr )
         return 0;
      return hypre_ParCSRMatrixGlobalNumCols( m );
   }

   IndexType
   getNonzeroElementsCount() const
   {
      if( m == nullptr )
         return 0;
      return hypre_ParCSRMatrixNumNonzeros( m );
   }

   MPI_Comm
   getCommunicator() const
   {
      if( m == nullptr )
         return MPI_COMM_NULL;
      return hypre_ParCSRMatrixComm( m );
   }

   MatrixType
   getDiagonalBlock()
   {
      if( m == nullptr )
         return {};
      return HypreCSRMatrix( hypre_ParCSRMatrixDiag( m ), false );
   }

   MatrixType
   getOffdiagonalBlock()
   {
      if( m == nullptr )
         return {};
      return HypreCSRMatrix( hypre_ParCSRMatrixOffd( m ), false );
   }

   Containers::VectorView< HYPRE_Int, HYPRE_Device, HYPRE_Int >
   getOffdiagonalColumnsMapping()
   {
      if( m == nullptr )
         return {};
      HYPRE_Int* col_map_offd = hypre_ParCSRMatrixColMapOffd( m );
      const HYPRE_Int num_cols = hypre_CSRMatrixNumCols( hypre_ParCSRMatrixOffd( m ) );
      return { col_map_offd, num_cols };
   }

   //! \brief Constructs a local matrix by merging the diagonal and off-diagonal blocks.
   HypreCSRMatrix
   getMergedLocalMatrix() const
   {
      hypre_CSRMatrix* local_A = hypre_MergeDiagAndOffd( m );
      return HypreCSRMatrix( local_A, true );
   }

   /**
    * \brief Convert Hypre's format to \e HypreParCSRMatrix
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the matrix should take ownership of
    * the handle, i.e. whether to call \e hypre_CSRMatrixDestroy when it does
    * not need it anymore.
    */
   void
   bind( hypre_ParCSRMatrix* handle, bool take_ownership = true )
   {
      // drop/deallocate the current data
      reset();

      // take ownership of the handle
      m = handle;
      owns_handle = take_ownership;
   }

   /**
    * \brief Binds \e ParCSRMatrix to local \e diag and \e offd blocks.
    *
    * \param communicator MPI communicator to associate with the matrix.
    * \param global_num_rows Global number of rows of the distributed matrix.
    * \param global_num_cols Global number of columns of the distributed matrix.
    * \param local_row_range The range `[begin, end)` of rows owned by the
    *                        calling rank.
    * \param local_col_range The range `[begin, end)` of columns owned by the
    *                        calling rank. For square matrices it should be
    *                        equal to \e local_row_range.
    * \param diag The local diagonal matrix block owned by the calling rank.
    * \param offd The local off-diagonal matrix block owned by the calling rank.
    * \param col_map_offd Mapping of local-to-global indices for the columns in
    *                     the off-diagonal block. It must be *always* a host
    *                     pointer.
    */
   void
   bind( MPI_Comm communicator,
         IndexType global_num_rows,
         IndexType global_num_cols,
         LocalRangeType local_row_range,
         LocalRangeType local_col_range,
         hypre_CSRMatrix* diag,
         hypre_CSRMatrix* offd,
         IndexType* col_map_offd )
   {
      TNL_ASSERT_TRUE( diag, "invalid input: diag" );
      TNL_ASSERT_TRUE( offd, "invalid input: offd" );
      TNL_ASSERT_TRUE( col_map_offd, "invalid input: col_map_offd" );
      // check the memory location of the input
      TNL_ASSERT_EQ( hypre_CSRMatrixMemoryLocation( diag ),
                     getHypreMemoryLocation(),
                     "memory location of the input Hypre matrix 'diag' does not match" );
      TNL_ASSERT_EQ( hypre_CSRMatrixMemoryLocation( offd ),
                     getHypreMemoryLocation(),
                     "memory location of the input Hypre matrix 'offd' does not match" );

      // drop/deallocate the current data
      reset();

      IndexType row_starts[ 2 ];
      row_starts[ 0 ] = local_row_range.getBegin();
      row_starts[ 1 ] = local_row_range.getEnd();
      IndexType col_starts[ 2 ];
      col_starts[ 0 ] = local_col_range.getBegin();
      col_starts[ 1 ] = local_col_range.getEnd();

      // create new matrix
      m = hypre_ParCSRMatrixCreate( communicator, global_num_rows, global_num_cols, row_starts, col_starts, 0, 0, 0 );

      // bind the local matrices
      hypre_CSRMatrixDestroy( hypre_ParCSRMatrixDiag( m ) );
      hypre_CSRMatrixDestroy( hypre_ParCSRMatrixOffd( m ) );
      hypre_ParCSRMatrixDiag( m ) = diag;
      hypre_ParCSRMatrixOffd( m ) = offd;
      owns_diag = false;
      owns_offd = false;

      // bind the offd col map
      hypre_TFree( hypre_ParCSRMatrixColMapOffd( m ), HYPRE_MEMORY_HOST );
      hypre_ParCSRMatrixColMapOffd( m ) = col_map_offd;
      owns_col_map_offd = false;

      // initialize auxiliary substructures
      hypre_CSRMatrixSetRownnz( hypre_ParCSRMatrixDiag( m ) );
      hypre_ParCSRMatrixSetNumNonzeros( m );
      hypre_MatvecCommPkgCreate( m );

      // check the memory location of the result
      TNL_ASSERT_EQ( hypre_ParCSRMatrixMemoryLocation( m ),
                     getHypreMemoryLocation(),
                     "memory location of the output Hypre matrix does not match" );
   }

   //! \brief Reset the matrix to empty state.
   void
   reset()
   {
      if( owns_handle && m != nullptr ) {
         if( ! owns_diag )
   #if HYPRE_RELEASE_NUMBER >= 22600
            hypre_ParCSRMatrixDiag( m ) = nullptr;
   #else
            // hypreParCSRMatrixDestroy does not work when diag is a nullptr
            hypre_ParCSRMatrixDiag( m ) = hypre_CSRMatrixCreate( 0, 0, 0 );
   #endif
         if( ! owns_offd )
   #if HYPRE_RELEASE_NUMBER >= 22600
            hypre_ParCSRMatrixOffd( m ) = nullptr;
   #else
            // hypreParCSRMatrixDestroy does not work when offd is a nullptr
            hypre_ParCSRMatrixOffd( m ) = hypre_CSRMatrixCreate( 0, 0, 0 );
   #endif
         if( ! owns_col_map_offd )
            hypre_ParCSRMatrixColMapOffd( m ) = nullptr;
         hypre_ParCSRMatrixDestroy( m );
         m = nullptr;
      }
      else
         m = nullptr;
      owns_handle = true;
      owns_diag = true;
      owns_offd = true;
      owns_col_map_offd = true;
   }

protected:
   hypre_ParCSRMatrix* m = nullptr;
   bool owns_handle = true;
   bool owns_diag = true;
   bool owns_offd = true;
   bool owns_col_map_offd = true;
};

}  // namespace Matrices
}  // namespace noa::TNL

#endif  // HAVE_HYPRE
