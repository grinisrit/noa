// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/reduce.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/scan.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/sort.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Containers/Array.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Hypre.h>

namespace noa::TNL {
namespace Matrices {
namespace detail {

/**
 * \brief Splits matrix A into the diagonal and off-diagonal blocks
 *
 * Note that the Hypre internal function \e GenerateDiagAndOffd does not work
 * on GPUs and even on the host it is not scalable (it uses an int array of
 * size equal to the number of columns in A which is the global number of
 * columns in the matrix).
 *
 * Note that this routine is "hybrid", i.e. part of the algorithm is computed
 * on the host even though most work is performed on the GPU.
 */
HYPRE_Int
GenerateDiagAndOffd( hypre_CSRMatrix* A, hypre_ParCSRMatrix* matrix, HYPRE_BigInt first_col_diag, HYPRE_BigInt last_col_diag )
{
   using ComplexView = Containers::ArrayView< HYPRE_Complex, HYPRE_Device, HYPRE_Int >;
   using IntView = Containers::ArrayView< HYPRE_Int, HYPRE_Device, HYPRE_Int >;
   using IntArray = Containers::Array< HYPRE_Int, HYPRE_Device, HYPRE_Int >;
   using IntViewHost = Containers::ArrayView< HYPRE_Int, Devices::Host, HYPRE_Int >;
   using IntArrayHost = Containers::Array< HYPRE_Int, Devices::Host, HYPRE_Int >;

   TNL_ASSERT_EQ(
      hypre_CSRMatrixMemoryLocation( A ), getHypreMemoryLocation(), "memory location of the input matrix does not match" );

   HYPRE_Int num_rows = hypre_CSRMatrixNumRows( A );
   HYPRE_Int num_cols = hypre_CSRMatrixNumCols( A );
   IntView a_i( hypre_CSRMatrixI( A ), num_rows + 1 );
   HYPRE_Int num_nonzeros = a_i.getElement( num_rows ) - a_i.getElement( 0 );
   IntView a_j( ( hypre_CSRMatrixBigJ( A ) != nullptr ) ? hypre_CSRMatrixBigJ( A ) : hypre_CSRMatrixJ( A ), num_nonzeros );
   ComplexView a_data( hypre_CSRMatrixData( A ), num_nonzeros );

   hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag( matrix );
   hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd( matrix );

   const HYPRE_Int num_cols_diag = last_col_diag - first_col_diag + 1;
   if( num_cols - num_cols_diag ) {
      hypre_CSRMatrixInitialize_v2( diag, 0, getHypreMemoryLocation() );
      IntView diag_i( hypre_CSRMatrixI( diag ), num_rows + 1 );

      hypre_CSRMatrixInitialize_v2( offd, 0, getHypreMemoryLocation() );
      IntView offd_i( hypre_CSRMatrixI( offd ), num_rows + 1 );

      const HYPRE_Int first_elmt = a_i.getElement( 0 );

      // count the numbers of diagonal and off-diagonal elements in each row
      Algorithms::ParallelFor< HYPRE_Device >::exec(  //
         HYPRE_Int( 0 ),
         num_rows,
         [ = ] __cuda_callable__( HYPRE_Int i ) mutable
         {
            HYPRE_Int diag_count = 0;
            HYPRE_Int offd_count = 0;
            for( HYPRE_Int j = a_i[ i ] - first_elmt; j < a_i[ i + 1 ] - first_elmt; j++ ) {
               if( a_j[ j ] < first_col_diag || a_j[ j ] > last_col_diag )
                  offd_count++;
               else
                  diag_count++;
            }
            diag_i[ i ] = diag_count;
            offd_i[ i ] = offd_count;
         } );

      // scan the diag_i and offd_i arrays to obtain the CSR format
      Algorithms::inplaceExclusiveScan( diag_i );
      Algorithms::inplaceExclusiveScan( offd_i );

      // allocate values and column indices for the diag and offd matrices
      hypre_CSRMatrixNumNonzeros( diag ) = diag_i.getElement( num_rows );
      hypre_CSRMatrixInitialize( diag );
      HYPRE_Complex* diag_data = hypre_CSRMatrixData( diag );
      HYPRE_Int* diag_j = hypre_CSRMatrixJ( diag );

      hypre_CSRMatrixNumNonzeros( offd ) = offd_i.getElement( num_rows );
      hypre_CSRMatrixInitialize( offd );
      HYPRE_Complex* offd_data = hypre_CSRMatrixData( offd );
      HYPRE_Int* offd_j = hypre_CSRMatrixJ( offd );

      // copy the values and column indices from A to the diag and offd parts
      Algorithms::ParallelFor< HYPRE_Device >::exec(  //
         HYPRE_Int( 0 ),
         num_rows,
         [ = ] __cuda_callable__( HYPRE_Int i ) mutable
         {
            HYPRE_Int jd = diag_i[ i ];
            HYPRE_Int jo = offd_i[ i ];
            for( HYPRE_Int j = a_i[ i ] - first_elmt; j < a_i[ i + 1 ] - first_elmt; j++ ) {
               if( a_j[ j ] < first_col_diag || a_j[ j ] > last_col_diag ) {
                  offd_data[ jo ] = a_data[ j ];
                  // offd column indices are copied here and will be transformed
                  // later
                  offd_j[ jo++ ] = a_j[ j ];
               }
               else {
                  diag_data[ jd ] = a_data[ j ];
                  // diag column indices are transformed here
                  diag_j[ jd++ ] = a_j[ j ] - first_col_diag;
               }
            }
         } );

      // prepare array for the mapping of offd column indices
      IntArray perm( hypre_CSRMatrixNumNonzeros( offd ) );
      perm.forAllElements(
         [] __cuda_callable__( HYPRE_Int idx, HYPRE_Int & value )
         {
            value = idx;
         } );
      IntView perm_view = perm.getView();

      // sorting: `offfd_j[ perm[ j ] ]` will be an increasing sequence
      Algorithms::sort< HYPRE_Device, HYPRE_Int >(
         0,
         perm.getSize(),                                                // range of indices
         [ = ] __cuda_callable__( HYPRE_Int a, HYPRE_Int b ) -> bool {  // comparator lambda function
            return offd_j[ perm_view[ a ] ] < offd_j[ perm_view[ b ] ];
         },
         [ = ] __cuda_callable__( HYPRE_Int a, HYPRE_Int b ) mutable {  // swapper lambda function
            TNL::swap( perm_view[ a ], perm_view[ b ] );
         } );

      // count offd columns
      const HYPRE_Int num_cols_offd = Algorithms::reduce< HYPRE_Device >(
         HYPRE_Int( 0 ),
         perm.getSize(),                                        // range of indices
         [ = ] __cuda_callable__( HYPRE_Int j ) -> HYPRE_Int {  // fetch: values to reduce
            const HYPRE_Int prev = ( j == 0 ) ? -1 : offd_j[ perm_view[ j - 1 ] ];
            const HYPRE_Int value = offd_j[ perm_view[ j ] ];
            if( value != prev )
               return 1;
            else
               return 0;
         },
         TNL::Plus{}  // reduction operation
      );
      hypre_CSRMatrixNumCols( offd ) = num_cols_offd;

      // create the offd columns mapping - must be always on host
      hypre_ParCSRMatrixColMapOffd( matrix ) = hypre_CTAlloc( HYPRE_BigInt, num_cols_offd, HYPRE_MEMORY_HOST );
      IntViewHost col_map_offd( hypre_ParCSRMatrixColMapOffd( matrix ), num_cols_offd );

      // copy some data to the host
      IntView offd_j_view( offd_j, perm.getSize() );
      IntArrayHost offd_j_host;
      offd_j_host = offd_j_view;
      IntArrayHost perm_host;
      perm_host = perm;

      // remap the columns on the host
      HYPRE_Int segmentIdx = 0;
      HYPRE_Int prev = 0;
      for( HYPRE_Int j = 0; j < perm_host.getSize(); j++ ) {
         const HYPRE_Int value = offd_j_host[ perm_host[ j ] ];
         if( j == 0 || value != prev )
            col_map_offd[ segmentIdx++ ] = value;
         offd_j_host[ perm_host[ j ] ] = segmentIdx - 1;
         prev = value;
      }

      // copy the result from the host to the device
      offd_j_view = offd_j_host;
   }
   else {
      hypre_CSRMatrixNumNonzeros( diag ) = num_nonzeros;
      hypre_CSRMatrixInitialize_v2( diag, 0, getHypreMemoryLocation() );

      // copy A into diag
      ComplexView diag_data( hypre_CSRMatrixData( diag ), num_nonzeros );
      IntView diag_i( hypre_CSRMatrixI( diag ), num_rows + 1 );
      IntView diag_j( hypre_CSRMatrixJ( diag ), num_nonzeros );
      diag_data = a_data;
      diag_i = a_i;
      diag_j = a_j;

      // create empty offd
      hypre_CSRMatrixNumNonzeros( offd ) = 0;
      hypre_CSRMatrixNumCols( offd ) = 0;
      hypre_CSRMatrixInitialize_v2( offd, 0, getHypreMemoryLocation() );
      // initialize offd_i with zeros (done in Hypre's GenerateDiagAndOffd)
      IntView offd_i( hypre_CSRMatrixI( offd ), num_rows + 1 );
      offd_i.setValue( 0 );
   }

   return hypre_error_flag;
}

}  // namespace detail
}  // namespace Matrices
}  // namespace noa::TNL

#endif  // HAVE_HYPRE
