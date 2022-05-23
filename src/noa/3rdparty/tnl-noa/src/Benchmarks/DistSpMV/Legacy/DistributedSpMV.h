// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/DistributedVectorView.h>

// buffers
#include <vector>
#include <utility>  // std::pair
#include <limits>   // std::numeric_limits
#include <TNL/Allocators/Host.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Matrices/ThreePartVector.h>

// operations
#include <type_traits>  // std::add_const
#include <TNL/Atomic.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Pointers/DevicePointer.h>

namespace TNL {
namespace Matrices {
namespace Legacy {

template< typename Matrix >
class DistributedSpMV
{
public:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using LocalRangeType = Containers::Subrange< typename Matrix::IndexType >;

   // - communication pattern: vector components whose indices are in the range
   //   [start_ij, end_ij) are copied from the j-th process to the i-th process
   //   (an empty range with start_ij == end_ij indicates that there is no
   //   communication between the i-th and j-th processes)
   // - communication pattern matrices - we need to assemble two nproc x nproc
   //   matrices commPatternStarts and commPatternEnds holding the values
   //   start_ij and end_ij respectively
   // - assembly of the i-th row involves traversal of the local matrix stored
   //   in the i-th process
   // - assembly of the full matrix needs all-to-all communication
   void updateCommunicationPattern( const MatrixType& localMatrix, const LocalRangeType& localRowRange, MPI_Comm communicator )
   {
      const int rank = MPI::GetRank( communicator );
      const int nproc = MPI::GetSize( communicator );
      commPatternStarts.setDimensions( nproc, nproc );
      commPatternEnds.setDimensions( nproc, nproc );

      // exchange global offsets (i.e. beginnings of the local ranges) so that each rank can determine the owner of each index
      Containers::Array< IndexType, DeviceType, int > globalOffsets( nproc );
      {
         Containers::Array< IndexType, Devices::Host, int > sendbuf( nproc );
         sendbuf.setValue( localRowRange.getBegin() );
         MPI::Alltoall( sendbuf.getData(), 1,
                        globalOffsets.getData(), 1,
                        communicator );
      }
      const auto globalOffsetsView = globalOffsets.getConstView();
      auto getOwner = [=] __cuda_callable__ ( IndexType global_idx ) -> int
      {
         const int nproc = globalOffsetsView.getSize();
         for( int i = 0; i < nproc - 1; i++ )
            if( globalOffsetsView[ i ] <= global_idx && global_idx < globalOffsetsView[ i + 1 ] )
               return i;
         return nproc - 1;
      };

      // pass the localMatrix to the device
      const Pointers::DevicePointer< const MatrixType > localMatrixPointer( localMatrix );

      // buffer for the local row of the commPattern matrix
      using AtomicIndex = Atomic< IndexType, DeviceType >;
      Containers::Array< AtomicIndex, DeviceType > span_starts( nproc ), span_ends( nproc );
      span_starts.setValue( std::numeric_limits<IndexType>::max() );
      span_ends.setValue( 0 );

      // optimization for banded matrices
      using AtomicIndex = Atomic< IndexType, DeviceType >;
      Containers::Array< AtomicIndex, DeviceType > local_span( 2 );
      local_span.setElement( 0, 0 );  // span start
      local_span.setElement( 1, localMatrix.getRows() );  // span end

      auto kernel = [=] __cuda_callable__ ( IndexType i, const MatrixType* localMatrix,
                                            AtomicIndex* span_starts, AtomicIndex* span_ends, AtomicIndex* local_span )
      {
         const IndexType columns = localMatrix->getColumns();
         const auto row = localMatrix->getRow( i );
         bool comm_left = false;
         bool comm_right = false;
         for( IndexType c = 0; c < row.getSize(); c++ ) {
            const IndexType j = row.getColumnIndex( c );
            if( j == localMatrix->getPaddingIndex() )
               continue;
            if( j < columns ) {
               const int owner = getOwner( j );
               // atomic assignment
               span_starts[ owner ].fetch_min( j );
               span_ends[ owner ].fetch_max( j + 1 );
               // update comm_left/right
               if( owner < rank )
                  comm_left = true;
               if( owner > rank )
                  comm_right = true;
            }
         }
         // update local span
         if( comm_left )
            local_span[0].fetch_max( i + 1 );
         if( comm_right )
            local_span[1].fetch_min( i );
      };

      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, localMatrix.getRows(),
                                                   kernel,
                                                   &localMatrixPointer.template getData< DeviceType >(),
                                                   span_starts.getData(),
                                                   span_ends.getData(),
                                                   local_span.getData()
                                                );

      // set the local-only span (optimization for banded matrices)
      localOnlySpan.first = local_span.getElement( 0 );
      localOnlySpan.second = local_span.getElement( 1 );

      // copy the buffer into all rows of the preCommPattern* matrices
      // (in-place copy does not work with some OpenMPI configurations)
      Matrices::DenseMatrix< IndexType, Devices::Host, int > preCommPatternStarts, preCommPatternEnds;
      preCommPatternStarts.setLike( commPatternStarts );
      preCommPatternEnds.setLike( commPatternEnds );
      for( int j = 0; j < nproc; j++ )
      for( int i = 0; i < nproc; i++ ) {
         preCommPatternStarts.setElement( j, i, span_starts.getElement( i ) );
         preCommPatternEnds.setElement( j, i, span_ends.getElement( i ) );
      }

      // assemble the commPattern* matrices
      MPI::Alltoall( &preCommPatternStarts(0, 0), nproc,
                     &commPatternStarts(0, 0), nproc,
                     communicator );
      MPI::Alltoall( &preCommPatternEnds(0, 0), nproc,
                     &commPatternEnds(0, 0), nproc,
                     communicator );
   }

   template< typename InVector,
             typename OutVector >
   void vectorProduct( OutVector& outVector,
                       const MatrixType& localMatrix,
                       const LocalRangeType& localRowRange,
                       const InVector& inVector,
                       MPI_Comm communicator )
   {
      const int rank = MPI::GetRank( communicator );
      const int nproc = MPI::GetSize( communicator );

      // handle trivial case
      if( nproc == 1 ) {
         const auto inVectorView = inVector.getConstLocalView();
         auto outVectorView = outVector.getLocalView();
         localMatrix.vectorProduct( inVectorView, outVectorView );
         return;
      }

      // update communication pattern
      if( commPatternStarts.getRows() != nproc || commPatternEnds.getRows() != nproc )
         updateCommunicationPattern( localMatrix, localRowRange, communicator );

      // prepare buffers
      globalBuffer.init( localRowRange.getBegin(),
                         inVector.getConstLocalView(),
                         localMatrix.getColumns() - localRowRange.getBegin() - inVector.getConstLocalView().getSize() );

      TNL_ASSERT_EQ( outVector.getLocalView().getSize(), localMatrix.getRows(), "the output vector size does not match the number of matrix rows" );
      TNL_ASSERT_EQ( globalBuffer.getSize(), localMatrix.getColumns(), "the global buffer size does not match the number of matrix columns" );

      // buffer for asynchronous communication requests
      std::vector< MPI_Request > commRequests;

      // send our data to all processes that need it
      for( int i = 0; i < commPatternStarts.getRows(); i++ ) {
         if( i == rank )
             continue;
         if( commPatternStarts( i, rank ) < commPatternEnds( i, rank ) )
            commRequests.push_back( MPI::Isend(
                     inVector.getConstLocalView().getData() + commPatternStarts( i, rank ) - localRowRange.getBegin(),
                     commPatternEnds( i, rank ) - commPatternStarts( i, rank ),
                     i, 0, communicator ) );
      }

      // receive data that we need
      for( int j = 0; j < commPatternStarts.getRows(); j++ ) {
         if( j == rank )
             continue;
         if( commPatternStarts( rank, j ) < commPatternEnds( rank, j ) )
            commRequests.push_back( MPI::Irecv(
                     globalBuffer.getPointer( commPatternStarts( rank, j ) ),
                     commPatternEnds( rank, j ) - commPatternStarts( rank, j ),
                     j, 0, communicator ) );
      }

      // general variant
      if( localOnlySpan.first >= localOnlySpan.second ) {
         // wait for all communications to finish
         MPI::Waitall( commRequests.data(), commRequests.size() );

         // perform matrix-vector multiplication
         auto outVectorView = outVector.getLocalView();
         localMatrix.vectorProduct( globalBuffer, outVectorView );
      }
      // optimization for banded matrices
      else {
         auto outVectorView = outVector.getLocalView();

         // matrix-vector multiplication using local-only rows
         localMatrix.vectorProduct( inVector, outVectorView, 1.0, 0.0, localOnlySpan.first, localOnlySpan.second );

         // wait for all communications to finish
         MPI::Waitall( commRequests.data(), commRequests.size() );

         // finish the multiplication by adding the non-local entries
         localMatrix.vectorProduct( globalBuffer, outVectorView, 1.0, 0.0, 0, localOnlySpan.first );
         localMatrix.vectorProduct( globalBuffer, outVectorView, 1.0, 0.0, localOnlySpan.second, localMatrix.getRows() );
      }
   }

   void reset()
   {
      commPatternStarts.reset();
      commPatternEnds.reset();
      localOnlySpan.first = localOnlySpan.second = 0;
      globalBuffer.reset();
   }

protected:
   // communication pattern
   Matrices::DenseMatrix< IndexType, Devices::Host, int > commPatternStarts, commPatternEnds;

   // span of rows with only block-diagonal entries
   std::pair< IndexType, IndexType > localOnlySpan;

   // global buffer for non-local elements of the vector
   __DistributedSpMV_impl::ThreePartVector< RealType, DeviceType, IndexType > globalBuffer;
};

} // namespace Legacy
} // namespace Matrices
} // namespace TNL
