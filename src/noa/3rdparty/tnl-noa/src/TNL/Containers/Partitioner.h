// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <utility>
#include <vector>

#include "Subrange.h"
#include "ByteArraySynchronizer.h"

#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>

namespace noa::TNL {
namespace Containers {

template< typename Index >
class Partitioner
{
public:
   using SubrangeType = Subrange< Index >;

   static SubrangeType
   splitRange( Index globalSize, const MPI::Comm& communicator )
   {
      if( communicator == MPI_COMM_NULL )
         return { 0, 0 };

      const int rank = communicator.rank();
      const int partitions = communicator.size();

      const Index partSize = globalSize / partitions;
      const int remainder = globalSize % partitions;
      if( rank < remainder ) {
         const Index begin = rank * ( partSize + 1 );
         const Index end = begin + partSize + 1;
         return { begin, end };
      }
      const Index begin = remainder * ( partSize + 1 ) + ( rank - remainder ) * partSize;
      const Index end = begin + partSize;
      return { begin, end };
   }

   // Gets the offset of data for given rank.
   __cuda_callable__
   static Index
   getOffset( Index globalSize, int rank, int partitions )
   {
      const Index partSize = globalSize / partitions;
      const int remainder = globalSize % partitions;
      if( rank < remainder )
         return rank * ( partSize + 1 );
      return remainder * ( partSize + 1 ) + ( rank - remainder ) * partSize;
   }

   // Gets the size of data assigned to given rank.
   __cuda_callable__
   static Index
   getSizeForRank( Index globalSize, int rank, int partitions )
   {
      const Index partSize = globalSize / partitions;
      const int remainder = globalSize % partitions;
      if( rank < remainder )
         return partSize + 1;
      return partSize;
   }

   template< typename Device >
   class ArraySynchronizer : public ByteArraySynchronizer< Device, Index >
   {
      using Base = ByteArraySynchronizer< Device, Index >;

      SubrangeType localRange;
      int overlaps;
      MPI::Comm communicator;

   public:
      using ByteArrayView = typename Base::ByteArrayView;
      using RequestsVector = typename Base::RequestsVector;

      ~ArraySynchronizer()
      {
         // wait for pending async operation, otherwise it would crash
         if( this->async_op.valid() )
            this->async_op.wait();
      }

      ArraySynchronizer() = delete;

      ArraySynchronizer( SubrangeType localRange, int overlaps, MPI::Comm communicator )
      : localRange( localRange ), overlaps( overlaps ), communicator( std::move( communicator ) )
      {}

      void
      synchronizeByteArray( ByteArrayView array, int bytesPerValue ) override
      {
         auto requests = synchronizeByteArrayAsyncWorker( array, bytesPerValue );
         MPI::Waitall( requests.data(), requests.size() );
      }

      RequestsVector
      synchronizeByteArrayAsyncWorker( ByteArrayView array, int bytesPerValue ) override
      {
         TNL_ASSERT_EQ( array.getSize(), bytesPerValue * ( localRange.getSize() + 2 * overlaps ), "unexpected array size" );

         const int rank = communicator.rank();
         const int nproc = communicator.size();
         const int left = ( rank > 0 ) ? rank - 1 : nproc - 1;
         const int right = ( rank < nproc - 1 ) ? rank + 1 : 0;

         // buffer for asynchronous communication requests
         std::vector< MPI_Request > requests;

         // issue all async receive operations
         requests.push_back( MPI::Irecv(
            array.getData() + bytesPerValue * localRange.getSize(), bytesPerValue * overlaps, left, 0, communicator ) );
         requests.push_back( MPI::Irecv( array.getData() + bytesPerValue * ( localRange.getSize() + overlaps ),
                                         bytesPerValue * overlaps,
                                         right,
                                         0,
                                         communicator ) );

         // issue all async send operations
         requests.push_back( MPI::Isend( array.getData(), bytesPerValue * overlaps, left, 0, communicator ) );
         requests.push_back( MPI::Isend( array.getData() + bytesPerValue * ( localRange.getSize() - overlaps ),
                                         bytesPerValue * overlaps,
                                         right,
                                         0,
                                         communicator ) );

         return requests;
      }
   };
};

}  // namespace Containers
}  // namespace noa::TNL
