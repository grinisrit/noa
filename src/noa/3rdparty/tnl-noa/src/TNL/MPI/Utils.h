// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>  // std::getenv

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CheckDevice.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Debugging/OutputRedirection.h>
#include <noa/3rdparty/tnl-noa/src/TNL/TypeTraits.h>

#include "Wrappers.h"
#include "Comm.h"

namespace noa::TNL {
namespace MPI {

inline bool
isInitialized()
{
   return Initialized() && ! Finalized();
}

inline void
setupRedirection( const std::string& outputDirectory )
{
#ifdef HAVE_MPI
   if( GetSize() > 1 && GetRank() != 0 ) {
      const std::string stdoutFile = outputDirectory + "/stdout_" + std::to_string( GetRank() ) + ".txt";
      const std::string stderrFile = outputDirectory + "/stderr_" + std::to_string( GetRank() ) + ".txt";
      std::cout << GetRank() << ": Redirecting stdout and stderr to files " << stdoutFile << " and " << stderrFile << std::endl;
      Debugging::redirect_stdout_stderr( stdoutFile, stderrFile );
   }
#endif
}

// restore redirection (usually not necessary, it uses RAII internally...)
inline void
restoreRedirection()
{
   if( GetSize() > 1 && GetRank() != 0 ) {
      Debugging::redirect_stdout_stderr( "", "", true );
   }
}

inline void
selectGPU()
{
#ifdef __CUDACC__
   int gpuCount;
   cudaGetDeviceCount( &gpuCount );

   // avoid division by zero
   if( gpuCount == 0 ) {
      std::cout << "Rank " << GetRank() << " detected 0 GPUs." << std::endl;
      return;
   }

   const int local_rank = getRankOnNode();
   const int gpuNumber = local_rank % gpuCount;

   // write debug output before calling cudaSetDevice
   const char* cuda_visible_devices = std::getenv( "CUDA_VISIBLE_DEVICES" );
   if( ! cuda_visible_devices )
      cuda_visible_devices = "";
   std::cout << "Rank " << GetRank() << ": rank on node is " << local_rank << ", using GPU id " << gpuNumber << " of "
             << gpuCount << ", CUDA_VISIBLE_DEVICES=" << cuda_visible_devices << std::endl;

   cudaSetDevice( gpuNumber );
   TNL_CHECK_CUDA_DEVICE;
#endif
}

/**
 * \brief Applies the given reduction operation to the values among all ranks
 * in the given communicator.
 *
 * This is a collective operation which uses \ref Allreduce internally. It
 * provides nicer semantics than the wrapper function: the input value is passed
 * by value (heh) rather then by pointer, and the result is returned rather than
 * written to the output pointer.
 *
 * \param value Value of the current rank to be reduced.
 * \param op The reduction operation to be applied.
 * \param communicator The communicator comprising ranks that participate in the
 *                     collective operation.
 * \return The reduced value (it is ensured that all ranks receive the same
 *         value).
 */
template< typename T >
T
reduce( T value, const MPI_Op& op, MPI_Comm communicator = MPI_COMM_WORLD )
{
   // call the in-place variant of Allreduce
   Allreduce( &value, 1, op, communicator );
   // return the reduced value
   return value;
}

/**
 * \brief Send data from an array (or array view or a string) to a different
 * rank.
 *
 * The destination rank must call \ref recv with a corresponding data structure
 * to receive the data.
 */
template< typename Array >
void
send( const Array& array, int dest, int tag, MPI_Comm communicator = MPI_COMM_WORLD )
{
   const auto size = array.getSize();
   MPI::Send( &size, 1, dest, tag, communicator );
   MPI::Send( array.getData(), array.getSize(), dest, tag, communicator );
}

/**
 * \brief Receive data into an array (or a string) from a different rank.
 *
 * The data must be coming from a rank that called \ref send on a corresponding
 * data structure.
 */
template< typename Array >
std::enable_if_t< ! IsViewType< Array >::value >
recv( Array& array, int src, int tag, MPI_Comm communicator = MPI_COMM_WORLD )
{
   using Index = decltype( array.getSize() );
   Index size;
   MPI::Recv( &size, 1, src, tag, communicator );
   array.setSize( size );
   MPI::Recv( array.getData(), size, src, tag, communicator );
}

/**
 * \brief Receive data into an array view from a different rank.
 *
 * The data must be coming from a rank that called \ref send on a corresponding
 * data structure.
 *
 * Since views are not resizable, the size of the incoming data must match the
 * array view size, otherwise \ref std::runtime_error is thrown.
 */
template< typename Array >
std::enable_if_t< IsViewType< Array >::value >
recv( Array& view, int src, int tag, MPI_Comm communicator = MPI_COMM_WORLD )
{
   using Index = decltype( view.getSize() );
   Index size;
   MPI::Recv( &size, 1, src, tag, communicator );
   if( size != view.getSize() )
      throw std::runtime_error( "MPI::recv error: The received size (" + std::to_string( size )
                                + ") does not match "
                                  "the array view size ("
                                + std::to_string( view.getSize() ) + ")" );
   MPI::Recv( view.getData(), size, src, tag, communicator );
}

/**
 * \brief Send and receive data from/into an array (or a string) to/from a
 * different rank.
 */
template< typename SendArray, typename RecvArray >
std::enable_if_t< ! IsViewType< RecvArray >::value >
sendrecv( const SendArray& sendArray,
          int dest,
          int sendTag,
          RecvArray& recvArray,
          int src,
          int recvTag,
          MPI_Comm communicator = MPI_COMM_WORLD )
{
   using SendIndex = decltype( sendArray.getSize() );
   using RecvIndex = decltype( recvArray.getSize() );

   const SendIndex sendSize = sendArray.getSize();
   RecvIndex recvSize;
   MPI::Sendrecv( &sendSize, 1, dest, sendTag, &recvSize, 1, src, recvTag, communicator );
   recvArray.setSize( recvSize );
   MPI::Sendrecv( sendArray.getData(),
                  sendArray.getSize(),
                  dest,
                  sendTag,
                  recvArray.getData(),
                  recvArray.getSize(),
                  src,
                  recvTag,
                  communicator );
}

/**
 * \brief Send and receive data from an array (or an array view) into an array
 * view to/from a different rank.
 *
 * Since views are not resizable, the size of the incoming data must match the
 * array view size, otherwise \ref std::runtime_error is thrown.
 */
template< typename SendArray, typename RecvArray >
std::enable_if_t< IsViewType< RecvArray >::value >
sendrecv( const SendArray& sendArray,
          int dest,
          int sendTag,
          RecvArray& recvArray,
          int src,
          int recvTag,
          MPI_Comm communicator = MPI_COMM_WORLD )
{
   using SendIndex = decltype( sendArray.getSize() );
   using RecvIndex = decltype( recvArray.getSize() );

   const SendIndex sendSize = sendArray.getSize();
   RecvIndex recvSize;
   MPI::Sendrecv( &sendSize, 1, dest, sendTag, &recvSize, 1, src, recvTag, communicator );
   if( recvSize != recvArray.getSize() )
      throw std::runtime_error( "MPI::sendrecv error: The received size (" + std::to_string( recvSize )
                                + ") does not match "
                                  "the array view size ("
                                + std::to_string( recvArray.getSize() ) + ")" );
   MPI::Sendrecv( sendArray.getData(),
                  sendArray.getSize(),
                  dest,
                  sendTag,
                  recvArray.getData(),
                  recvArray.getSize(),
                  src,
                  recvTag,
                  communicator );
}

/**
 * \brief Broadcast a scalar value.
 */
template< typename T >
std::enable_if_t< IsScalarType< T >::value, T >
bcast( T value, int root, MPI_Comm communicator = MPI_COMM_WORLD )
{
   MPI::Bcast( &value, 1, root, communicator );
   return value;
}

/**
 * \brief Broadcast an array (or a string).
 */
template< typename Array >
std::enable_if_t< ! IsScalarType< Array >::value && ! IsViewType< Array >::value >
bcast( Array& array, int root, MPI_Comm communicator = MPI_COMM_WORLD )
{
   auto size = array.getSize();
   MPI::Bcast( &size, 1, root, communicator );
   array.setSize( size );
   MPI::Bcast( array.getData(), size, root, communicator );
}

}  // namespace MPI
}  // namespace noa::TNL
