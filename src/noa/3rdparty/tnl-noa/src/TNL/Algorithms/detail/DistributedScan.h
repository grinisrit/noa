// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <memory>  // std::unique_ptr

#include "Scan.h"

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Array.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Wrappers.h>

namespace noa::TNL {
namespace Algorithms {
namespace detail {

template< ScanType Type, ScanPhaseType PhaseType >
struct DistributedScan
{
   template< typename InputDistributedArray, typename OutputDistributedArray, typename Reduction >
   static void
   perform( const InputDistributedArray& input,
            OutputDistributedArray& output,
            typename InputDistributedArray::IndexType begin,
            typename InputDistributedArray::IndexType end,
            Reduction&& reduction,
            typename OutputDistributedArray::ValueType identity )
   {
      using ValueType = typename OutputDistributedArray::ValueType;
      using DeviceType = typename OutputDistributedArray::DeviceType;

      const auto& communicator = input.getCommunicator();
      if( communicator != MPI_COMM_NULL ) {
         // adjust begin and end for the local range
         const auto localRange = input.getLocalRange();
         begin = min( max( begin, localRange.getBegin() ), localRange.getEnd() ) - localRange.getBegin();
         end = max( min( end, localRange.getEnd() ), localRange.getBegin() ) - localRange.getBegin();

         // perform first phase on the local data
         const auto inputLocalView = input.getConstLocalView();
         auto outputLocalView = output.getLocalView();
         const auto block_results = Scan< DeviceType, Type, PhaseType >::performFirstPhase(
            inputLocalView, outputLocalView, begin, end, begin, reduction, identity );
         const ValueType local_result = block_results.getElement( block_results.getSize() - 1 );

         // exchange local results between ranks
         const int nproc = communicator.size();
         std::unique_ptr< ValueType[] > dataForScatter{ new ValueType[ nproc ] };
         for( int i = 0; i < nproc; i++ )
            dataForScatter[ i ] = local_result;
         Containers::Array< ValueType, Devices::Host > rank_results( nproc );
         // NOTE: exchanging general data types does not work with MPI
         MPI::Alltoall( dataForScatter.get(), 1, rank_results.getData(), 1, communicator );

         // compute the scan of the per-rank results
         Scan< Devices::Host, ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase >::perform(
            rank_results, rank_results, 0, nproc, 0, reduction, identity );

         // perform the second phase, using the per-block and per-rank results
         const int rank = communicator.rank();
         Scan< DeviceType, Type, PhaseType >::performSecondPhase(
            inputLocalView, outputLocalView, block_results, begin, end, begin, reduction, identity, rank_results[ rank ] );
      }
   }
};

}  // namespace detail
}  // namespace Algorithms
}  // namespace noa::TNL
