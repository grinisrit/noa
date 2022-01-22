// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <utility>  // std::forward

#include "Scan.h"
#include "CudaScanKernel.h"

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Containers/Array.h>
#include <noa/3rdparty/TNL/Containers/StaticArray.h>
#include <noa/3rdparty/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/TNL/Exceptions/CudaSupportMissing.h>

namespace noaTNL {
namespace Algorithms {
namespace detail {

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
typename OutputArray::ValueType
Scan< Devices::Sequential, Type, PhaseType >::
perform( const InputArray& input,
         OutputArray& output,
         typename InputArray::IndexType begin,
         typename InputArray::IndexType end,
         typename OutputArray::IndexType outputBegin,
         Reduction&& reduction,
         typename OutputArray::ValueType identity )
{
   using ValueType = typename OutputArray::ValueType;

   // simple sequential algorithm - not split into phases
   ValueType aux = identity;
   if( Type == ScanType::Inclusive ) {
      for( ; begin < end; begin++, outputBegin++ )
         output[ outputBegin ] = aux = reduction( aux, input[ begin ] );
   }
   else // Exclusive scan
   {
      for( ; begin < end; begin++, outputBegin++ ) {
         const ValueType x = input[ begin ];
         output[ outputBegin ] = aux;
         aux = reduction( aux, x );
      }
   }
   // return the last value of inclusive scan (reduction of the whole input)
   return aux;
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
auto
Scan< Devices::Sequential, Type, PhaseType >::
performFirstPhase( const InputArray& input,
                   OutputArray& output,
                   typename InputArray::IndexType begin,
                   typename InputArray::IndexType end,
                   typename OutputArray::IndexType outputBegin,
                   Reduction&& reduction,
                   typename OutputArray::ValueType identity )
{
   if( end <= begin ) {
      Containers::Array< typename OutputArray::ValueType, Devices::Sequential > block_results( 1 );
      block_results.setValue( identity );
      return block_results;
   }

   switch( PhaseType )
   {
      case ScanPhaseType::WriteInFirstPhase:
      {
         // artificial second phase - pre-scan the block
         Containers::Array< typename OutputArray::ValueType, Devices::Sequential > block_results( 2 );
         block_results[ 0 ] = identity;
         block_results[ 1 ] = perform( input, output, begin, end, outputBegin, reduction, identity );
         return block_results;
      }

      case ScanPhaseType::WriteInSecondPhase:
      {
         // artificial first phase - only reduce the block
         Containers::Array< typename OutputArray::ValueType, Devices::Sequential > block_results( 2 );
         block_results[ 0 ] = identity;
         block_results[ 1 ] = reduce< Devices::Sequential >( begin, end, input, reduction, identity );
         return block_results;
      }
   };
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Sequential, Type, PhaseType >::
performSecondPhase( const InputArray& input,
                    OutputArray& output,
                    const BlockShifts& blockShifts,
                    typename InputArray::IndexType begin,
                    typename InputArray::IndexType end,
                    typename OutputArray::IndexType outputBegin,
                    Reduction&& reduction,
                    typename OutputArray::ValueType identity,
                    typename OutputArray::ValueType shift )
{
   switch( PhaseType )
   {
      case ScanPhaseType::WriteInFirstPhase:
      {
         // artificial second phase - uniform shift of a pre-scanned block
         shift = reduction( shift, blockShifts[ 0 ] );
         typename InputArray::IndexType outputEnd = outputBegin + end - begin;
         for( typename InputArray::IndexType i = outputBegin; i < outputEnd; i++ )
            output[ i ] = reduction( output[ i ], shift );
         break;
      }

      case ScanPhaseType::WriteInSecondPhase:
      {
         // artificial second phase - only one block, use the shift as the initial value
         perform( input, output, begin, end, outputBegin, reduction, reduction( shift, blockShifts[ 0 ] ) );
         break;
      }
   }
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
void
Scan< Devices::Host, Type, PhaseType >::
perform( const InputArray& input,
         OutputArray& output,
         typename InputArray::IndexType begin,
         typename InputArray::IndexType end,
         typename OutputArray::IndexType outputBegin,
         Reduction&& reduction,
         typename OutputArray::ValueType identity )
{
#ifdef HAVE_OPENMP
   using ValueType = typename OutputArray::ValueType;
   using IndexType = typename InputArray::IndexType;

   if( end <= begin )
      return;

   const IndexType size = end - begin;
   const int max_threads = Devices::Host::getMaxThreadsCount();
   const IndexType block_size = noaTNL::max( 1024, noaTNL::roundUpDivision( size, max_threads ) );
   const IndexType blocks = noaTNL::roundUpDivision( size, block_size );

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = noaTNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      Containers::Array< ValueType > block_results( blocks + 1 );

      #pragma omp parallel num_threads(threads)
      {
         const int block_idx = omp_get_thread_num();
         const IndexType block_offset = block_idx * block_size;
         const IndexType block_begin = begin + block_offset;
         const IndexType block_end = noaTNL::min( block_begin + block_size, end );
         const IndexType block_output_begin = outputBegin + block_offset;

         switch( PhaseType )
         {
            case ScanPhaseType::WriteInFirstPhase:
            {
               // step 1: pre-scan the block and save the result of the block reduction
               block_results[ block_idx ] = Scan< Devices::Sequential, Type >::perform( input, output, block_begin, block_end, block_output_begin, reduction, identity );

               #pragma omp barrier

               // step 2: scan the block results
               #pragma omp single
               {
                  Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, block_results, 0, blocks + 1, 0, reduction, identity );
               }

               // step 3: uniform shift of the pre-scanned block
               const ValueType block_shift = block_results[ block_idx ];
               const IndexType block_output_end = block_output_begin + block_end - block_begin;
               for( IndexType i = block_output_begin; i < block_output_end; i++ )
                  output[ i ] = reduction( output[ i ], block_shift );

               break;
            }

            case ScanPhaseType::WriteInSecondPhase:
            {
               // step 1: per-block reductions, write the result into the buffer
               block_results[ block_idx ] = reduce< Devices::Sequential >( block_begin, block_end, input, reduction, identity );

               #pragma omp barrier

               // step 2: scan the block results
               #pragma omp single
               {
                  Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, block_results, 0, blocks + 1, 0, reduction, identity );
               }

               // step 3: per-block scan using the block results as initial values
               Scan< Devices::Sequential, Type >::perform( input, output, block_begin, block_end, block_output_begin, reduction, block_results[ block_idx ] );

               break;
            }
         }
      }
   }
   else
#endif
      Scan< Devices::Sequential, Type >::perform( input, output, begin, end, outputBegin, reduction, identity );
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
auto
Scan< Devices::Host, Type, PhaseType >::
performFirstPhase( const InputArray& input,
                   OutputArray& output,
                   typename InputArray::IndexType begin,
                   typename InputArray::IndexType end,
                   typename OutputArray::IndexType outputBegin,
                   Reduction&& reduction,
                   typename OutputArray::ValueType identity )
{
#ifdef HAVE_OPENMP
   using ValueType = typename OutputArray::ValueType;
   using IndexType = typename InputArray::IndexType;

   if( end <= begin ) {
      Containers::Array< ValueType, Devices::Sequential > block_results( 1 );
      block_results.setValue( identity );
      return block_results;
   }

   const IndexType size = end - begin;
   const int max_threads = Devices::Host::getMaxThreadsCount();
   const IndexType block_size = noaTNL::max( 1024, noaTNL::roundUpDivision( size, max_threads ) );
   const IndexType blocks = noaTNL::roundUpDivision( size, block_size );

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = noaTNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      Containers::Array< ValueType, Devices::Sequential > block_results( blocks + 1 );

      #pragma omp parallel num_threads(threads)
      {
         const int block_idx = omp_get_thread_num();
         const IndexType block_offset = block_idx * block_size;
         const IndexType block_begin = begin + block_offset;
         const IndexType block_end = noaTNL::min( block_begin + block_size, end );
         const IndexType block_output_begin = outputBegin + block_offset;

         switch( PhaseType )
         {
            case ScanPhaseType::WriteInFirstPhase:
            {
               // pre-scan the block, write the result of the block reduction into the buffer
               block_results[ block_idx ] = Scan< Devices::Sequential, Type >::perform( input, output, block_begin, block_end, block_output_begin, reduction, identity );
               break;
            }

            case ScanPhaseType::WriteInSecondPhase:
            {
               // upsweep: per-block reductions, write the result into the buffer
               block_results[ block_idx ] = reduce< Devices::Sequential >( block_begin, block_end, input, reduction, identity );
               break;
            }
         }
      }

      // spine step: scan the block results
      Scan< Devices::Sequential, ScanType::Exclusive >::perform( block_results, block_results, 0, blocks + 1, 0, reduction, identity );

      // block_results now contains shift values for each block - to be used in the second phase
      return block_results;
   }
   else
#endif
      return Scan< Devices::Sequential, Type >::performFirstPhase( input, output, begin, end, outputBegin, reduction, identity );
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Host, Type, PhaseType >::
performSecondPhase( const InputArray& input,
                    OutputArray& output,
                    const BlockShifts& blockShifts,
                    typename InputArray::IndexType begin,
                    typename InputArray::IndexType end,
                    typename OutputArray::IndexType outputBegin,
                    Reduction&& reduction,
                    typename OutputArray::ValueType identity,
                    typename OutputArray::ValueType shift )
{
#ifdef HAVE_OPENMP
   using ValueType = typename OutputArray::ValueType;
   using IndexType = typename InputArray::IndexType;

   if( end <= begin )
      return;

   const IndexType size = end - begin;
   const int max_threads = Devices::Host::getMaxThreadsCount();
   const IndexType block_size = noaTNL::max( 1024, noaTNL::roundUpDivision( size, max_threads ) );
   const IndexType blocks = noaTNL::roundUpDivision( size, block_size );

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = noaTNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      #pragma omp parallel num_threads(threads)
      {
         const int block_idx = omp_get_thread_num();
         const IndexType block_offset = block_idx * block_size;
         const IndexType block_begin = begin + block_offset;
         const IndexType block_end = noaTNL::min( block_begin + block_size, end );
         const IndexType block_output_begin = outputBegin + block_offset;

         const ValueType block_shift = reduction( shift, blockShifts[ block_idx ] );

         switch( PhaseType )
         {
            case ScanPhaseType::WriteInFirstPhase:
            {
               // uniform shift of a pre-scanned block
               const IndexType block_output_end = block_output_begin + block_end - block_begin;
               for( IndexType i = block_output_begin; i < block_output_end; i++ )
                  output[ i ] = reduction( output[ i ], block_shift );
               break;
            }

            case ScanPhaseType::WriteInSecondPhase:
            {
               // downsweep: per-block scan using the block results as initial values
               Scan< Devices::Sequential, Type >::perform( input, output, block_begin, block_end, block_output_begin, reduction, block_shift );
               break;
            }
         }
      }
   }
   else
#endif
      Scan< Devices::Sequential, Type >::performSecondPhase( input, output, blockShifts, begin, end, outputBegin, reduction, identity, shift );
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
void
Scan< Devices::Cuda, Type, PhaseType >::
perform( const InputArray& input,
         OutputArray& output,
         typename InputArray::IndexType begin,
         typename InputArray::IndexType end,
         typename OutputArray::IndexType outputBegin,
         Reduction&& reduction,
         typename OutputArray::ValueType identity )
{
#ifdef HAVE_CUDA
   if( end <= begin )
      return;

   detail::CudaScanKernelLauncher< Type, PhaseType, typename OutputArray::ValueType >::perform(
      input,
      output,
      begin,
      end,
      outputBegin,
      std::forward< Reduction >( reduction ),
      identity );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename Reduction >
auto
Scan< Devices::Cuda, Type, PhaseType >::
performFirstPhase( const InputArray& input,
                   OutputArray& output,
                   typename InputArray::IndexType begin,
                   typename InputArray::IndexType end,
                   typename OutputArray::IndexType outputBegin,
                   Reduction&& reduction,
                   typename OutputArray::ValueType identity )
{
#ifdef HAVE_CUDA
   if( end <= begin ) {
      Containers::Array< typename OutputArray::ValueType, Devices::Cuda > block_results( 1 );
      block_results.setValue( identity );
      return block_results;
   }

   return detail::CudaScanKernelLauncher< Type, PhaseType, typename OutputArray::ValueType >::performFirstPhase(
      input,
      output,
      begin,
      end,
      outputBegin,
      std::forward< Reduction >( reduction ),
      identity );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< ScanType Type, ScanPhaseType PhaseType >
   template< typename InputArray,
             typename OutputArray,
             typename BlockShifts,
             typename Reduction >
void
Scan< Devices::Cuda, Type, PhaseType >::
performSecondPhase( const InputArray& input,
                    OutputArray& output,
                    const BlockShifts& blockShifts,
                    typename InputArray::IndexType begin,
                    typename InputArray::IndexType end,
                    typename OutputArray::IndexType outputBegin,
                    Reduction&& reduction,
                    typename OutputArray::ValueType identity,
                    typename OutputArray::ValueType shift )
{
#ifdef HAVE_CUDA
   if( end <= begin )
      return;

   detail::CudaScanKernelLauncher< Type, PhaseType, typename OutputArray::ValueType >::performSecondPhase(
      input,
      output,
      blockShifts,
      begin,
      end,
      outputBegin,
      std::forward< Reduction >( reduction ),
      identity,
      shift );
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

} // namespace detail
} // namespace Algorithms
} // namespace noaTNL
