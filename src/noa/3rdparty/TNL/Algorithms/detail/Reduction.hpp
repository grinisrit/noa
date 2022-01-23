// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <memory>  // std::unique_ptr

//#define CUDA_REDUCTION_PROFILING

#include <noa/3rdparty/TNL/Algorithms/detail/Reduction.h>
#include <noa/3rdparty/TNL/Algorithms/detail/CudaReductionKernel.h>
#include <noa/3rdparty/TNL/Algorithms/MultiDeviceMemoryOperations.h>

#ifdef CUDA_REDUCTION_PROFILING
#include <iostream>
#include <noa/3rdparty/TNL/Timer.h>
#endif

namespace noa::TNL {
   namespace Algorithms {
      namespace detail {

/****
 * Arrays smaller than the following constant
 * are reduced on CPU. The constant must not be larger
 * than maximal CUDA grid size.
 */
static constexpr int Reduction_minGpuDataSize = 256;//65536; //16384;//1024;//256;

template< typename Index,
          typename Result,
          typename Fetch,
          typename Reduce >
constexpr Result
Reduction< Devices::Sequential >::
reduce( const Index begin,
        const Index end,
        Fetch&& fetch,
        Reduce&& reduce,
        const Result& identity )
{
   constexpr int block_size = 128;
   const Index size = end - begin;
   const Index blocks = size / block_size;

   if( blocks > 1 ) {
      // initialize array for unrolled results
      Result r[ 4 ] = { identity, identity, identity, identity };

      // main reduce (explicitly unrolled loop)
      for( Index b = 0; b < blocks; b++ ) {
         const Index offset = begin + b * block_size;
         for( int i = 0; i < block_size; i += 4 ) {
            r[ 0 ] = reduce( r[ 0 ], fetch( offset + i ) );
            r[ 1 ] = reduce( r[ 1 ], fetch( offset + i + 1 ) );
            r[ 2 ] = reduce( r[ 2 ], fetch( offset + i + 2 ) );
            r[ 3 ] = reduce( r[ 3 ], fetch( offset + i + 3 ) );
         }
      }

      // reduce of the last, incomplete block (not unrolled)
      for( Index i = begin + blocks * block_size; i < end; i++ )
         r[ 0 ] = reduce( r[ 0 ], fetch( i ) );

      // reduce of unrolled results
      r[ 0 ] = reduce( r[ 0 ], r[ 2 ] );
      r[ 1 ] = reduce( r[ 1 ], r[ 3 ] );
      r[ 0 ] = reduce( r[ 0 ], r[ 1 ] );
      return r[ 0 ];
   }
   else {
      Result result = identity;
      for( Index i = begin; i < end; i++ )
         result = reduce( result, fetch( i ) );
      return result;
   }
}

template< typename Index,
          typename Result,
          typename Fetch,
          typename Reduce >
constexpr std::pair< Result, Index >
Reduction< Devices::Sequential >::
reduceWithArgument( const Index begin,
                    const Index end,
                    Fetch&& fetch,
                    Reduce&& reduce,
                    const Result& identity )
{
   constexpr int block_size = 128;
   const Index size = end - begin;
   const Index blocks = size / block_size;

   if( blocks > 1 ) {
      // initialize array for unrolled results
      Index arg[ 4 ] = { 0, 0, 0, 0 };
      Result r[ 4 ] = { identity, identity, identity, identity };
      bool initialized( false );

      // main reduce (explicitly unrolled loop)
      for( Index b = 0; b < blocks; b++ ) {
         const Index offset = begin + b * block_size;
         for( int i = 0; i < block_size; i += 4 ) {
            if( ! initialized )
            {
               arg[ 0 ] = offset + i;
               arg[ 1 ] = offset + i + 1;
               arg[ 2 ] = offset + i + 2;
               arg[ 3 ] = offset + i + 3;
               r[ 0 ] = fetch( offset + i );
               r[ 1 ] = fetch( offset + i + 1 );
               r[ 2 ] = fetch( offset + i + 2 );
               r[ 3 ] = fetch( offset + i + 3 );
               initialized = true;
               continue;
            }
            reduce( r[ 0 ], fetch( offset + i ),     arg[ 0 ], offset + i );
            reduce( r[ 1 ], fetch( offset + i + 1 ), arg[ 1 ], offset + i + 1 );
            reduce( r[ 2 ], fetch( offset + i + 2 ), arg[ 2 ], offset + i + 2 );
            reduce( r[ 3 ], fetch( offset + i + 3 ), arg[ 3 ], offset + i + 3 );
         }
      }

      // reduce of the last, incomplete block (not unrolled)
      for( Index i = begin + blocks * block_size; i < size; i++ )
         reduce( r[ 0 ], fetch( i ), arg[ 0 ], i );

      // reduce of unrolled results
      reduce( r[ 0 ], r[ 2 ], arg[ 0 ], arg[ 2 ] );
      reduce( r[ 1 ], r[ 3 ], arg[ 1 ], arg[ 3 ] );
      reduce( r[ 0 ], r[ 1 ], arg[ 0 ], arg[ 1 ] );
      return std::make_pair( r[ 0 ], arg[ 0 ] );
   }
   else if( begin >= end ) {
      // trivial case, fetch should not be called in this case
      return std::make_pair( identity, end );
   }
   else {
      std::pair< Result, Index > result( fetch( begin ), begin );
      for( Index i = begin + 1; i < end; i++ )
         reduce( result.first, fetch( i ), result.second, i );
      return result;
   }
}

template< typename Index,
          typename Result,
          typename Fetch,
          typename Reduce >
Result
Reduction< Devices::Host >::
reduce( const Index begin,
        const Index end,
        Fetch&& fetch,
        Reduce&& reduce,
        const Result& identity )
{
#ifdef HAVE_OPENMP
   constexpr int block_size = 128;
   const Index size = end - begin;
   const Index blocks = size / block_size;

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      // global result variable
      Result result = identity;
      const int threads = noa::TNL::min( blocks, Devices::Host::getMaxThreadsCount() );
#pragma omp parallel num_threads(threads)
      {
         // initialize array for thread-local results
         Result r[ 4 ] = { identity, identity, identity, identity  };

         #pragma omp for nowait
         for( Index b = 0; b < blocks; b++ ) {
            const Index offset = begin + b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               r[ 0 ] = reduce( r[ 0 ], fetch( offset + i ) );
               r[ 1 ] = reduce( r[ 1 ], fetch( offset + i + 1 ) );
               r[ 2 ] = reduce( r[ 2 ], fetch( offset + i + 2 ) );
               r[ 3 ] = reduce( r[ 3 ], fetch( offset + i + 3 ) );
            }
         }

         // the first thread that reaches here processes the last, incomplete block
         #pragma omp single nowait
         {
            for( Index i = begin + blocks * block_size; i < end; i++ )
               r[ 0 ] = reduce( r[ 0 ], fetch( i ) );
         }

         // local reduce of unrolled results
         r[ 0 ] = reduce( r[ 0 ], r[ 2 ] );
         r[ 1 ] = reduce( r[ 1 ], r[ 3 ] );
         r[ 0 ] = reduce( r[ 0 ], r[ 1 ] );

         // inter-thread reduce of local results
         #pragma omp critical
         {
            result = reduce( result, r[ 0 ] );
         }
      }
      return result;
   }
   else
#endif
      return Reduction< Devices::Sequential >::reduce( begin, end, fetch, reduce, identity );
}

template< typename Index,
          typename Result,
          typename Fetch,
          typename Reduce >
std::pair< Result, Index >
Reduction< Devices::Host >::
reduceWithArgument( const Index begin,
                    const Index end,
                    Fetch&& fetch,
                    Reduce&& reduce,
                    const Result& identity )
{
#ifdef HAVE_OPENMP
   constexpr int block_size = 128;
   const Index size = end - begin;
   const Index blocks = size / block_size;

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      // global result variable
      std::pair< Result, Index > result( identity, -1 );
      const int threads = noa::TNL::min( blocks, Devices::Host::getMaxThreadsCount() );
#pragma omp parallel num_threads(threads)
      {
         // initialize array for thread-local results
         Index arg[ 4 ] = { 0, 0, 0, 0 };
         Result r[ 4 ] = { identity, identity, identity, identity  };
         bool initialized( false );

         #pragma omp for nowait
         for( Index b = 0; b < blocks; b++ ) {
            const Index offset = begin + b * block_size;
            for( int i = 0; i < block_size; i += 4 ) {
               if( ! initialized ) {
                  arg[ 0 ] = offset + i;
                  arg[ 1 ] = offset + i + 1;
                  arg[ 2 ] = offset + i + 2;
                  arg[ 3 ] = offset + i + 3;
                  r[ 0 ] = fetch( offset + i );
                  r[ 1 ] = fetch( offset + i + 1 );
                  r[ 2 ] = fetch( offset + i + 2 );
                  r[ 3 ] = fetch( offset + i + 3 );
                  initialized = true;
                  continue;
               }
               reduce( r[ 0 ], fetch( offset + i ),     arg[ 0 ], offset + i );
               reduce( r[ 1 ], fetch( offset + i + 1 ), arg[ 1 ], offset + i + 1 );
               reduce( r[ 2 ], fetch( offset + i + 2 ), arg[ 2 ], offset + i + 2 );
               reduce( r[ 3 ], fetch( offset + i + 3 ), arg[ 3 ], offset + i + 3 );
            }
         }

         // the first thread that reaches here processes the last, incomplete block
         #pragma omp single nowait
         {
            for( Index i = begin + blocks * block_size; i < end; i++ )
               reduce( r[ 0 ], fetch( i ), arg[ 0 ], i );
         }

         // local reduce of unrolled results
         reduce( r[ 0 ], r[ 2 ], arg[ 0 ], arg[ 2 ] );
         reduce( r[ 1 ], r[ 3 ], arg[ 1 ], arg[ 3 ] );
         reduce( r[ 0 ], r[ 1 ], arg[ 0 ], arg[ 1 ] );

         // inter-thread reduce of local results
         #pragma omp critical
         {
            if( result.second == -1 )
               result.second = arg[ 0 ];
            reduce( result.first, r[ 0 ], result.second, arg[ 0 ] );
         }
      }
      return result;
   }
   else
#endif
      return Reduction< Devices::Sequential >::reduceWithArgument( begin, end, fetch, reduce, identity );
}

template< typename Index,
          typename Result,
          typename Fetch,
          typename Reduce >
Result
Reduction< Devices::Cuda >::
reduce( const Index begin,
        const Index end,
        Fetch&& fetch,
        Reduce&& reduce,
        const Result& identity )
{
   // trivial case, nothing to reduce
   if( begin >= end )
      return identity;

   // Only fundamental and pointer types can be safely reduced on host. Complex
   // objects stored on the device might contain pointers into the device memory,
   // in which case reduce on host might fail.
   constexpr bool can_reduce_later_on_host = std::is_fundamental< Result >::value || std::is_pointer< Result >::value;

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   detail::CudaReductionKernelLauncher< Index, Result > reductionLauncher( begin, end );

   // start the reduce on the GPU
   Result* deviceAux1( 0 );
   const int reducedSize = reductionLauncher.start(
      reduce,
      fetch,
      identity,
      deviceAux1 );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Reduction on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

   if( can_reduce_later_on_host ) {
      // transfer the reduced data from device to host
      std::unique_ptr< Result[] > resultArray{
         // Workaround for nvcc 10.1.168 - it would modify the simple expression
         // `new Result[reducedSize]` in the source code to `new (Result[reducedSize])`
         // which is not correct - see e.g. https://stackoverflow.com/a/39671946
         // Thus, the host compiler would spit out hundreds of warnings...
         // Funnily enough, nvcc's behaviour depends on the context rather than the
         // expression, because exactly the same simple expression in different places
         // does not produce warnings.
         #ifdef __NVCC__
         new Result[ static_cast<const int&>(reducedSize) ]
         #else
         new Result[ reducedSize ]
         #endif
      };
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( resultArray.get(), deviceAux1, reducedSize );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      // finish the reduce on the host
      auto fetch = [&] ( Index i ) { return resultArray[ i ]; };
      const Result result = Reduction< Devices::Sequential >::reduce( 0, reducedSize, fetch, reduce, identity );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
      return result;
   }
   else {
      // data can't be safely reduced on host, so continue with the reduce on the GPU
      auto result = reductionLauncher.finish( reduce, identity );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      return result;
   }
}

template< typename Index,
          typename Result,
          typename Fetch,
          typename Reduce >
std::pair< Result, Index >
Reduction< Devices::Cuda >::
reduceWithArgument( const Index begin,
                    const Index end,
                    Fetch&& fetch,
                    Reduce&& reduce,
                    const Result& identity )
{
   // trivial case, nothing to reduce
   if( begin >= end )
      return std::make_pair( identity, end );

   // Only fundamental and pointer types can be safely reduced on host. Complex
   // objects stored on the device might contain pointers into the device memory,
   // in which case reduce on host might fail.
   constexpr bool can_reduce_later_on_host = std::is_fundamental< Result >::value || std::is_pointer< Result >::value;

   #ifdef CUDA_REDUCTION_PROFILING
      Timer timer;
      timer.reset();
      timer.start();
   #endif

   detail::CudaReductionKernelLauncher< Index, Result > reductionLauncher( begin, end );

   // start the reduce on the GPU
   Result* deviceAux1( nullptr );
   Index* deviceIndexes( nullptr );
   const int reducedSize = reductionLauncher.startWithArgument(
      reduce,
      fetch,
      identity,
      deviceAux1,
      deviceIndexes );

   #ifdef CUDA_REDUCTION_PROFILING
      timer.stop();
      std::cout << "   Reduction on GPU to size " << reducedSize << " took " << timer.getRealTime() << " sec. " << std::endl;
      timer.reset();
      timer.start();
   #endif

   if( can_reduce_later_on_host ) {
      // transfer the reduced data from device to host
      std::unique_ptr< Result[] > resultArray{
         // Workaround for nvcc 10.1.168 - it would modify the simple expression
         // `new Result[reducedSize]` in the source code to `new (Result[reducedSize])`
         // which is not correct - see e.g. https://stackoverflow.com/a/39671946
         // Thus, the host compiler would spit out hundreds of warnings...
         // Funnily enough, nvcc's behaviour depends on the context rather than the
         // expression, because exactly the same simple expression in different places
         // does not produce warnings.
         #ifdef __NVCC__
         new Result[ static_cast<const int&>(reducedSize) ]
         #else
         new Result[ reducedSize ]
         #endif
      };
      std::unique_ptr< Index[] > indexArray{
         // Workaround for nvcc 10.1.168 - it would modify the simple expression
         // `new Index[reducedSize]` in the source code to `new (Index[reducedSize])`
         // which is not correct - see e.g. https://stackoverflow.com/a/39671946
         // Thus, the host compiler would spit out hundreds of warnings...
         // Funnily enough, nvcc's behaviour depends on the context rather than the
         // expression, because exactly the same simple expression in different places
         // does not produce warnings.
         #ifdef __NVCC__
         new Index[ static_cast<const int&>(reducedSize) ]
         #else
         new Index[ reducedSize ]
         #endif
      };
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( resultArray.get(), deviceAux1, reducedSize );
      MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( indexArray.get(), deviceIndexes, reducedSize );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      // finish the reduce on the host
//      auto fetch = [&] ( Index i ) { return resultArray[ i ]; };
//      const Result result = Reduction< Devices::Sequential >::reduceWithArgument( reducedSize, argument, reduce, fetch, identity );
      for( Index i = 1; i < reducedSize; i++ )
         reduce( resultArray[ 0 ], resultArray[ i ], indexArray[ 0 ], indexArray[ i ]  );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
      #endif
      return std::make_pair( resultArray[ 0 ], indexArray[ 0 ] );
   }
   else {
      // data can't be safely reduced on host, so continue with the reduce on the GPU
      auto result = reductionLauncher.finishWithArgument( reduce, identity );

      #ifdef CUDA_REDUCTION_PROFILING
         timer.stop();
         std::cout << "   Reduction of small data set on GPU took " << timer.getRealTime() << " sec. " << std::endl;
         timer.reset();
         timer.start();
      #endif

      return result;
   }
}

      } // namespace detail
   } // namespace Algorithms
} // namespace noa::TNL
