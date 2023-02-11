// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <memory>  // std::unique_ptr

//#define CUDA_REDUCTION_PROFILING

#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Multireduction.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MultiDeviceMemoryOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/detail/CudaMultireductionKernel.h>

#ifdef CUDA_REDUCTION_PROFILING
   #include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>
   #include <iostream>
#endif

namespace noa::TNL {
namespace Algorithms {

template< typename Result, typename DataFetcher, typename Reduction, typename Index >
void constexpr Multireduction< Devices::Sequential >::reduce( Result identity,
                                                              DataFetcher dataFetcher,
                                                              Reduction reduction,
                                                              Index size,
                                                              int n,
                                                              Result* result )
{
   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );

   constexpr int block_size = 128;
   const int blocks = size / block_size;

   if( blocks > 1 ) {
      // initialize array for unrolled results
      // (it is accessed as a row-major matrix with n rows and 4 columns)
      std::unique_ptr< Result[] > r{ new Result[ n * 4 ] };
      for( int k = 0; k < n * 4; k++ )
         r[ k ] = identity;

      // main reduction (explicitly unrolled loop)
      for( int b = 0; b < blocks; b++ ) {
         const Index offset = b * block_size;
         for( int k = 0; k < n; k++ ) {
            Result* _r = r.get() + 4 * k;
            for( int i = 0; i < block_size; i += 4 ) {
               _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( offset + i, k ) );
               _r[ 1 ] = reduction( _r[ 1 ], dataFetcher( offset + i + 1, k ) );
               _r[ 2 ] = reduction( _r[ 2 ], dataFetcher( offset + i + 2, k ) );
               _r[ 3 ] = reduction( _r[ 3 ], dataFetcher( offset + i + 3, k ) );
            }
         }
      }

      // reduction of the last, incomplete block (not unrolled)
      for( int k = 0; k < n; k++ ) {
         Result* _r = r.get() + 4 * k;
         for( Index i = blocks * block_size; i < size; i++ )
            _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( i, k ) );
      }

      // reduction of unrolled results
      for( int k = 0; k < n; k++ ) {
         Result* _r = r.get() + 4 * k;
         _r[ 0 ] = reduction( _r[ 0 ], _r[ 1 ] );
         _r[ 0 ] = reduction( _r[ 0 ], _r[ 2 ] );
         _r[ 0 ] = reduction( _r[ 0 ], _r[ 3 ] );

         // copy the result into the output parameter
         result[ k ] = _r[ 0 ];
      }
   }
   else {
      for( int k = 0; k < n; k++ )
         result[ k ] = identity;

      for( int b = 0; b < blocks; b++ ) {
         const Index offset = b * block_size;
         for( int k = 0; k < n; k++ ) {
            for( int i = 0; i < block_size; i++ )
               result[ k ] = reduction( result[ k ], dataFetcher( offset + i, k ) );
         }
      }

      for( int k = 0; k < n; k++ ) {
         for( Index i = blocks * block_size; i < size; i++ )
            result[ k ] = reduction( result[ k ], dataFetcher( i, k ) );
      }
   }
}

template< typename Result, typename DataFetcher, typename Reduction, typename Index >
void
Multireduction< Devices::Host >::reduce( Result identity,
                                         DataFetcher dataFetcher,
                                         Reduction reduction,
                                         Index size,
                                         int n,
                                         Result* result )
{
   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );

#ifdef HAVE_OPENMP
   constexpr int block_size = 128;
   const int blocks = size / block_size;

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = TNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      #pragma omp parallel num_threads(threads)
      {
         // first thread initializes the result array
         #pragma omp single nowait
         {
            for( int k = 0; k < n; k++ )
               result[ k ] = identity;
         }

         // initialize array for thread-local results
         // (it is accessed as a row-major matrix with n rows and 4 columns)
         std::unique_ptr< Result[] > r{ new Result[ n * 4 ] };
         for( int k = 0; k < n * 4; k++ )
            r[ k ] = identity;

         #pragma omp for nowait
         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int k = 0; k < n; k++ ) {
               Result* _r = r.get() + 4 * k;
               for( int i = 0; i < block_size; i += 4 ) {
                  _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( offset + i, k ) );
                  _r[ 1 ] = reduction( _r[ 1 ], dataFetcher( offset + i + 1, k ) );
                  _r[ 2 ] = reduction( _r[ 2 ], dataFetcher( offset + i + 2, k ) );
                  _r[ 3 ] = reduction( _r[ 3 ], dataFetcher( offset + i + 3, k ) );
               }
            }
         }

         // the first thread that reaches here processes the last, incomplete block
         #pragma omp single nowait
         {
            for( int k = 0; k < n; k++ ) {
               Result* _r = r.get() + 4 * k;
               for( Index i = blocks * block_size; i < size; i++ )
                  _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( i, k ) );
            }
         }

         // local reduction of unrolled results
         for( int k = 0; k < n; k++ ) {
            Result* _r = r.get() + 4 * k;
            _r[ 0 ] = reduction( _r[ 0 ], _r[ 1 ] );
            _r[ 0 ] = reduction( _r[ 0 ], _r[ 2 ] );
            _r[ 0 ] = reduction( _r[ 0 ], _r[ 3 ] );
         }

         // inter-thread reduction of local results
         #pragma omp critical
         {
            for( int k = 0; k < n; k++ )
               result[ k ] = reduction( result[ k ], r[ 4 * k ] );
         }
      }
   }
   else
#endif
      Multireduction< Devices::Sequential >::reduce( identity, dataFetcher, reduction, size, n, result );
}

template< typename Result, typename DataFetcher, typename Reduction, typename Index >
void
Multireduction< Devices::Cuda >::reduce( Result identity,
                                         DataFetcher dataFetcher,
                                         Reduction reduction,
                                         Index size,
                                         int n,
                                         Result* hostResult )
{
   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );

#ifdef CUDA_REDUCTION_PROFILING
   Timer timer;
   timer.reset();
   timer.start();
#endif

   // start the reduction on the GPU
   Result* deviceAux1 = nullptr;
   const int reducedSize = detail::CudaMultireductionKernelLauncher( identity, dataFetcher, reduction, size, n, deviceAux1 );

#ifdef CUDA_REDUCTION_PROFILING
   timer.stop();
   std::cout << "   Multireduction of " << n << " datasets on GPU to size " << reducedSize << " took " << timer.getRealTime()
             << " sec. " << std::endl;
   timer.reset();
   timer.start();
#endif

   // transfer the reduced data from device to host
   std::unique_ptr< Result[] > resultArray{ new Result[ n * reducedSize ] };
   MultiDeviceMemoryOperations< void, Devices::Cuda >::copy( resultArray.get(), deviceAux1, n * reducedSize );

#ifdef CUDA_REDUCTION_PROFILING
   timer.stop();
   std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
   timer.reset();
   timer.start();
#endif

   // finish the reduction on the host
   auto dataFetcherFinish = [ & ]( int i, int k )
   {
      return resultArray[ i + k * reducedSize ];
   };
   Multireduction< Devices::Sequential >::reduce( identity, dataFetcherFinish, reduction, reducedSize, n, hostResult );

#ifdef CUDA_REDUCTION_PROFILING
   timer.stop();
   std::cout << "   Multireduction of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
#endif
}

}  // namespace Algorithms
}  // namespace noa::TNL
