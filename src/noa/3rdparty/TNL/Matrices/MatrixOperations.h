// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

/*
 * TODO: This is just a temporary file, used only in the CWYGMRES solver.
 * The algorithms should be incorporated into the Matrices::Dense class.
 */

#include <noa/3rdparty/TNL/Exceptions/CudaSupportMissing.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Math.h>
#include <noa/3rdparty/TNL/Cuda/DeviceInfo.h>
#include <noa/3rdparty/TNL/Cuda/SharedMemory.h>

namespace noa::TNL {
namespace Matrices {

template< typename DeviceType = Devices::Host >
class MatrixOperations
{
public:
   /*
    * This function performs the matrix-vector multiplication
    *    y = alpha * A * x + beta * y
    * where:
    *    alpha and beta are scalars,
    *    A is an (lda by n) matrix stored in column-major format,
    *    lda >= m is the leading dimension of two-dimensional array used to store matrix A,
    *    x is a vector of n elements,
    *    y is a vector of m elements.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType,
             typename IndexType >
   static void
   gemv( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType* x,
         const RealType beta,
         RealType* y )
   {
      TNL_ASSERT_GT( m, 0, "m must be positive" );
      TNL_ASSERT_GT( n, 0, "n must be positive" );
      TNL_ASSERT_GE( lda, m, "lda must be at least m" );

      RealType alphax[ n ];
      for( IndexType k = 0; k < n; k++ )
         alphax[ k ] = alpha * x[ k ];

      if( n == 1 ) {
         if( beta != 0.0 ) {
            #ifdef HAVE_OPENMP
            #pragma omp parallel for if( noa::TNL::Devices::Host::isOMPEnabled() )
            #endif
            for( IndexType j = 0; j < m; j++ )
               y[ j ] = A[ j ] * alphax[ 0 ] + beta * y[ j ];
         }
         else {
            // the vector y might be uninitialized, and 0.0 * NaN = NaN
            #ifdef HAVE_OPENMP
            #pragma omp parallel for if( noa::TNL::Devices::Host::isOMPEnabled() )
            #endif
            for( IndexType j = 0; j < m; j++ )
               y[ j ] = A[ j ] * alphax[ 0 ];
         }
      }
      else {
         // the matrix A should be accessed column-wise so we split the work into small
         // blocks and each block process by columns, either parallelly or serially
         constexpr IndexType block_size = 128;
         const IndexType blocks = m / block_size;

         #ifdef HAVE_OPENMP
         #pragma omp parallel if( noa::TNL::Devices::Host::isOMPEnabled() && blocks >= 2 )
         #endif
         {
            RealType aux[ block_size ];

            #ifdef HAVE_OPENMP
            #pragma omp for nowait
            #endif
            for( IndexType b = 0; b < blocks; b++ ) {
               const IndexType block_offset = b * block_size;

               // initialize array for thread-local results
               for( IndexType j = 0; j < block_size; j++ )
                  aux[ j ] = 0.0;

               // compute aux = A * alphax
               for( IndexType k = 0; k < n; k++ ) {
                  const IndexType offset = block_offset + k * lda;
                  for( IndexType j = 0; j < block_size; j++ )
                     aux[ j ] += A[ offset + j ] * alphax[ k ];
               }

               // write result: y = aux + beta * y
               if( beta != 0.0 ) {
                  for( IndexType j = 0; j < block_size; j++ )
                     y[ block_offset + j ] = aux[ j ] + beta * y[ block_offset + j ];
               }
               else {
                  // the vector y might be uninitialized, and 0.0 * NaN = NaN
                  for( IndexType j = 0; j < block_size; j++ )
                     y[ block_offset + j ] = aux[ j ];
               }
            }

            // the first thread that reaches here processes the last, incomplete block
            #ifdef HAVE_OPENMP
            #pragma omp single nowait
            #endif
            {
               // TODO: unlike the complete blocks, the tail is traversed row-wise
               if( beta != 0.0 ) {
                  for( IndexType j = blocks * block_size; j < m; j++ ) {
                     RealType tmp = 0.0;
                     for( IndexType k = 0; k < n; k++ )
                        tmp += A[ j + k * lda ] * alphax[ k ];
                     y[ j ] = tmp + beta * y[ j ];
                  }
               }
               else {
                  // the vector y might be uninitialized, and 0.0 * NaN = NaN
                  for( IndexType j = blocks * block_size; j < m; j++ ) {
                     RealType tmp = 0.0;
                     for( IndexType k = 0; k < n; k++ )
                        tmp += A[ j + k * lda ] * alphax[ k ];
                     y[ j ] = tmp;
                  }
               }
            }
         }
      }

   }

   /*
    * This function performs the matrix-matrix addition
    *    C = alpha * A + beta * B
    * where:
    *    alpha and beta are scalars,
    *    A, B, C are (m by n) matrices stored in column-major format on Devices::Cuda,
    *    lda, ldb, ldc (all >= m) are the leading dimensions of matrices A, B, C,
    *    respectively.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType,
             typename IndexType >
   static void
   geam( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType beta,
         const RealType* B,
         const IndexType ldb,
         RealType* C,
         const IndexType ldc )
   {
      TNL_ASSERT_GT( m, 0, "m must be positive" );
      TNL_ASSERT_GT( n, 0, "n must be positive" );
      TNL_ASSERT_GE( lda, m, "lda must be at least m" );
      TNL_ASSERT_GE( ldb, m, "lda must be at least m" );
      TNL_ASSERT_GE( ldc, m, "lda must be at least m" );

      if( n == 1 ) {
         #ifdef HAVE_OPENMP
         #pragma omp parallel for if( noa::TNL::Devices::Host::isOMPEnabled() )
         #endif
         for( IndexType j = 0; j < m; j++ )
            C[ j ] = alpha * A[ j ] + beta * B[ j ];
      }
      else {
         // all matrices should be accessed column-wise so we split the work into small
         // blocks and each block process by columns, either parallelly or serially
         constexpr IndexType block_size = 128;
         const IndexType blocks = m / block_size;

         #ifdef HAVE_OPENMP
         #pragma omp parallel if( noa::TNL::Devices::Host::isOMPEnabled() && blocks >= 2 )
         #endif
         {
            #ifdef HAVE_OPENMP
            #pragma omp for nowait
            #endif
            for( IndexType b = 0; b < blocks; b++ ) {
               const IndexType block_offset = b * block_size;
               for( IndexType j = 0; j < n; j++ ) {
                  const IndexType offset_A = j * lda + block_offset;
                  const IndexType offset_B = j * ldb + block_offset;
                  const IndexType offset_C = j * ldc + block_offset;
                  for( IndexType i = 0; i < block_size; i++ )
                     C[ offset_C + i ] = alpha * A[ offset_A + i ] + beta * B[ offset_B + i ];
               }
            }

            // the first thread that reaches here processes the last, incomplete block
            #ifdef HAVE_OPENMP
            #pragma omp single nowait
            #endif
            {
               for( IndexType j = 0; j < n; j++ ) {
                  const IndexType offset_A = j * lda;
                  const IndexType offset_B = j * ldb;
                  const IndexType offset_C = j * ldc;
                  for( IndexType i = blocks * block_size; i < m; i++ )
                     C[ offset_C + i ] = alpha * A[ offset_A + i ] + beta * B[ offset_B + i ];
               }
            }
         }
      }

   }
};


// CUDA kernels
#ifdef HAVE_CUDA

template< typename RealType,
          typename IndexType >
__global__ void
GemvCudaKernel( const IndexType m,
                const IndexType n,
                const RealType alpha,
                const RealType* A,
                const IndexType lda,
                const RealType* x,
                const RealType beta,
                RealType* y )
{
   IndexType elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   RealType* shx = Cuda::getSharedMemory< RealType >();

   if( threadIdx.x < n )
      shx[ threadIdx.x ] = alpha * x[ threadIdx.x ];
   __syncthreads();

   if( beta != 0.0 ) {
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = tmp + beta * y[ elementIdx ];
         elementIdx += gridSize;
      }
   }
   else {
      // the vector y might be uninitialized, and 0.0 * NaN = NaN
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = tmp;
         elementIdx += gridSize;
      }
   }
}

template< typename RealType,
          typename IndexType >
__global__ void
GeamCudaKernel( const IndexType m,
                const IndexType n,
                const RealType alpha,
                const RealType* A,
                const IndexType lda,
                const RealType beta,
                const RealType* B,
                const IndexType ldb,
                RealType* C,
                const IndexType ldc )
{
   IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSizeX = blockDim.x * gridDim.x;
   const IndexType y = blockIdx.y * blockDim.y + threadIdx.y;
   const IndexType offset_A = y * lda;
   const IndexType offset_B = y * ldb;
   const IndexType offset_C = y * ldc;

   if( y < n )
      while( x < m ) {
         C[ x + offset_C ] = alpha * A[ x + offset_A ] + beta * B[ x + offset_B ];
         x += gridSizeX;
      }
}
#endif

// specialization for CUDA
template<>
class MatrixOperations< Devices::Cuda >
{
public:
   /*
    * This function performs the matrix-vector multiplication
    *    y = alpha * A * x + beta * y
    * where:
    *    alpha and beta are scalars,
    *    A is an (lda by n) matrix stored in column-major format on Devices::Cuda,
    *    lda >= m is the leading dimension of two-dimensional array used to store matrix A,
    *    x is a vector of n elements, stored on Devices::Host,
    *    y is a vector of m elements, stored on Devices::Cuda.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType,
             typename IndexType >
   static void
   gemv( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType* x,
         const RealType beta,
         RealType* y )
   {
      TNL_ASSERT( m <= lda, );
      TNL_ASSERT( n <= 256,
              std::cerr << "The gemv kernel is optimized only for small 'n' and assumes that n <= 256." << std::endl; );

#ifdef HAVE_CUDA
      // TODO: use static storage, e.g. from the CudaReductionBuffer, to avoid frequent reallocations
      Containers::Vector< RealType, Devices::Cuda, IndexType > xDevice;
      xDevice.setSize( n );
      Algorithms::MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::copy< RealType, RealType, IndexType >( xDevice.getData(), x, n );

      // desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
      const int desGridSize = 32 * Cuda::DeviceInfo::getCudaMultiprocessors( Cuda::DeviceInfo::getActiveDevice() );
      dim3 blockSize, gridSize;
      blockSize.x = 256;
      gridSize.x = min( desGridSize, Cuda::getNumberOfBlocks( m, blockSize.x ) );

      GemvCudaKernel<<< gridSize, blockSize, n * sizeof( RealType ) >>>(
            m, n,
            alpha, A, lda,
            xDevice.getData(), beta, y );
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }

   /*
    * This function performs the matrix-matrix addition
    *    C = alpha * A + beta * B
    * where:
    *    alpha and beta are scalars,
    *    A, B, C are (m by n) matrices stored in column-major format on Devices::Cuda,
    *    lda, ldb, ldc (all >= m) are the leading dimensions of matrices A, B, C,
    *    respectively.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType,
             typename IndexType >
   static void
   geam( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType beta,
         const RealType* B,
         const IndexType ldb,
         RealType* C,
         const IndexType ldc )
   {
      TNL_ASSERT_GT( m, 0, "m must be positive" );
      TNL_ASSERT_GT( n, 0, "n must be positive" );
      TNL_ASSERT_GE( lda, m, "lda must be at least m" );
      TNL_ASSERT_GE( ldb, m, "lda must be at least m" );
      TNL_ASSERT_GE( ldc, m, "lda must be at least m" );

#ifdef HAVE_CUDA
      dim3 blockSize, gridSize;

      // max 16 columns of threads
      blockSize.y = min( n, 16 );
      // max 256 threads per block, power of 2
      blockSize.x = 256;
      while( blockSize.x * blockSize.y > 256 )
         blockSize.x /= 2;

      // desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
      const int desGridSize = 32 * Cuda::DeviceInfo::getCudaMultiprocessors( Cuda::DeviceInfo::getActiveDevice() );
      gridSize.x = min( desGridSize, Cuda::getNumberOfBlocks( m, blockSize.x ) );
      gridSize.y = Cuda::getNumberOfBlocks( n, blockSize.y );

      GeamCudaKernel<<< gridSize, blockSize >>>(
            m, n,
            alpha, A, lda,
            beta, B, ldb,
            C, ldc );
      TNL_CHECK_CUDA_DEVICE;
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

} // namespace Matrices
} // namespace noa::TNL
