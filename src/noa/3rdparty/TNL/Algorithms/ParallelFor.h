// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Devices/Sequential.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Cuda/CheckDevice.h>
#include <noa/3rdparty/TNL/Cuda/DeviceInfo.h>
#include <noa/3rdparty/TNL/Cuda/LaunchHelpers.h>
#include <noa/3rdparty/TNL/Exceptions/CudaSupportMissing.h>
#include <noa/3rdparty/TNL/Math.h>

/****
 * The implementation of ParallelFor is not meant to provide maximum performance
 * at every cost, but maximum flexibility for operating with data stored on the
 * device.
 *
 * The grid-stride loop for CUDA has been inspired by Nvidia's blog post:
 * https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
 *
 * Implemented by: Jakub Klinkovsky
 */

namespace noa::TNL {
/**
 * \brief Namespace for fundamental TNL algorithms
 *
 * It contains algorithms like for-loops, memory operations, (parallel) reduction,
 * multireduction, scan etc.
 */
namespace Algorithms {

// TODO: ParallelForMode should be moved to Device (=Executor)

/**
 * \brief Enum for the parallel processing of the for-loop.
 *
 * Synchronous means that the program control returns to the caller when the loop is processed completely.
 * Asynchronous means that the program control returns to the caller immediately even before the loop is processing is finished.
 *
 * Only parallel for-loops in CUDA are affected by this mode.
 */
enum ParallelForMode { SynchronousMode, AsynchronousMode };


/**
 * \brief Parallel for loop for one dimensional interval of indices.
 *
 * \tparam Device specifies the device where the for-loop will be executed.
 *    It can be \ref noa::TNL::Devices::Host, \ref noa::TNL::Devices::Cuda or
 *    \ref noa::TNL::Devices::Sequential.
 * \tparam Mode defines synchronous/asynchronous mode on parallel devices.
 */
template< typename Device = Devices::Sequential,
          ParallelForMode Mode = SynchronousMode >
struct ParallelFor
{
   /**
    * \brief Static method for the execution of the loop.
    *
    * \tparam Index is the type of the loop indices.
    * \tparam Function is the type of the functor to be called in each iteration
    *    (it is usually deduced from the argument used in the function call).
    * \tparam FunctionArgs is a variadic pack of types for additional parameters
    *    that are forwarded to the functor in every iteration.
    *
    * \param start is the left bound of the iteration range `[begin, end)`.
    * \param end is the right bound of the iteration range `[begin, end)`.
    * \param f is the function to be called in each iteration.
    * \param args are additional parameters to be passed to the function f.
    *
    * \par Example
    * \include Algorithms/ParallelForExample.cpp
    * \par Output
    * \include ParallelForExample.out
    *
    */
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index start, Index end, Function f, FunctionArgs... args )
   {
      for( Index i = start; i < end; i++ )
         f( i, args... );
   }
};

/**
 * \brief Parallel for loop for two dimensional domain of indices.
 *
 * \tparam Device specifies the device where the for-loop will be executed.
 *    It can be \ref noa::TNL::Devices::Host, \ref noa::TNL::Devices::Cuda or
 *    \ref noa::TNL::Devices::Sequential.
 * \tparam Mode defines synchronous/asynchronous mode on parallel devices.
 */
template< typename Device = Devices::Sequential,
          ParallelForMode Mode = SynchronousMode >
struct ParallelFor2D
{
   /**
    * \brief Static method for the execution of the loop.
    *
    * \tparam Index is the type of the loop indices.
    * \tparam Function is the type of the functor to be called in each iteration
    *    (it is usually deduced from the argument used in the function call).
    * \tparam FunctionArgs is a variadic pack of types for additional parameters
    *    that are forwarded to the functor in every iteration.
    *
    * \param startX the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param startY the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param endX the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param endY the for-loop iterates over index domain `[startX,endX) x [startY,endY)`.
    * \param f is the function to be called in each iteration
    * \param args are additional parameters to be passed to the function f.
    *
    * The function f is called for each iteration as
    *
    * \code
    * f( i, j, args... )
    * \endcode
    *
    * where the first parameter is changing more often than the second one.
    *
    * \par Example
    * \include Algorithms/ParallelForExample-2D.cpp
    * \par Output
    * \include ParallelForExample-2D.out
    *
    */
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
   {
      for( Index j = startY; j < endY; j++ )
      for( Index i = startX; i < endX; i++ )
         f( i, j, args... );
   }
};

/**
 * \brief Parallel for loop for three dimensional domain of indices.
 *
 * \tparam Device specifies the device where the for-loop will be executed.
 *    It can be \ref noa::TNL::Devices::Host, \ref noa::TNL::Devices::Cuda or
 *    \ref noa::TNL::Devices::Sequential.
 * \tparam Mode defines synchronous/asynchronous mode on parallel devices.
 */
template< typename Device = Devices::Sequential,
          ParallelForMode Mode = SynchronousMode >
struct ParallelFor3D
{
   /**
    * \brief Static method for the execution of the loop.
    *
    * \tparam Index is the type of the loop indices.
    * \tparam Function is the type of the functor to be called in each iteration
    *    (it is usually deduced from the argument used in the function call).
    * \tparam FunctionArgs is a variadic pack of types for additional parameters
    *    that are forwarded to the functor in every iteration.
    *
    * \param startX the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param startY the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param startZ the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param endX the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param endY the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param endZ the for-loop iterates over index domain `[startX,endX) x [startY,endY) x [startZ,endZ)`.
    * \param f is the function to be called in each iteration
    * \param args are additional parameters to be passed to the function f.
    *
    * The function f is called for each iteration as
    *
    * \code
    * f( i, j, k, args... )
    * \endcode
    *
    * where the first parameter is changing the most often.
    *
    * \par Example
    * \include Algorithms/ParallelForExample-3D.cpp
    * \par Output
    * \include ParallelForExample-3D.out
    *
    */
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
   {
      for( Index k = startZ; k < endZ; k++ )
      for( Index j = startY; j < endY; j++ )
      for( Index i = startX; i < endX; i++ )
         f( i, j, k, args... );
   }
};

template< ParallelForMode Mode >
struct ParallelFor< Devices::Host, Mode >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index start, Index end, Function f, FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      // Benchmarks show that this is significantly faster compared
      // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() && end - start > 512 )'
      if( Devices::Host::isOMPEnabled() && end - start > 512 )
      {
         #pragma omp parallel for
         for( Index i = start; i < end; i++ )
            f( i, args... );
      }
      else
         ParallelFor< Devices::Sequential >::exec( start, end, f, args... );
#else
      ParallelFor< Devices::Sequential >::exec( start, end, f, args... );
#endif
   }
};

template< ParallelForMode Mode >
struct ParallelFor2D< Devices::Host, Mode >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      // Benchmarks show that this is significantly faster compared
      // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
      if( Devices::Host::isOMPEnabled() )
      {
         #pragma omp parallel for
         for( Index j = startY; j < endY; j++ )
         for( Index i = startX; i < endX; i++ )
            f( i, j, args... );
      }
      else
         ParallelFor2D< Devices::Sequential >::exec( startX, startY, endX, endY, f, args... );
#else
      ParallelFor2D< Devices::Sequential >::exec( startX, startY, endX, endY, f, args... );
#endif
   }
};

template< ParallelForMode Mode >
struct ParallelFor3D< Devices::Host, Mode >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      // Benchmarks show that this is significantly faster compared
      // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
      if( Devices::Host::isOMPEnabled() )
      {
         #pragma omp parallel for collapse(2)
         for( Index k = startZ; k < endZ; k++ )
         for( Index j = startY; j < endY; j++ )
         for( Index i = startX; i < endX; i++ )
            f( i, j, k, args... );
      }
      else
         ParallelFor3D< Devices::Sequential >::exec( startX, startY, startZ, endX, endY, endZ, f, args... );
#else
      ParallelFor3D< Devices::Sequential >::exec( startX, startY, startZ, endX, endY, endZ, f, args... );
#endif
   }
};

#ifdef HAVE_CUDA
template< bool gridStrideX = true,
          typename Index,
          typename Function,
          typename... FunctionArgs >
__global__ void
ParallelForKernel( Index start, Index end, Function f, FunctionArgs... args )
{
   Index i = start + blockIdx.x * blockDim.x + threadIdx.x;
   while( i < end ) {
      f( i, args... );
      if( gridStrideX ) i += blockDim.x * gridDim.x;
      else break;
   }
}

template< bool gridStrideX = true,
          bool gridStrideY = true,
          typename Index,
          typename Function,
          typename... FunctionArgs >
__global__ void
ParallelFor2DKernel( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
{
   Index j = startY + blockIdx.y * blockDim.y + threadIdx.y;
   Index i = startX + blockIdx.x * blockDim.x + threadIdx.x;
   while( j < endY ) {
      while( i < endX ) {
         f( i, j, args... );
         if( gridStrideX ) i += blockDim.x * gridDim.x;
         else break;
      }
      if( gridStrideY ) j += blockDim.y * gridDim.y;
      else break;
   }
}

template< bool gridStrideX = true,
          bool gridStrideY = true,
          bool gridStrideZ = true,
          typename Index,
          typename Function,
          typename... FunctionArgs >
__global__ void
ParallelFor3DKernel( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
{
   Index k = startZ + blockIdx.z * blockDim.z + threadIdx.z;
   Index j = startY + blockIdx.y * blockDim.y + threadIdx.y;
   Index i = startX + blockIdx.x * blockDim.x + threadIdx.x;
   while( k < endZ ) {
      while( j < endY ) {
         while( i < endX ) {
            f( i, j, k, args... );
            if( gridStrideX ) i += blockDim.x * gridDim.x;
            else break;
         }
         if( gridStrideY ) j += blockDim.y * gridDim.y;
         else break;
      }
      if( gridStrideZ ) k += blockDim.z * gridDim.z;
      else break;
   }
}
#endif

template< ParallelForMode Mode >
struct ParallelFor< Devices::Cuda, Mode >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index start, Index end, Function f, FunctionArgs... args )
   {
#ifdef HAVE_CUDA
      if( end > start ) {
         dim3 blockSize( 256 );
         dim3 gridSize;
         gridSize.x = noa::TNL::min( Cuda::getMaxGridSize(), Cuda::getNumberOfBlocks( end - start, blockSize.x ) );

         if( (std::size_t) blockSize.x * gridSize.x >= (std::size_t) end - start )
            ParallelForKernel< false ><<< gridSize, blockSize >>>( start, end, f, args... );
         else {
            // decrease the grid size and align to the number of multiprocessors
            const int desGridSize = 32 * Cuda::DeviceInfo::getCudaMultiprocessors( Cuda::DeviceInfo::getActiveDevice() );
            gridSize.x = noa::TNL::min( desGridSize, Cuda::getNumberOfBlocks( end - start, blockSize.x ) );
            ParallelForKernel< true ><<< gridSize, blockSize >>>( start, end, f, args... );
         }

         if( Mode == SynchronousMode )
         {
            cudaStreamSynchronize(0);
            TNL_CHECK_CUDA_DEVICE;
         }
      }
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

template< ParallelForMode Mode >
struct ParallelFor2D< Devices::Cuda, Mode >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
   {
#ifdef HAVE_CUDA
      if( endX > startX && endY > startY ) {
         const Index sizeX = endX - startX;
         const Index sizeY = endY - startY;

         dim3 blockSize;
         if( sizeX >= sizeY * sizeY ) {
            blockSize.x = noa::TNL::min( 256, sizeX );
            blockSize.y = 1;
         }
         else if( sizeY >= sizeX * sizeX ) {
            blockSize.x = 1;
            blockSize.y = noa::TNL::min( 256, sizeY );
         }
         else {
            blockSize.x = noa::TNL::min( 32, sizeX );
            blockSize.y = noa::TNL::min( 8, sizeY );
         }
         dim3 gridSize;
         gridSize.x = noa::TNL::min( Cuda::getMaxGridSize(), Cuda::getNumberOfBlocks( sizeX, blockSize.x ) );
         gridSize.y = noa::TNL::min( Cuda::getMaxGridSize(), Cuda::getNumberOfBlocks( sizeY, blockSize.y ) );

         dim3 gridCount;
         gridCount.x = roundUpDivision( sizeX, blockSize.x * gridSize.x );
         gridCount.y = roundUpDivision( sizeY, blockSize.y * gridSize.y );

         if( gridCount.x == 1 && gridCount.y == 1 )
            ParallelFor2DKernel< false, false ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );
         else if( gridCount.x == 1 && gridCount.y > 1 )
            ParallelFor2DKernel< false, true ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );
         else if( gridCount.x > 1 && gridCount.y == 1 )
            ParallelFor2DKernel< true, false ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );
         else
            ParallelFor2DKernel< true, true ><<< gridSize, blockSize >>>
               ( startX, startY, endX, endY, f, args... );

         if( Mode == SynchronousMode )
         {
            cudaStreamSynchronize(0);
            TNL_CHECK_CUDA_DEVICE;
         }
      }
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

template< ParallelForMode Mode >
struct ParallelFor3D< Devices::Cuda, Mode >
{
   template< typename Index,
             typename Function,
             typename... FunctionArgs >
   static void exec( Index startX, Index startY, Index startZ, Index endX, Index endY, Index endZ, Function f, FunctionArgs... args )
   {
#ifdef HAVE_CUDA
      if( endX > startX && endY > startY && endZ > startZ ) {
         const Index sizeX = endX - startX;
         const Index sizeY = endY - startY;
         const Index sizeZ = endZ - startZ;

         dim3 blockSize;
         if( sizeX >= sizeY * sizeY * sizeZ * sizeZ ) {
            blockSize.x = noa::TNL::min( 256, sizeX );
            blockSize.y = 1;
            blockSize.z = 1;
         }
         else if( sizeY >= sizeX * sizeX * sizeZ * sizeZ ) {
            blockSize.x = 1;
            blockSize.y = noa::TNL::min( 256, sizeY );
            blockSize.z = 1;
         }
         else if( sizeZ >= sizeX * sizeX * sizeY * sizeY ) {
            blockSize.x = noa::TNL::min( 2, sizeX );
            blockSize.y = noa::TNL::min( 2, sizeY );
            // CUDA allows max 64 for blockSize.z
            blockSize.z = noa::TNL::min( 64, sizeZ );
         }
         else if( sizeX >= sizeZ * sizeZ && sizeY >= sizeZ * sizeZ ) {
            blockSize.x = noa::TNL::min( 32, sizeX );
            blockSize.y = noa::TNL::min( 8, sizeY );
            blockSize.z = 1;
         }
         else if( sizeX >= sizeY * sizeY && sizeZ >= sizeY * sizeY ) {
            blockSize.x = noa::TNL::min( 32, sizeX );
            blockSize.y = 1;
            blockSize.z = noa::TNL::min( 8, sizeZ );
         }
         else if( sizeY >= sizeX * sizeX && sizeZ >= sizeX * sizeX ) {
            blockSize.x = 1;
            blockSize.y = noa::TNL::min( 32, sizeY );
            blockSize.z = noa::TNL::min( 8, sizeZ );
         }
         else {
            blockSize.x = noa::TNL::min( 16, sizeX );
            blockSize.y = noa::TNL::min( 4, sizeY );
            blockSize.z = noa::TNL::min( 4, sizeZ );
         }
         dim3 gridSize;
         gridSize.x = noa::TNL::min( Cuda::getMaxGridSize(), Cuda::getNumberOfBlocks( sizeX, blockSize.x ) );
         gridSize.y = noa::TNL::min( Cuda::getMaxGridSize(), Cuda::getNumberOfBlocks( sizeY, blockSize.y ) );
         gridSize.z = noa::TNL::min( Cuda::getMaxGridSize(), Cuda::getNumberOfBlocks( sizeZ, blockSize.z ) );

         dim3 gridCount;
         gridCount.x = roundUpDivision( sizeX, blockSize.x * gridSize.x );
         gridCount.y = roundUpDivision( sizeY, blockSize.y * gridSize.y );
         gridCount.z = roundUpDivision( sizeZ, blockSize.z * gridSize.z );

         if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z == 1 )
            ParallelFor3DKernel< false, false, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z > 1 )
            ParallelFor3DKernel< false, false, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z == 1 )
            ParallelFor3DKernel< false, true, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z == 1 )
            ParallelFor3DKernel< true, false, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z > 1 )
            ParallelFor3DKernel< false, true, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x > 1 && gridCount.y > 1 && gridCount.z == 1 )
            ParallelFor3DKernel< true, true, false ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z > 1 )
            ParallelFor3DKernel< true, false, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );
         else
            ParallelFor3DKernel< true, true, true ><<< gridSize, blockSize >>>
               ( startX, startY, startZ, endX, endY, endZ, f, args... );

         if( Mode == SynchronousMode )
         {
            cudaStreamSynchronize(0);
            TNL_CHECK_CUDA_DEVICE;
         }
      }
#else
      throw Exceptions::CudaSupportMissing();
#endif
   }
};

} // namespace Algorithms
} // namespace noa::TNL
