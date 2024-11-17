// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/DummyDefs.h>

namespace noa::TNL {
namespace Cuda {

[[deprecated( "this function is deprecated - use either getMaxGridXSize(), getMaxGridYSize(), or "
              "getMaxGridZSize()" )]] inline constexpr std::size_t
getMaxGridSize()
{
   return 65535;
}

inline constexpr std::size_t
getMaxGridXSize()
{
   return 2147483647;
}

inline constexpr std::size_t
getMaxGridYSize()
{
   return 65535;
}

inline constexpr std::size_t
getMaxGridZSize()
{
   return 65535;
}

inline constexpr int
getMaxBlockXSize()
{
   return 1024;
}

inline constexpr int
getMaxBlockYSize()
{
   return 1024;
}

inline constexpr int
getMaxBlockZSize()
{
   return 64;
}

inline constexpr int
getWarpSize()
{
   return 32;
}

// When we transfer data between the GPU and the CPU we use 1 MiB buffer. This
// size should ensure good performance.
// We use the same buffer size even for retyping data during IO operations.
inline constexpr int
getTransferBufferSize()
{
   return 1 << 20;
}

#ifdef __CUDACC__
__device__
inline int
getGlobalThreadIdx_x( const dim3& gridIdx )
{
   return ( gridIdx.x * getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
}

__device__
inline int
getGlobalThreadIdx_y( const dim3& gridIdx )
{
   return ( gridIdx.y * getMaxGridYSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
}

__device__
inline int
getGlobalThreadIdx_z( const dim3& gridIdx )
{
   return ( gridIdx.z * getMaxGridZSize() + blockIdx.z ) * blockDim.z + threadIdx.z;
}
#endif

inline int
getNumberOfBlocks( const int threads, const int blockSize )
{
   return roundUpDivision( threads, blockSize );
}

inline int
getNumberOfGrids( const int blocks, const int gridSize )
{
   return roundUpDivision( blocks, gridSize );
}

inline void
setupThreads( const dim3& blockSize,
              dim3& blocksCount,
              dim3& gridsCount,
              long long int xThreads,
              long long int yThreads = 0,
              long long int zThreads = 0 )
{
   blocksCount.x = max( 1, xThreads / blockSize.x + ( xThreads % blockSize.x != 0 ) );
   blocksCount.y = max( 1, yThreads / blockSize.y + ( yThreads % blockSize.y != 0 ) );
   blocksCount.z = max( 1, zThreads / blockSize.z + ( zThreads % blockSize.z != 0 ) );

   /****
    * TODO: Fix the following:
    * I do not known how to get max grid size in kernels :(
    *
    * Also, this is very slow. */
   /*int currentDevice( 0 );
   cudaGetDevice( currentDevice );
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, currentDevice );
   gridsCount.x = blocksCount.x / properties.maxGridSize[ 0 ] + ( blocksCount.x % properties.maxGridSize[ 0 ] != 0 );
   gridsCount.y = blocksCount.y / properties.maxGridSize[ 1 ] + ( blocksCount.y % properties.maxGridSize[ 1 ] != 0 );
   gridsCount.z = blocksCount.z / properties.maxGridSize[ 2 ] + ( blocksCount.z % properties.maxGridSize[ 2 ] != 0 );
   */
   gridsCount.x = blocksCount.x / getMaxGridXSize() + ( blocksCount.x % getMaxGridXSize() != 0 );
   gridsCount.y = blocksCount.y / getMaxGridYSize() + ( blocksCount.y % getMaxGridYSize() != 0 );
   gridsCount.z = blocksCount.z / getMaxGridZSize() + ( blocksCount.z % getMaxGridZSize() != 0 );
}

inline void
setupGrid( const dim3& blocksCount, const dim3& gridsCount, const dim3& gridIdx, dim3& gridSize )
{
   /* TODO: this is ext slow!!!!
   int currentDevice( 0 );
   cudaGetDevice( &currentDevice );
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, currentDevice );*/

   /****
    * TODO: fix the following
   if( gridIdx.x < gridsCount.x )
      gridSize.x = properties.maxGridSize[ 0 ];
   else
      gridSize.x = blocksCount.x % properties.maxGridSize[ 0 ];

   if( gridIdx.y < gridsCount.y )
      gridSize.y = properties.maxGridSize[ 1 ];
   else
      gridSize.y = blocksCount.y % properties.maxGridSize[ 1 ];

   if( gridIdx.z < gridsCount.z )
      gridSize.z = properties.maxGridSize[ 2 ];
   else
      gridSize.z = blocksCount.z % properties.maxGridSize[ 2 ];*/

   if( gridIdx.x < gridsCount.x - 1 )
      gridSize.x = getMaxGridXSize();
   else
      gridSize.x = blocksCount.x % getMaxGridXSize();

   if( gridIdx.y < gridsCount.y - 1 )
      gridSize.y = getMaxGridYSize();
   else
      gridSize.y = blocksCount.y % getMaxGridYSize();

   if( gridIdx.z < gridsCount.z - 1 )
      gridSize.z = getMaxGridZSize();
   else
      gridSize.z = blocksCount.z % getMaxGridZSize();
}

inline std::ostream&
operator<<( std::ostream& str, const dim3& d )
{
   str << "( " << d.x << ", " << d.y << ", " << d.z << " )";
   return str;
}

inline void
printThreadsSetup( const dim3& blockSize,
                   const dim3& blocksCount,
                   const dim3& gridSize,
                   const dim3& gridsCount,
                   std::ostream& str = std::cout )
{
   str << "Block size: " << blockSize << std::endl
       << " Blocks count: " << blocksCount << std::endl
       << " Grid size: " << gridSize << std::endl
       << " Grids count: " << gridsCount << std::endl;
}

}  // namespace Cuda
}  // namespace noa::TNL
