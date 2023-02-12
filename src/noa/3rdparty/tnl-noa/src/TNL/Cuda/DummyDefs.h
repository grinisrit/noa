// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __CUDACC__

   #define __host__
   #define __device__
   #define __global__

struct dim3
{
   unsigned int x = 1;
   unsigned int y = 1;
   unsigned int z = 1;

   dim3() = default;
   constexpr dim3( const dim3& ) = default;
   constexpr dim3( dim3&& ) = default;

   constexpr dim3( unsigned int x, unsigned int y = 1, unsigned int z = 1 ) : x( x ), y( y ), z( z ) {}
};

using cudaError_t = int;
using cudaStream_t = int;

extern cudaError_t
cudaGetDevice( int* device );
extern cudaError_t
cudaSetDevice( int device );
extern cudaError_t
cudaDeviceSynchronize();

enum
{
   cudaStreamDefault,
   cudaStreamNonBlocking,
};

extern cudaError_t
cudaStreamSynchronize( cudaStream_t stream );

enum cudaFuncCache
{
   cudaFuncCachePreferNone = 0,
   cudaFuncCachePreferShared = 1,
   cudaFuncCachePreferL1 = 2,
   cudaFuncCachePreferEqual = 3
};

template< class T >
static cudaError_t
cudaFuncSetCacheConfig( T* func, enum cudaFuncCache cacheConfig )
{
   return 0;
}

#endif
