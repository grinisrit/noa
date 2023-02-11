// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

/*
 * Declaration of variables for dynamic shared memory is difficult in templated
 * functions. For example, the following does not work for different types T:
 *
 *    template< typename T >
 *    void foo()
 *    {
 *        extern __shared__ T shx[];
 *    }
 *
 * This is because extern variables must be declared exactly once. In templated
 * functions we need to have same variable name with different type, which
 * causes the conflict.
 *
 * Until CUDA 8.0, it was possible to use reinterpret_cast this way:
 *
 *    template< typename T, size_t Alignment >
 *    __device__ T* getSharedMemory()
 *    {
 *       extern __shared__ __align__ ( Alignment ) unsigned char __sdata[];
 *       return reinterpret_cast< T* >( __sdata );
 *    }
 *
 * But since CUDA 9.0 there is a new restriction that the alignment of the
 * extern variable must be the same in all template instances. Therefore we
 * follow the idea introduced in the CUDA samples, where the problem is solved
 * using template class specializations.
 */

#include <stdint.h>

namespace noa::TNL {
namespace Cuda {

#ifdef __CUDACC__
template< typename T, std::size_t _alignment = CHAR_BIT * sizeof( T ) >
struct SharedMemory;

template< typename T >
struct SharedMemory< T, 8 >
{
   __device__
   inline
   operator T*()
   {
      extern __shared__ uint8_t __smem8[];
      return reinterpret_cast< T* >( __smem8 );
   }

   __device__
   inline operator const T*() const
   {
      extern __shared__ uint8_t __smem8[];
      return reinterpret_cast< T* >( __smem8 );
   }
};

template< typename T >
struct SharedMemory< T, 16 >
{
   __device__
   inline
   operator T*()
   {
      extern __shared__ uint16_t __smem16[];
      return reinterpret_cast< T* >( __smem16 );
   }

   __device__
   inline operator const T*() const
   {
      extern __shared__ uint16_t __smem16[];
      return reinterpret_cast< T* >( __smem16 );
   }
};

template< typename T >
struct SharedMemory< T, 32 >
{
   __device__
   inline
   operator T*()
   {
      extern __shared__ uint32_t __smem32[];
      return reinterpret_cast< T* >( __smem32 );
   }

   __device__
   inline operator const T*() const
   {
      extern __shared__ uint32_t __smem32[];
      return reinterpret_cast< T* >( __smem32 );
   }
};

template< typename T >
struct SharedMemory< T, 64 >
{
   __device__
   inline
   operator T*()
   {
      extern __shared__ uint64_t __smem64[];
      return reinterpret_cast< T* >( __smem64 );
   }

   __device__
   inline operator const T*() const
   {
      extern __shared__ uint64_t __smem64[];
      return reinterpret_cast< T* >( __smem64 );
   }
};

template< typename T >
__device__
inline T*
getSharedMemory()
{
   static_assert( sizeof( T ) == 1 || sizeof( T ) == 2 || sizeof( T ) == 4 || sizeof( T ) == 8,
                  "Requested type has unsupported size." );
   return SharedMemory< T >{};
}
#endif

// helper functions for indexing shared memory
inline constexpr int
getNumberOfSharedMemoryBanks()
{
   return 32;
}

template< typename Index >
__device__
Index
getInterleaving( const Index index )
{
   return index + index / Cuda::getNumberOfSharedMemoryBanks();
}

}  // namespace Cuda
}  // namespace noa::TNL
