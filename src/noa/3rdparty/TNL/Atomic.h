// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <atomic>  // std::atomic

#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Devices/Sequential.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>

// double-precision atomicAdd function for Maxwell and older GPUs
// copied from: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#ifdef HAVE_CUDA
#if __CUDA_ARCH__ < 600
namespace {
   __device__ double atomicAdd(double* address, double val)
   {
       unsigned long long int* address_as_ull =
                                 (unsigned long long int*)address;
       unsigned long long int old = *address_as_ull, assumed;

       do {
           assumed = old;
           old = atomicCAS(address_as_ull, assumed,
                           __double_as_longlong(val +
                                  __longlong_as_double(assumed)));

       // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
       } while (assumed != old);

       return __longlong_as_double(old);
   }
} // namespace
#endif
#endif

namespace noa::TNL {

template< typename T, typename Device >
class Atomic;

template< typename T >
class Atomic< T, Devices::Host >
: public std::atomic< T >
{
public:
   Atomic() noexcept = default;

   // inherit constructors
   using std::atomic< T >::atomic;

   // NOTE: std::atomic is not copyable (see https://stackoverflow.com/a/15250851 for
   // an explanation), but we need copyability for noa::TNL::Containers::Array. Note that
   // this copy-constructor and copy-assignment operator are not atomic as they
   // synchronize only with respect to one or the other object.
   Atomic( const Atomic& desired ) noexcept
   : std::atomic< T >()
   {
      this->store(desired.load());
   }
   Atomic& operator=( const Atomic& desired ) noexcept
   {
      this->store(desired.load());
      return *this;
   }

   // CAS loops for updating maximum and minimum
   // reference: https://stackoverflow.com/a/16190791
   T fetch_max( T value ) noexcept
   {
      const T old = *this;
      T prev_value = old;
      while(prev_value < value &&
            ! this->compare_exchange_weak(prev_value, value))
         ;
      return old;
   }

   T fetch_min( T value ) noexcept
   {
      const T old = *this;
      T prev_value = old;
      while(prev_value > value &&
            ! this->compare_exchange_weak(prev_value, value))
         ;
      return old;
   }
};

template< typename T >
class Atomic< T, Devices::Sequential > : public Atomic< T, Devices::Host >
{
   using Base = Atomic< T, Devices::Host >;
   public:

   using Base::Atomic;
   using Base::operator=;
   using Base::fetch_max;
   using Base::fetch_min;
};

template< typename T >
class Atomic< T, Devices::Cuda >
{
public:
   using value_type = T;
   // FIXME
//   using difference_type = typename std::atomic< T >::difference_type;

   __cuda_callable__
   Atomic() noexcept = default;

   __cuda_callable__
   constexpr Atomic( T desired ) noexcept : value(desired) {}

   __cuda_callable__
   T operator=( T desired ) noexcept
   {
      store( desired );
      return desired;
   }

   // NOTE: std::atomic is not copyable (see https://stackoverflow.com/a/15250851 for
   // an explanation), but we need copyability for noa::TNL::Containers::Array. Note that
   // this copy-constructor and copy-assignment operator are not atomic as they
   // synchronize only with respect to one or the other object.
   __cuda_callable__
   Atomic( const Atomic& desired ) noexcept
   {
      // FIXME
//      *this = desired.load();
      *this = desired.value;
   }
   __cuda_callable__
   Atomic& operator=( const Atomic& desired ) noexcept
   {
      // FIXME
//      *this = desired.load();
      *this = desired.value;
      return *this;
   }

   bool is_lock_free() const noexcept
   {
      return true;
   }

   constexpr bool is_always_lock_free() const noexcept
   {
      return true;
   }

   __cuda_callable__
   void store( T desired ) noexcept
   {
      // CUDA does not have a native atomic store, but it can be emulated with atomic exchange
      exchange( desired );
   }

   __cuda_callable__
   T load() const noexcept
   {
      // CUDA does not have a native atomic load:
      // https://stackoverflow.com/questions/32341081/how-to-have-atomic-load-in-cuda

      // const-cast on pointer fails in CUDA 10.1.105
//      return const_cast<Atomic*>(this)->fetch_add( 0 );
      return const_cast<Atomic&>(*this).fetch_add( 0 );
   }

   __cuda_callable__
   operator T() const noexcept
   {
      return load();
   }

   __cuda_callable__
   T exchange( T desired ) noexcept
   {
#ifdef __CUDA_ARCH__
      return atomicExch( &value, desired );
#else
      const T old = value;
      value = desired;
      return old;
#endif
   }

   __cuda_callable__
   bool compare_exchange_weak( T& expected, T desired ) noexcept
   {
      return compare_exchange_strong( expected, desired );
   }

   __cuda_callable__
   bool compare_exchange_strong( T& expected, T desired ) noexcept
   {
#ifdef __CUDA_ARCH__
      const T old = atomicCAS( &value, expected, desired );
      const bool result = old == expected;
      expected = old;
      return result;
#else
      if( value == expected ) {
         value = desired;
         return true;
      }
      else {
         expected = value;
         return false;
      }
#endif
   }

   __cuda_callable__
   T fetch_add( T arg )
   {
#ifdef __CUDA_ARCH__
      return atomicAdd( &value, arg );
#else
      const T old = value;
      value += arg;
      return old;
#endif
   }

   __cuda_callable__
   T fetch_sub( T arg )
   {
#ifdef __CUDA_ARCH__
      return atomicSub( &value, arg );
#else
      const T old = value;
      value -= arg;
      return old;
#endif
   }

   __cuda_callable__
   T fetch_and( T arg )
   {
#ifdef __CUDA_ARCH__
      return atomicAnd( &value, arg );
#else
      const T old = value;
      value = value & arg;
      return old;
#endif
   }

   __cuda_callable__
   T fetch_or( T arg )
   {
#ifdef __CUDA_ARCH__
      return atomicOr( &value, arg );
#else
      const T old = value;
      value = value | arg;
      return old;
#endif
   }

   __cuda_callable__
   T fetch_xor( T arg )
   {
#ifdef __CUDA_ARCH__
      return atomicXor( &value, arg );
#else
      const T old = value;
      value = value ^ arg;
      return old;
#endif
   }

   __cuda_callable__
   T operator+=( T arg ) noexcept
   {
      return fetch_add( arg ) + arg;
   }

   __cuda_callable__
   T operator-=( T arg ) noexcept
   {
      return fetch_sub( arg ) - arg;
   }

   __cuda_callable__
   T operator&=( T arg ) noexcept
   {
      return fetch_and( arg ) & arg;
   }

   __cuda_callable__
   T operator|=( T arg ) noexcept
   {
      return fetch_or( arg ) | arg;
   }

   __cuda_callable__
   T operator^=( T arg ) noexcept
   {
      return fetch_xor( arg ) ^ arg;
   }

   // pre-increment
   __cuda_callable__
   T operator++() noexcept
   {
      return fetch_add(1) + 1;
   }

   // post-increment
   __cuda_callable__
   T operator++(int) noexcept
   {
      return fetch_add(1);
   }

   // pre-decrement
   __cuda_callable__
   T operator--() noexcept
   {
      return fetch_sub(1) - 1;
   }

   // post-decrement
   __cuda_callable__
   T operator--(int) noexcept
   {
      return fetch_sub(1);
   }

   // extensions (methods not present in C++ standards)

   __cuda_callable__
   T fetch_max( T arg ) noexcept
   {
#ifdef __CUDA_ARCH__
      return atomicMax( &value, arg );
#else
      const T old = value;
      value = ( value > arg ) ? value : arg;
      return old;
#endif
   }

   __cuda_callable__
   T fetch_min( T arg ) noexcept
   {
#ifdef __CUDA_ARCH__
      return atomicMin( &value, arg );
#else
      const T old = value;
      value = ( value < arg ) ? value : arg;
      return old;
#endif
   }

protected:
   T value;
};

} // namespace noa::TNL
