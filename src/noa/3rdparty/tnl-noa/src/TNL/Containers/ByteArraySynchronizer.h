// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <future>
// 3rd-party async library providing a thread-pool
#include <noa/3rdparty/tnl-noa/src/TNL/3rdparty/async/threadpool.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ArrayView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Wrappers.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>

namespace noa::TNL {
namespace Containers {

template< typename Device, typename Index >
class ByteArraySynchronizer
{
private:
   // NOTE: async::threadpool has alignment requirements, which causes problems:
   //  - it may become misaligned in derived classes, see e.g.
   //    https://stackoverflow.com/a/46475498
   //    solution: specify it as the first member of the base class
   //  - operator new before C++17 may not support over-aligned types, see
   //    https://stackoverflow.com/a/53485295
   //    solution: relaxed alignment requirements to not exceed the value of
   //    alignof(std::max_align_t), which is the strongest alignment supported
   //    by plain new. See https://github.com/d36u9/async/pull/2
   async::threadpool tp;

   int gpu_id = 0;

public:
   using ByteArrayView = ArrayView< std::uint8_t, Device, Index >;
   using RequestsVector = std::vector< MPI_Request >;

   enum class AsyncPolicy
   {
      synchronous,
      deferred,
      threadpool,
      async,
   };

   ByteArraySynchronizer() : tp( 1 ) {}

   /**
    * \brief Main synchronization function.
    *
    * This is only a pure virtual function -- the functionality must be
    * implemented in a subclass.
    */
   virtual void
   synchronizeByteArray( ByteArrayView array, int bytesPerValue ) = 0;

   virtual RequestsVector
   synchronizeByteArrayAsyncWorker( ByteArrayView array, int bytesPerValue ) = 0;

   /**
    * \brief An asynchronous version of \ref TNL::Containers::ByteArraySynchronizer::synchronizeByteArray
    * "synchronizeByteArray".
    *
    * Note that this method is not thread-safe - only the thread which created
    * and "owns" the instance of this object can call this method.
    *
    * Note that at most one async operation may be active at a time, the
    * following calls will block until the pending operation is finished.
    */
   void
   synchronizeByteArrayAsync( ByteArrayView array, int bytesPerValue, AsyncPolicy policy = AsyncPolicy::synchronous )
   {
      // wait for any previous synchronization (multiple objects can share the
      // same synchronizer)
      if( async_op.valid() ) {
         async_wait_before_start_timer.start();
         async_op.wait();
         async_wait_before_start_timer.stop();
      }

      async_start_timer.start();

      // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
      if constexpr( std::is_same< Device, Devices::Cuda >::value )
         cudaGetDevice( &gpu_id );

      if( policy == AsyncPolicy::threadpool || policy == AsyncPolicy::async ) {
         // everything offloaded to a separate thread
         auto worker = [ = ]()
         {
            // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
            if constexpr( std::is_same< Device, Devices::Cuda >::value )
               cudaSetDevice( this->gpu_id );

            this->synchronizeByteArray( array, bytesPerValue );
         };

         if( policy == AsyncPolicy::threadpool )
            async_op = tp.post( worker );
         else
            async_op = std::async( std::launch::async, worker );
      }
      else if( policy == AsyncPolicy::deferred ) {
         // immediate start, deferred synchronization (but still in the same thread)
         auto requests = synchronizeByteArrayAsyncWorker( array, bytesPerValue );
         auto worker = [ requests ]() mutable
         {
            MPI::Waitall( requests.data(), requests.size() );
         };
         this->async_op = std::async( std::launch::deferred, worker );
      }
      else {
         // synchronous
         synchronizeByteArray( array, bytesPerValue );
      }

      async_ops_count++;
      async_start_timer.stop();
   }

   virtual ~ByteArraySynchronizer() = default;

   /**
    * \brief Can be used for checking if a synchronization started
    * asynchronously has been finished.
    *
    * Note that derived classes *must* make this check in the destructor,
    * otherwise running \ref synchronizeByteArrayAsync would lead to the error
    * `pure virtual method called` when the derived object is destructed before
    * the async operation finishes. This cannot be implemented in the base class
    * destructor, because the derived destructor is run first.
    *
    *    ~Derived()
    *    {
    *       if( this->async_op.valid() )
    *          this->async_op.wait();
    *    }
    */
   std::future< void > async_op;

   // attributes for profiling
   Timer async_wait_before_start_timer, async_start_timer, async_wait_timer;
   std::size_t async_ops_count = 0;
};

}  // namespace Containers
}  // namespace noa::TNL
