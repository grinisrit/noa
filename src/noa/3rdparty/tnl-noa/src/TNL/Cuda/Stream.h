// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/DummyDefs.h>

namespace noa::TNL {
namespace Cuda {

class Stream
{
private:
   struct Wrapper
   {
      cudaStream_t handle = 0;

      Wrapper() = default;
      Wrapper( const Wrapper& other ) = delete;
      Wrapper( Wrapper&& other ) noexcept = default;
      Wrapper&
      operator=( const Wrapper& other ) = delete;
      Wrapper&
      operator=( Wrapper&& other ) noexcept = default;

      Wrapper( cudaStream_t handle ) : handle( handle ) {}

      ~Wrapper()  // NOLINT
      {
#ifdef __CUDACC__
         // cannot free a 0 stream
         if( handle != 0 )
            cudaStreamDestroy( handle );
#endif
      }
   };

   std::shared_ptr< Wrapper > wrapper;

   //! \brief Internal constructor for the factory methods - initialization by the wrapper.
   Stream( std::shared_ptr< Wrapper >&& wrapper ) : wrapper( std::move( wrapper ) ) {}

public:
   //! \brief Constructs a stream wrapping the CUDA 0 (`NULL`) stream.
   Stream() = default;

   //! \brief Default copy-constructor.
   Stream( const Stream& other ) = default;

   //! \brief Default move-constructor.
   Stream( Stream&& other ) noexcept = default;

   //! \brief Default copy-assignment operator.
   Stream&
   operator=( const Stream& other ) = default;

   //! \brief Default move-assignment operator.
   Stream&
   operator=( Stream&& other ) noexcept = default;

   /**
    * \brief Creates a new stream.
    *
    * The stream is created by calling \e cudaStreamCreateWithPriority with the
    * following parameters:
    *
    * \param flags Custom flags for stream creation. Possible values are:
    *    - \e cudaStreamDefault: Default stream creation flag.
    *    - \e cudaStreamNonBlocking: Specifies that work running in the created
    *      stream may run concurrently with work in stream 0 (the `NULL`
    *      stream), and that the created stream should perform no implicit
    *      synchronization with stream 0.
    * \param priority Priority of the stream. Lower numbers represent higher
    *    priorities. See \e cudaDeviceGetStreamPriorityRange for more
    *    information about the meaningful stream priorities that can be passed.
    */
   static Stream
   create( unsigned int flags = cudaStreamDefault, int priority = 0 )
   {
      cudaStream_t stream;
#ifdef __CUDACC__
      cudaStreamCreateWithPriority( &stream, flags, priority );
#else
      stream = 0;
#endif
      return { std::make_shared< Wrapper >( stream ) };
   }

   /**
    * \brief Access the CUDA stream handle associated with this object.
    *
    * This routine permits the implicit conversion from \ref Stream to
    * `cudaStream_t`.
    *
    * \warning The obtained `cudaStream_t` handle becomes invalid when the
    * originating \ref Stream object is destroyed. For example, the following
    * code is invalid, because the \ref Stream object managing the lifetime of
    * the `cudaStream_t` handle is destroyed as soon as it is cast to
    * `cudaStream_t`:
    *
    * \code{.cpp}
    * const cudaStream_t stream = TNL::Cuda::Stream::create();
    * my_kernel<<< gridSize, blockSize, 0, stream >>>( args... );
    * \endcode
    */
   operator const cudaStream_t&() const
   {
      return wrapper->handle;
   }
};

}  // namespace Cuda
}  // namespace noa::TNL
