// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <map>

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/Stream.h>

namespace noa::TNL {
namespace Cuda {

class StreamPool
{
public:
   static StreamPool&
   getInstance()
   {
      // note that this ensures construction on first use, and thus also correct
      // destruction before the CUDA context is destroyed
      // https://stackoverflow.com/questions/335369/finding-c-static-initialization-order-problems#335746
      static StreamPool instance;
      return instance;
   }

   /**
    * \brief Get a stream from the pool.
    *
    * \param id Numeric ID of the requested stream.
    *
    * If the stream with given ID was not created yet, it is created by calling
    * \ref Cuda::Stream::create with the following parameters:
    *
    * \param flags Custom flags for stream creation.
    *              See \ref Cuda::Stream::create for details.
    * \param priority Priority of the stream.
    *                 See \ref Cuda::Stream::create for details.
    */
   const cudaStream_t&
   getStream( int id, unsigned int flags = cudaStreamDefault, int priority = 0 )
   {
      const auto& result = pool.find( id );
      if( result != pool.end() )
         return result->second;
      return pool.emplace( id, Stream::create( flags, priority ) ).first->second;
   }

   // copy-constructor and copy-assignment are meaningless for a singleton class
   StreamPool( StreamPool const& copy ) = delete;
   StreamPool&
   operator=( StreamPool const& copy ) = delete;

private:
   // private constructor of the singleton
   StreamPool() = default;

   using MapType = std::map< int, Stream >;

   MapType pool;
};

}  // namespace Cuda
}  // namespace noa::TNL
