// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <new>

namespace noa::TNL {
namespace Exceptions {

struct CudaBadAlloc : public std::bad_alloc
{
   CudaBadAlloc()
   {
#ifdef __CUDACC__
      // Make sure to clear the CUDA error, otherwise the exception handler
      // might throw another exception with the same error.
      cudaGetLastError();
#endif
   }

   const char*
   what() const noexcept override
   {
      return "Failed to allocate memory on the CUDA device: "
             "most likely there is not enough space on the device memory.";
   }
};

}  // namespace Exceptions
}  // namespace noa::TNL
