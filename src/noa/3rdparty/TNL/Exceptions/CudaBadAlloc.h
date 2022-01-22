// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <new>

namespace noaTNL {
namespace Exceptions {

struct CudaBadAlloc
   : public std::bad_alloc
{
   CudaBadAlloc()
   {
#ifdef HAVE_CUDA
      // Make sure to clear the CUDA error, otherwise the exception handler
      // might throw another exception with the same error.
      cudaGetLastError();
#endif
   }

   const char* what() const throw()
   {
      return "Failed to allocate memory on the CUDA device: "
             "most likely there is not enough space on the device memory.";
   }
};

} // namespace Exceptions
} // namespace noaTNL
