// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

//! \file CudaCallable.h

// The __cuda_callable__ macro has to be in a separate header file to avoid
// infinite loops by the #include directives.

#ifdef __CUDACC__
   /**
    * This macro serves for annotating functions which are supposed to be called
    * even from the GPU device. If __CUDACC__ is defined, functions annotated
    * with `__cuda_callable__` are compiled for both CPU and GPU. If __CUDACC__
    * is not defined, this macro has no effect.
    */
   // clang-format off
   #define __cuda_callable__  __device__ __host__
// clang-format on
#else
   #define __cuda_callable__
#endif
