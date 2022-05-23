// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

// The __cuda_callable__ macro has to be in a separate header file to avoid
// infinite loops by the #include directives.

/***
 * This macro serves for definition of function which are supposed to be called
 * even from device. If HAVE_CUDA is defined, the __cuda_callable__ function
 * is compiled for both CPU and GPU. If HAVE_CUDA is not defined, this macro has
 * no effect.
 */
#ifdef HAVE_CUDA
   #define __cuda_callable__ \
      __device__             \
      __host__
#else
   #define __cuda_callable__
#endif
