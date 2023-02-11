// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaRuntimeError.h>

namespace noa::TNL {
namespace Cuda {

#ifdef __CUDACC__
/****
 * I do not know why, but it is more reliable to pass the error code instead
 * of calling cudaGetLastError() inside the function.
 * We recommend to use macro 'TNL_CHECK_CUDA_DEVICE' defined bellow.
 */
inline void
checkDevice( const char* file_name, int line, cudaError error )
{
   if( error != cudaSuccess )
      throw Exceptions::CudaRuntimeError( error, file_name, line );
}
#else
inline void
checkDevice()
{}
#endif

}  // namespace Cuda
}  // namespace noa::TNL

#ifdef __CUDACC__
   #define TNL_CHECK_CUDA_DEVICE ::noa::TNL::Cuda::checkDevice( __FILE__, __LINE__, cudaGetLastError() )
#else
   #define TNL_CHECK_CUDA_DEVICE ::noa::TNL::Cuda::checkDevice()
#endif
