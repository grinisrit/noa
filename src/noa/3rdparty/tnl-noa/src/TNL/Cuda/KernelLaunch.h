// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CheckDevice.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/DummyDefs.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/CudaSupportMissing.h>
#include <noa/3rdparty/tnl-noa/src/TNL/TypeInfo.h>

namespace noa::TNL {
namespace Cuda {

/**
 * Holds the parameters necessary to "launch" a CUDA kernel (i.e. schedule it for
 * execution on some stream of some device).
 */
struct LaunchConfiguration
{
   // kernel grid dimensions (in blocks)
   dim3 gridSize;

   // kernel block dimensions (in threads)
   dim3 blockSize;

   // size of dynamic shared memory (in bytes per block)
   std::size_t dynamicSharedMemorySize = 0U;

   // stream handle
   cudaStream_t stream = 0;

   // indicates whether host execution is blocked until the CUDA kernel execution is finished
   bool blockHostUntilFinished = true;

   LaunchConfiguration() = default;
   constexpr LaunchConfiguration( const LaunchConfiguration& ) = default;
   constexpr LaunchConfiguration( LaunchConfiguration&& ) = default;

   constexpr LaunchConfiguration( dim3 gridSize,
                                  dim3 blockSize,
                                  std::size_t dynamicSharedMemorySize = 0U,
                                  cudaStream_t stream = 0,
                                  bool blockHostUntilFinished = true )
   : gridSize( gridSize ), blockSize( blockSize ), dynamicSharedMemorySize( dynamicSharedMemorySize ), stream( stream ),
     blockHostUntilFinished( blockHostUntilFinished )
   {}
};

template< typename RawKernel, typename... KernelParameters >
inline void
launchKernel( RawKernel kernel_function, LaunchConfiguration launch_configuration, KernelParameters&&... parameters )
{
   static_assert(
      ::std::is_function< RawKernel >::value
         || ( ::std::is_pointer< RawKernel >::value && ::std::is_function< ::std::remove_pointer_t< RawKernel > >::value ),
      "Only a plain function or function pointer can be launched as a CUDA kernel. "
      "You are attempting to launch something else." );

   if( kernel_function == nullptr )
      throw std::logic_error( "cannot call a function via nullptr" );

      // TODO: basic verification of the configuration

#ifdef TNL_DEBUG_KERNEL_LAUNCHES
   // clang-format off
   std::cout << "Type of kernel function: " << TNL::getType( kernel_function ) << "\n";
   std::cout << "Kernel launch configuration:\n"
             << "\t- grid size: " << launch_configuration.gridSize.x << " x "
                                  << launch_configuration.gridSize.y << " x "
                                  << launch_configuration.gridSize.z << "\n"
             << "\t- block size: " << launch_configuration.blockSize.x << " x "
                                   << launch_configuration.blockSize.y << " x "
                                   << launch_configuration.blockSize.z
             << "\n"
             << "\t- stream: " << launch_configuration.stream << "\n"
             << "\t- dynamic shared memory size: " << launch_configuration.dynamicSharedMemorySize << "\n";
   std::cout.flush();
   // clang-format on
#endif

#ifdef __CUDACC__
   // FIXME: clang-format 13.0.0 is still inserting spaces between "<<<" and ">>>":
   // https://github.com/llvm/llvm-project/issues/52881
   // clang-format off
   kernel_function <<<
         launch_configuration.gridSize,
         launch_configuration.blockSize,
         launch_configuration.dynamicSharedMemorySize,
         launch_configuration.stream
      >>>( ::std::forward< KernelParameters >( parameters )... );
   // clang-format on

   if( launch_configuration.blockHostUntilFinished )
      cudaStreamSynchronize( launch_configuration.stream );

   // use custom error handling instead of TNL_CHECK_CUDA_DEVICE
   // to add the kernel function type to the error message
   const cudaError_t status = cudaGetLastError();
   if( status != cudaSuccess ) {
      std::string msg = "detected after launching kernel " + TNL::getType( kernel_function ) + "\nSource: line "
                      + std::to_string( __LINE__ ) + " in " + __FILE__;
      throw Exceptions::CudaRuntimeError( status, msg );
   }
#else
   throw Exceptions::CudaSupportMissing();
#endif
}

template< typename RawKernel, typename... KernelParameters >
inline void
launchKernelSync( RawKernel kernel_function, LaunchConfiguration launch_configuration, KernelParameters&&... parameters )
{
   launch_configuration.blockHostUntilFinished = true;
   launchKernel( kernel_function, launch_configuration, std::forward< KernelParameters >( parameters )... );
}

template< typename RawKernel, typename... KernelParameters >
inline void
launchKernelAsync( RawKernel kernel_function, LaunchConfiguration launch_configuration, KernelParameters&&... parameters )
{
   launch_configuration.blockHostUntilFinished = false;
   launchKernel( kernel_function, launch_configuration, std::forward< KernelParameters >( parameters )... );
}

}  // namespace Cuda
}  // namespace noa::TNL
