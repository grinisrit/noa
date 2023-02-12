// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include <mpi.h>

   // Hypre header files
   #include <seq_mv.h>
   #include <_hypre_parcsr_mv.h>
   #include <_hypre_parcsr_ls.h>

   #ifdef HYPRE_MIXEDINT
      #error "TNL does not work with HYPRE's mixed-int support (i.e. when HYPRE_Int and HYPRE_BigInt are different types)"
   #endif
   #ifdef HYPRE_COMPLEX
      #error "TNL does not work with HYPRE's complex numbers support"
   #endif

   #if defined( HYPRE_USING_GPU ) && ! ( defined( HYPRE_USING_CUDA ) || defined( HYPRE_USING_HIP ) )
      #error "Unsupported GPU build of HYPRE! Only CUDA and HIP builds are supported."
   #endif
   #if defined( HYPRE_USING_CUDA ) && ! defined( __CUDACC__ )
      #error "__CUDACC__ is required when HYPRE is built with CUDA!"
   #endif
   #if defined( HYPRE_USING_HIP ) && ! defined( HAVE_HIP )
      #error "HAVE_HIP is required when HYPRE is built with HIP!"
   #endif

namespace noa::TNL {

/**
 * \defgroup Hypre  Wrappers for the Hypre library
 *
 * This group includes various wrapper classes for data structures and
 * algorithms implemented in the [Hypre library][hypre]. See the
 * [example][example] for how these wrappers can be used.
 *
 * [hypre]: https://github.com/hypre-space/hypre
 * [example]: https://gitlab.com/tnl-project/tnl/-/blob/main/src/Examples/Hypre/tnl-hypre-ex5.cpp
 */

/**
 * \brief A simple RAII wrapper for Hypre's initialization and finalization.
 *
 * When the object is constructed, it calls \e HYPRE_Init() and sets some
 * GPU-relevant options. The \e HYPRE_Finalize() function is called
 * automatically from the object's destructor.
 *
 * \ingroup Hypre
 */
struct Hypre
{
   //! \brief Constructor initializes Hypre by calling \e HYPRE_Init() and set default options.
   Hypre()
   {
      HYPRE_Init();

      setDefaultOptions();
   }

   //! \brief Sets the default Hypre global options (mostly GPU-relevant).
   static void
   setDefaultOptions()
   {
      // Global Hypre options, see
      // https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html#gpu-supported-options

      // Use Hypre's SpGEMM instead of vendor (cuSPARSE, rocSPARSE, etc.) for performance reasons
      HYPRE_SetSpGemmUseVendor( 0 );

      // The following options are Hypre's defaults as of version 2.24

      // Allocate Hypre objects in GPU memory (default)
      // HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);

      // Where to execute when using UVM (default)
      // HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

      // Use GPU-based random number generator (default)
      // HYPRE_SetUseGpuRand(1);
   }

   //! \brief Destructor that finalizes Hypre when the object goes out of scope.
   ~Hypre()
   {
      HYPRE_Finalize();
   }
};

//! \brief Returns the memory location used by Hypre objects.
inline constexpr HYPRE_MemoryLocation
getHypreMemoryLocation()
{
   #ifdef HYPRE_USING_GPU
   return HYPRE_MEMORY_DEVICE;
   #else
   return HYPRE_MEMORY_HOST;
   #endif
}

}  // namespace noa::TNL

// clang-format off
   #ifdef HYPRE_USING_CUDA
      #include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
      namespace noa::TNL {
         using HYPRE_Device = Devices::Cuda;
      }
   #else
      #include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>
      namespace noa::TNL {
         /**
          * \brief The \ref TNL::Devices "device" compatible with Hypre's data
          * structures.
          *
          * The type depends on how the Hypre library was configured. By
          * default, it is \ref Devices::Host. When using Hypre built with CUDA
          * support, it is \ref Devices::Cuda.
          *
          * \ingroup Hypre
          */
         using HYPRE_Device = Devices::Host;
      }
   #endif
// clang-format on

#endif  // HAVE_HYPRE
