// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

namespace noa::TNL {
namespace Devices {

class Cuda
{
public:
   static inline void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" )
   {
#ifdef HAVE_CUDA
      config.addEntry< int >( prefix + "cuda-device", "Choose CUDA device to run the computation.", 0 );
#else
      config.addEntry< int >(
         prefix + "cuda-device", "Choose CUDA device to run the computation (not supported on this system).", 0 );
#endif
   }

   static inline bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" )
   {
#ifdef HAVE_CUDA
      int cudaDevice = parameters.getParameter< int >( prefix + "cuda-device" );
      if( cudaSetDevice( cudaDevice ) != cudaSuccess ) {
         std::cerr << "I cannot activate CUDA device number " << cudaDevice << "." << std::endl;
         return false;
      }
#endif
      return true;
   }
};

}  // namespace Devices
}  // namespace noa::TNL
