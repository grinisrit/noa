// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/String.h>

namespace TNL {
namespace Cuda {

struct DeviceInfo
{
   static int getNumberOfDevices();

   static int getActiveDevice();

   static String getDeviceName( int deviceNum );

   static int getArchitectureMajor( int deviceNum );

   static int getArchitectureMinor( int deviceNum );

   static int getClockRate( int deviceNum );

   static std::size_t getGlobalMemory( int deviceNum );

   static std::size_t getFreeGlobalMemory();

   static int getMemoryClockRate( int deviceNum );

   static bool getECCEnabled( int deviceNum );

   static int getCudaMultiprocessors( int deviceNum );

   static int getCudaCoresPerMultiprocessors( int deviceNum );

   static int getCudaCores( int deviceNum );

   static int getRegistersPerMultiprocessor( int deviceNum );
};

} // namespace Cuda
} // namespace TNL

#include <TNL/Cuda/DeviceInfo.hpp>
