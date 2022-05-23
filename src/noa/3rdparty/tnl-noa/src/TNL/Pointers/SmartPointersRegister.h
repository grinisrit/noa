// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_set>
#include <unordered_map>

#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SmartPointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/DeviceInfo.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Sequential.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>

namespace noa::TNL {
namespace Pointers {

// Since TNL currently supports only execution on host (which does not need
// to register and synchronize smart pointers) and CUDA GPU's, the smart
// pointers register is implemented only for CUDA. If more execution types
// which need to register smart pointers are implemented in the future, this
// should beome a class template specialization.
class SmartPointersRegister
{
public:
   /**
    * Negative deviceId means that \ref Cuda::DeviceInfo::getActiveDevice will be
    * called to get the device ID.
    */
   void
   insert( SmartPointer* pointer, int deviceId = -1 )
   {
      if( deviceId < 0 )
         deviceId = Cuda::DeviceInfo::getActiveDevice();
      pointersOnDevices[ deviceId ].insert( pointer );
   }

   /**
    * Negative deviceId means that \ref Cuda::DeviceInfo::getActiveDevice will be
    * called to get the device ID.
    */
   void
   remove( SmartPointer* pointer, int deviceId = -1 )
   {
      if( deviceId < 0 )
         deviceId = Cuda::DeviceInfo::getActiveDevice();
      try {
         pointersOnDevices.at( deviceId ).erase( pointer );
      }
      catch( const std::out_of_range& ) {
         std::cerr << "Given deviceId " << deviceId << " does not have any pointers yet. "
                   << "Requested to remove pointer " << pointer << ". "
                   << "This is most likely a bug in the smart pointer." << std::endl;
         throw;
      }
   }

   /**
    * Negative deviceId means that \ref Cuda::DeviceInfo::getActiveDevice will be
    * called to get the device ID.
    */
   bool
   synchronizeDevice( int deviceId = -1 )
   {
      if( deviceId < 0 )
         deviceId = Cuda::DeviceInfo::getActiveDevice();
      try {
         const auto& set = pointersOnDevices.at( deviceId );
         for( auto&& it : set )
            ( *it ).synchronize();
         return true;
      }
      catch( const std::out_of_range& ) {
         return false;
      }
   }

protected:
   using SetType = std::unordered_set< SmartPointer* >;

   std::unordered_map< int, SetType > pointersOnDevices;
};

// TODO: Device -> Allocator (in all smart pointers)
template< typename Device >
SmartPointersRegister&
getSmartPointersRegister()
{
   static SmartPointersRegister reg;
   return reg;
}

template< typename Device >
Timer&
getSmartPointersSynchronizationTimer()
{
   static Timer timer;
   return timer;
}

/**
 * Negative deviceId means that the ID of the currently active device will be
 * determined automatically.
 */
template< typename Device >
bool
synchronizeSmartPointersOnDevice( int deviceId = -1 )
{
   // TODO: better way to skip synchronization of host-only smart pointers
   if( std::is_same< Device, Devices::Sequential >::value || std::is_same< Device, Devices::Host >::value )
      return true;

   getSmartPointersSynchronizationTimer< Device >().start();
   bool b = getSmartPointersRegister< Device >().synchronizeDevice( deviceId );
   getSmartPointersSynchronizationTimer< Device >().stop();
   return b;
}

}  // namespace Pointers
}  // namespace noa::TNL
