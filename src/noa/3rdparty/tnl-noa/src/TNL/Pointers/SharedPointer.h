// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <cstring>
#include <type_traits>
#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/TypeInfo.h>

//#define TNL_DEBUG_SHARED_POINTERS

namespace noa::TNL {
namespace Pointers {

/**
 * \brief Cross-device shared smart pointer.
 *
 * This smart pointer is inspired by std::shared_ptr from STL library. It means
 * that the object owned by the smart pointer can be shared with other
 * smart pointers. One can make a copy of this smart pointer. In addition,
 * the smart pointer is able to work across different devices which means that the
 * object owned by the smart pointer is mirrored on both host and device.
 *
 * **NOTE: When using smart pointers to pass objects on GPU, one must call
 * \ref Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >()
 * before calling a CUDA kernel working with smart pointers.**
 *
 * \tparam Object is a type of object to be owned by the pointer.
 * \tparam Device is device where the object is to be allocated. The object is
 * always allocated on the host system as well for easier object manipulation.
 *
 * See also \ref UniquePointer and \ref DevicePointer.
 *
 * See also \ref SharedPointer< Object, Devices::Host > and \ref SharedPointer< Object, Devices::Cuda >.
 *
 * \par Example
 * \include Pointers/SharedPointerExample.cpp
 * \par Output
 * \include SharedPointerExample.out
 */
template< typename Object, typename Device = typename Object::DeviceType >
class SharedPointer
{
   static_assert( ! std::is_same< Device, void >::value,
                  "The device cannot be void. You need to specify the device explicitly in your code." );
};

}  // namespace Pointers

#ifndef NDEBUG
namespace Assert {

template< typename Object, typename Device >
struct Formatter< Pointers::SharedPointer< Object, Device > >
{
   static std::string
   printToString( const Pointers::SharedPointer< Object, Device >& value )
   {
      ::std::stringstream ss;
      ss << "(" + getType< Pointers::SharedPointer< Object, Device > >() << " > object at " << &value << ")";
      return ss.str();
   }
};

}  // namespace Assert
#endif

}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointerHost.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointerCuda.h>
