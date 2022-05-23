// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>

namespace noa::TNL {
namespace Meshes {

template< typename Real = double, typename Device = Devices::Host, typename Index = int >
class DummyMesh
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;

   constexpr static int
   getMeshDimension()
   {
      return 1;
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
