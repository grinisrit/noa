// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Devices/Host.h>

namespace noa::TNL {
namespace Meshes {

template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class DummyMesh
{
public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   constexpr static int getMeshDimension() { return 1; }
};

} // namespace Meshes
} // namespace noa::TNL
