// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Host.h>

namespace TNL {
namespace Devices {

class AnyDevice
{
};

template< typename Device >
struct PickDevice
{
   using DeviceType = Device;
};

template<>
struct PickDevice< Devices::AnyDevice >
{
   using DeviceType = Devices::Host;
};

} // namespace Devices
} // namespace TNL
