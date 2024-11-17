// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Problems {

template< typename Real, typename Device, typename Index >
class Problem
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
};

}  // namespace Problems
}  // namespace noa::TNL
