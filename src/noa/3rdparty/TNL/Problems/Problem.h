// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
namespace Problems {

template< typename Real,
          typename Device,
          typename Index >
class Problem
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
};

} // namespace Problems
} // namespace noaTNL
