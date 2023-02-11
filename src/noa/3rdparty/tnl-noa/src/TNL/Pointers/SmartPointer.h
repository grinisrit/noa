// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {

/**
 * \brief Namespace for TNL pointers.
 *
 * Pointers in TNL are similar to STL pointers but they work across different device.
 */
namespace Pointers {

class SmartPointer
{
public:
   virtual bool
   synchronize() = 0;

   virtual ~SmartPointer() = default;
};

}  // namespace Pointers
}  // namespace noa::TNL
