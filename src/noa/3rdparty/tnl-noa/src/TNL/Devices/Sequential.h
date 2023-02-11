// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
//! \brief Namespace for TNL execution models
namespace Devices {

struct Sequential
{
   //! Not used by any sequential algorithm, only for compatibility with parallel execution models.
   struct LaunchConfiguration
   {};
};

}  // namespace Devices
}  // namespace noa::TNL
