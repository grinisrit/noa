// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdexcept>

namespace noa::TNL {
namespace Exceptions {

struct ConfigError
   : public std::runtime_error
{
   ConfigError( std::string msg )
   : std::runtime_error( msg )
   {}
};

} // namespace Exceptions
} // namespace noa::TNL
