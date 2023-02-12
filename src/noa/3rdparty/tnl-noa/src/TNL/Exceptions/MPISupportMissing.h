// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

namespace noa::TNL {
namespace Exceptions {

struct MPISupportMissing : public std::runtime_error
{
   MPISupportMissing()
   : std::runtime_error( "MPI support is missing, but the program called a function which needs it. "
                         "Please recompile the program with MPI support." )
   {}
};

}  // namespace Exceptions
}  // namespace noa::TNL
