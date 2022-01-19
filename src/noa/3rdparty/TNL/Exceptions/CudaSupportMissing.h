// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <stdexcept>

namespace TNL {
/**
 * \brief Namespace for TNL exceptions.
 */
namespace Exceptions {

struct CudaSupportMissing
   : public std::runtime_error
{
   CudaSupportMissing()
   : std::runtime_error( "CUDA support is missing, but the program called a function which needs it. "
                         "Please recompile the program with CUDA support." )
   {}
};

} // namespace Exceptions
} // namespace TNL
