// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <string>
#include <stdexcept>

namespace noa::TNL {
namespace Exceptions {

class FileDeserializationError : public std::runtime_error
{
public:
   FileDeserializationError( const std::string& fileName, const std::string& details )
   : std::runtime_error( "Failed to deserialize an object from the file '" + fileName + "': " + details )
   {}
};

}  // namespace Exceptions
}  // namespace noa::TNL
