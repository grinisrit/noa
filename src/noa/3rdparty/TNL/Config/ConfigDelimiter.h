// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Config/ConfigEntryBase.h>

namespace noa::TNL {
namespace Config {

class ConfigDelimiter : public ConfigEntryBase
{
public:
   ConfigDelimiter( const std::string& delimiter )
   : ConfigEntryBase( "", delimiter, false )
   {}

   virtual bool isDelimiter() const override { return true; };

   virtual std::string getUIEntryType() const override { return ""; };
};

} //namespace Config
} //namespace noa::TNL
