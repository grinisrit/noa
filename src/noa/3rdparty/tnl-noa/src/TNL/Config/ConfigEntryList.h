// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigEntry.h>

namespace noa::TNL {
namespace Config {

template< typename EntryType >
class ConfigEntryList : public ConfigEntry< EntryType, std::vector< EntryType > >
{
public:
   // inherit constructors
   using ConfigEntry< EntryType, std::vector< EntryType > >::ConfigEntry;
};

}  // namespace Config
}  // namespace noa::TNL
