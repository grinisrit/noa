// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <string>

namespace noa::TNL {
namespace Config {

class ConfigEntryBase
{
protected:
   std::string name;

   std::string description;

   bool required;

   bool _hasDefaultValue;

public:
   ConfigEntryBase( const std::string& name,
                    const std::string& description,
                    bool required )
      : name( name ),
        description( description ),
        required( required ),
        _hasDefaultValue( false )
   {}

   const std::string& getName() const { return name; }

   const std::string& getDescription() const { return description; }

   bool isRequired() const { return required; }

   bool hasDefaultValue() const { return _hasDefaultValue; }

   virtual std::string getUIEntryType() const = 0;

   virtual bool isDelimiter() const { return false; }

   virtual std::string printDefaultValue() const { return ""; }

   virtual bool hasEnumValues() const { return false; }

   virtual void printEnumValues( std::ostream& str ) const {}

   virtual ~ConfigEntryBase() = default;
};

} // namespace Config
} // namespace noa::TNL
