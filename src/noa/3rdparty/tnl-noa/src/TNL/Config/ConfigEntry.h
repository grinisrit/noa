// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <utility>
#include <vector>
#include <sstream>

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigEntryBase.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigEntryType.h>

namespace noa::TNL {
namespace Config {

template< typename EntryType, typename DefaultValueType = EntryType >
class ConfigEntry : public ConfigEntryBase
{
   static_assert( std::is_same< EntryType, DefaultValueType >::value
                     || std::is_same< std::vector< EntryType >, DefaultValueType >::value,
                  "DefaultValueType must be the same as either EntryType or std::vector< EntryType >" );

   DefaultValueType defaultValue;

   std::vector< EntryType > enumValues;

public:
   ConfigEntry( const std::string& name, const std::string& description, bool required )
   : ConfigEntryBase( name, description, required )
   {
      _hasDefaultValue = false;
   }

   ConfigEntry( const std::string& name, const std::string& description, bool required, DefaultValueType defaultValue )
   : ConfigEntryBase( name, description, required ), defaultValue( std::move( defaultValue ) )
   {
      _hasDefaultValue = true;
   }

   std::string
   getUIEntryType() const override
   {
      return Config::getUIEntryType< DefaultValueType >();
   }

   std::string
   printDefaultValue() const override
   {
      // printDefaultValue must be compilable even if DefaultValueType is std::vector,
      // so we can't override the method in ConfigEntryList
      return _print_value( defaultValue );
   }

   bool
   hasEnumValues() const override
   {
      return ! enumValues.empty();
   }

   void
   printEnumValues( std::ostream& str ) const override
   {
      str << "- Can be:           ";
      int i;
      for( i = 0; i < (int) enumValues.size() - 1; i++ )
         str << enumValues[ i ] << ", ";
      str << enumValues[ i ];
      str << " ";
   }

   virtual DefaultValueType
   getDefaultValue() const
   {
      return defaultValue;
   }

   virtual std::vector< EntryType >&
   getEnumValues()
   {
      return enumValues;
   }

   virtual const std::vector< EntryType >&
   getEnumValues() const
   {
      return enumValues;
   }

private:
   static std::string
   _print_value( const EntryType& value )
   {
      std::stringstream str;
      str << value;
      return str.str();
   }

   static std::string
   _print_value( const std::vector< EntryType >& vec )
   {
      std::stringstream str;
      str << "[ ";
      for( std::size_t i = 0; i < vec.size() - 1; i++ )
         str << vec[ i ] << ", ";
      str << vec[ vec.size() - 1 ];
      str << " ]";
      return str.str();
   }
};

}  // namespace Config
}  // namespace noa::TNL
