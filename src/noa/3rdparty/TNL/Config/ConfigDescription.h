// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <vector>
#include <memory>

#include <TNL/Config/ConfigEntry.h>
#include <TNL/Config/ConfigEntryList.h>
#include <TNL/Config/ConfigDelimiter.h>
#include <TNL/Exceptions/ConfigError.h>

namespace TNL {
namespace Config {

class ConfigDescription
{
public:
   /**
    * \brief Adds new entry to the configuration description.
    *
    * \tparam EntryType Type of the entry.
    * \param name Name of the entry.
    * \param description More specific information about the entry.
    */
   template< typename EntryType >
   void addEntry( const std::string& name,
                  const std::string& description )
   {
      using CoercedEntryType = typename ParameterTypeCoercion< EntryType >::type;
      entries.push_back( std::make_unique< ConfigEntry< CoercedEntryType > >( name, description, false ) );
      currentEntry = entries.back().get();
      isCurrentEntryList = false;
   }

   /**
    * \brief Adds new entry to the configuration description, that requires set value.
    *
    * \tparam EntryType Type of the entry.
    * \param name Name of the entry.
    * \param description More specific information about the entry.
    */
   template< typename EntryType >
   void addRequiredEntry( const std::string& name,
                          const std::string& description )
   {
      using CoercedEntryType = typename ParameterTypeCoercion< EntryType >::type;
      entries.push_back( std::make_unique< ConfigEntry< CoercedEntryType > >( name, description, true ) );
      currentEntry = entries.back().get();
      isCurrentEntryList = false;
   }

   /**
    * \brief Adds new entry to the configuration description.
    *
    * \tparam EntryType Type of the entry.
    * \param name Name of the entry.
    * \param description More specific information about the entry.
    * \param defaultValue Default value of the entry.
    */
   template< typename EntryType >
   void addEntry( const std::string& name,
                  const std::string& description,
                  const EntryType& defaultValue )
   {
      using CoercedEntryType = typename ParameterTypeCoercion< EntryType >::type;
      const auto convertedDefaultValue = ParameterTypeCoercion< EntryType >::convert( defaultValue );
      entries.push_back( std::make_unique< ConfigEntry< CoercedEntryType > >( name, description, false, convertedDefaultValue ) );
      currentEntry = entries.back().get();
      isCurrentEntryList = false;
   }

   /**
    * \brief Adds new list to the configuration description.
    *
    * \tparam EntryType Type of the list.
    * \param name Name of the list.
    * \param description More specific information about the list.
    */
   template< typename EntryType >
   void addList( const std::string& name,
                 const std::string& description )
   {
      using CoercedEntryType = typename ParameterTypeCoercion< EntryType >::type;
      entries.push_back( std::make_unique< ConfigEntryList< CoercedEntryType > >( name, description, false ) );
      currentEntry = entries.back().get();
      isCurrentEntryList = true;
   }

   /**
    * \brief Adds new list to the configuration description, that requires specific value.
    *
    * \tparam EntryType Type of the list.
    * \param name Name of the list.
    * \param description More specific information about the list.
    */
   template< typename EntryType >
   void addRequiredList( const std::string& name,
                         const std::string& description )
   {
      using CoercedEntryType = typename ParameterTypeCoercion< EntryType >::type;
      entries.push_back( std::make_unique< ConfigEntryList< CoercedEntryType > >( name, description, true ) );
      currentEntry = entries.back().get();
      isCurrentEntryList = true;
   }

   /**
    * \brief Adds new list to the configuration description.
    *
    * \tparam EntryType Type of the list.
    * \param name Name of the list.
    * \param description More specific information about the list.
    * \param defaultValue Default value of the list.
    */
   template< typename EntryType >
   void addList( const std::string& name,
                 const std::string& description,
                 const std::vector< EntryType >& defaultValue )
   {
      using CoercedEntryType = typename ParameterTypeCoercion< EntryType >::type;
      const auto convertedDefaultValue = ParameterTypeCoercion< std::vector< EntryType > >::convert( defaultValue );
      entries.push_back( std::make_unique< ConfigEntryList< CoercedEntryType > >( name, description, false, convertedDefaultValue ) );
      currentEntry = entries.back().get();
      isCurrentEntryList = true;
   }

   /**
    * \brief Adds new entry enumeration of type \e EntryType.
    *
    * Adds new option of setting an entry value.
    * \tparam EntryType Type of the entry enumeration.
    * \param entryEnum Value of the entry enumeration.
    */
   template< typename EntryType = std::string >
   void addEntryEnum( const EntryType entryEnum = EntryType{} )
   {
      if( this->currentEntry == nullptr )
         throw Exceptions::ConfigError( "there is no current entry" );

      using CoercedEntryType = typename ParameterTypeCoercion< EntryType >::type;
      if( isCurrentEntryList ) {
         ConfigEntryList< CoercedEntryType >& entry = dynamic_cast< ConfigEntryList< CoercedEntryType >& >( *currentEntry );
         entry.getEnumValues().push_back( ParameterTypeCoercion< EntryType >::convert( entryEnum ) );
      }
      else {
         ConfigEntry< CoercedEntryType >& entry = dynamic_cast< ConfigEntry< CoercedEntryType >& >( *currentEntry );
         entry.getEnumValues().push_back( ParameterTypeCoercion< EntryType >::convert( entryEnum ) );
      }
   }

   /**
    * \brief Adds delimeter/section to the configuration description.
    *
    * \param delimeter String that defines how the delimeter looks like.
    */
   void addDelimiter( const std::string& delimiter )
   {
      entries.push_back( std::make_unique< ConfigDelimiter >( delimiter ) );
      currentEntry = nullptr;
   }

   /**
    * \brief Gets entry out of the configuration description.
    *
    * \param name Name of the entry.
    */
   const ConfigEntryBase* getEntry( const std::string& name ) const
   {
      // ConfigDelimiter has empty name
      if( name.empty() )
         return nullptr;

      const int entries_num = entries.size();
      for( int i = 0; i < entries_num; i++ )
         if( entries[ i ]->getName() == name )
            return entries[ i ].get();
      return nullptr;
   }

   // iterators
   auto begin() noexcept { return entries.begin(); }
   auto begin() const noexcept { return entries.begin(); }
   auto cbegin() const noexcept { return entries.cbegin(); }
   auto end() noexcept { return entries.end(); }
   auto end() const noexcept { return entries.end(); }
   auto cend() const noexcept { return entries.cend(); }

protected:
   std::vector< std::unique_ptr< ConfigEntryBase > > entries;
   ConfigEntryBase* currentEntry = nullptr;
   bool isCurrentEntryList = false;
};

} // namespace Config
} // namespace TNL
