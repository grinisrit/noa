// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <string>
#include <sstream>
#include <iomanip>

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

namespace noa::TNL {
namespace Config {

/**
 * \brief Fills in the parameters from the \e parameters.
 *
 * Parameters which were not defined in the command line by user but have their default value are added to the congiguration
 * description. If there is missing entry with defined default value in the Config::ParameterContainer it is going to be added.
 * \param parameters Name of the ParameterContainer object.
 */
inline void
addDefaultValues( const ConfigDescription& config, ParameterContainer& parameters )
{
   for( const auto& entryBase : config ) {
      const std::string entry_name = entryBase->getName();
      if( entryBase->hasDefaultValue() && ! parameters.checkParameter( entry_name ) ) {
         if( entryBase->getUIEntryType() == "bool" ) {
            auto& entry = dynamic_cast< ConfigEntry< bool >& >( *entryBase );
            parameters.addParameter< bool >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "integer" ) {
            auto& entry = dynamic_cast< ConfigEntry< Integer >& >( *entryBase );
            parameters.addParameter< Integer >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "unsigned integer" ) {
            auto& entry = dynamic_cast< ConfigEntry< UnsignedInteger >& >( *entryBase );
            parameters.addParameter< UnsignedInteger >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "real" ) {
            auto& entry = dynamic_cast< ConfigEntry< double >& >( *entryBase );
            parameters.addParameter< double >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "string" ) {
            auto& entry = dynamic_cast< ConfigEntry< std::string >& >( *entryBase );
            parameters.addParameter< std::string >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "list of bool" ) {
            auto& entry = dynamic_cast< ConfigEntryList< bool >& >( *entryBase );
            parameters.addList< bool >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "list of integer" ) {
            auto& entry = dynamic_cast< ConfigEntryList< Integer >& >( *entryBase );
            parameters.addList< Integer >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "list of unsigned integer" ) {
            auto& entry = dynamic_cast< ConfigEntryList< UnsignedInteger >& >( *entryBase );
            parameters.addList< UnsignedInteger >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "list of real" ) {
            auto& entry = dynamic_cast< ConfigEntryList< double >& >( *entryBase );
            parameters.addList< double >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else if( entryBase->getUIEntryType() == "list of string" ) {
            auto& entry = dynamic_cast< ConfigEntryList< std::string >& >( *entryBase );
            parameters.addList< std::string >( entry_name, entry.getDefaultValue() );
            continue;
         }
         else
            throw std::runtime_error( "Function addDefaultValues encountered unsupported entry type: "
                                      + entryBase->getUIEntryType() );
      }
   }
}

/**
 * \brief Prints configuration description with the \e program_name at the top.
 *
 * \param program_name Name of the program
 */
inline void
printUsage( const ConfigDescription& config, const char* program_name )
{
   std::cout << "Usage of: " << program_name << std::endl << std::endl;
   std::size_t max_name_length = 0;
   std::size_t max_type_length = 0;

   for( const auto& entry : config ) {
      if( ! entry->isDelimiter() ) {
         max_name_length = std::max( max_name_length, entry->getName().length() );
         max_type_length = std::max( max_type_length, entry->getUIEntryType().length() );
      }
   }
   max_name_length += 2;  // this is for "--"

   for( const auto& entry : config ) {
      if( entry->isDelimiter() ) {
         std::cout << std::endl;
         std::cout << entry->getDescription();
         std::cout << std::endl << std::endl;
      }
      else {
         std::cout << std::setw( max_name_length + 3 ) << "--" << entry->getName() << std::setw( max_type_length + 5 )
                   << entry->getUIEntryType() << "    " << entry->getDescription();
         if( entry->isRequired() )
            std::cout << " *** REQUIRED ***";
         if( entry->hasEnumValues() ) {
            std::cout << std::endl
                      << std::setw( max_name_length + 3 ) << "" << std::setw( max_type_length + 5 ) << ""
                      << "    ";
            entry->printEnumValues( std::cout );
         }
         if( entry->hasDefaultValue() ) {
            std::cout << std::endl
                      << std::setw( max_name_length + 3 ) << "" << std::setw( max_type_length + 5 ) << ""
                      << "    ";
            std::cout << "- Default value is: " << entry->printDefaultValue();
         }
         std::cout << std::endl;
      }
   }
   std::cout << std::endl;
}

//! Check for all entries with the flag 'required'.
/*! Returns false if any parameter is missing.
 */
inline bool
checkMissingEntries( const ConfigDescription& config,
                     Config::ParameterContainer& parameters,
                     bool printUsage,
                     const char* programName )
{
   std::vector< std::string > missingParameters;
   for( const auto& entryBase : config ) {
      const std::string entry_name = entryBase->getName();
      if( entryBase->isRequired() && ! parameters.checkParameter( entry_name ) )
         missingParameters.push_back( entry_name );
   }
   if( ! missingParameters.empty() ) {
      std::cerr << "Some mandatory parameters are misssing. They are listed at the end." << std::endl;
      if( printUsage )
         Config::printUsage( config, programName );
      std::cerr << "Add the following missing parameters to the command line:" << std::endl << "   ";
      for( auto& missingParameter : missingParameters )
         std::cerr << "--" << missingParameter << " ... ";
      std::cerr << std::endl;
      return false;
   }
   return true;
}

template< typename EntryType, typename DefaultValueType >
void
checkEnumValues( const ConfigEntry< EntryType, DefaultValueType >& entry, const std::string& name, const EntryType& value )
{
   if( entry.getEnumValues().size() > 0 ) {
      for( const auto& enumValue : entry.getEnumValues() )
         if( value == enumValue )
            return;
      std::stringstream str;
      str << "The value " << value << " is not allowed for the config entry " << name << ".\n";
      entry.printEnumValues( str );
      throw Exceptions::ConfigError( str.str() );
   }
}

template< typename EntryType, typename DefaultValueType >
void
checkEnumValues( const ConfigEntry< EntryType, DefaultValueType >& entry,
                 const std::string& name,
                 const std::vector< EntryType >& values )
{
   if( entry.getEnumValues().size() > 0 ) {
      for( auto value : values )
         checkEnumValues( entry, name, value );
   }
}

template< typename T >
T
convertStringValue( const std::string& value, const std::string& param )
{
   T v;
   std::stringstream str;
   str << value;
   str >> v;
   if( str.fail() )
      throw Exceptions::ConfigError( "Value '" + value + "' could not be converted to type " + TNL::getType< T >()
                                     + " as required for the parameter " + param + "." );
   return v;
}

inline bool
parseCommandLine( int argc,
                  char* argv[],
                  const Config::ConfigDescription& config_description,
                  Config::ParameterContainer& parameters,
                  bool printUsage = true )
{
   auto iequals = []( const std::string& a, const std::string& b )
   {
      if( a.size() != b.size() )
         return false;
      for( unsigned int i = 0; i < a.size(); i++ )
         if( std::tolower( a[ i ] ) != std::tolower( b[ i ] ) )
            return false;
      return true;
   };

   auto matob = [ iequals ]( const std::string& value, const std::string& option ) -> bool
   {
      if( iequals( value, "yes" ) || iequals( value, "true" ) )
         return true;
      if( iequals( value, "no" ) || iequals( value, "false" ) )
         return false;
      throw Exceptions::ConfigError( "Yes/true or no/false is required for the parameter " + option + "." );
   };

   int i;
   try {
      for( i = 1; i < argc; i++ ) {
         const std::string _option = argv[ i ];
         if( _option.length() < 3 || _option[ 0 ] != '-' || _option[ 1 ] != '-' )
            throw Exceptions::ConfigError( "Unknown option " + _option + ". Options must be prefixed with \"--\"." );
         if( _option == "--help" ) {
            Config::printUsage( config_description, argv[ 0 ] );
            // FIXME: false here indicates that the program should terminate, but the exit code should indicate success instead
            // of failure
            return false;
         }
         const std::string option = _option.substr( 2 );
         const ConfigEntryBase* entryBase = config_description.getEntry( option );
         if( entryBase == nullptr )
            throw Exceptions::ConfigError( "Unknown parameter " + _option + "." );
         const String entryType = entryBase->getUIEntryType();
         if( i == argc - 1 )
            throw Exceptions::ConfigError( "Missing value for the parameter " + option + "." );
         const std::string value = argv[ ++i ];
         if( value.empty() )
            throw Exceptions::ConfigError( "Missing value for the parameter " + option + "." );
         const std::vector< String > parsedEntryType = entryType.split();
         if( parsedEntryType.empty() )
            throw Exceptions::ConfigError( "Internal error: Unknown config entry type " + entryType + "." );
         if( parsedEntryType[ 0 ] == "list" ) {
            std::vector< bool > bool_list;
            std::vector< Integer > integer_list;
            std::vector< UnsignedInteger > unsigned_integer_list;
            std::vector< double > real_list;
            std::vector< std::string > string_list;

            while( i < argc && std::string( argv[ i ] ).substr( 0, 2 ) != "--" ) {
               const std::string value = argv[ i++ ];
               if( entryType == "list of bool" ) {
                  const bool v = matob( value, option );
                  bool_list.push_back( v );
               }
               else if( entryType == "list of integer" ) {
                  const Integer v = convertStringValue< Integer >( value, option );
                  const auto& entry = dynamic_cast< const ConfigEntryList< Integer >& >( *entryBase );
                  checkEnumValues( entry, option, v );
                  integer_list.push_back( v );
               }
               else if( entryType == "list of unsigned integer" ) {
                  const auto v = convertStringValue< UnsignedInteger >( value, option );
                  const auto& entry = dynamic_cast< const ConfigEntryList< UnsignedInteger >& >( *entryBase );
                  checkEnumValues( entry, option, v );
                  unsigned_integer_list.push_back( v );
               }
               else if( entryType == "list of real" ) {
                  const double v = convertStringValue< double >( value, option );
                  const auto& entry = dynamic_cast< const ConfigEntryList< double >& >( *entryBase );
                  checkEnumValues( entry, option, v );
                  real_list.push_back( v );
               }
               else if( entryType == "list of string" ) {
                  const auto& entry = dynamic_cast< const ConfigEntryList< std::string >& >( *entryBase );
                  checkEnumValues( entry, option, value );
                  string_list.push_back( value );
               }
               else
                  // this will not happen if all entry types are handled above
                  throw std::runtime_error( "Function parseCommandLine encountered unsupported entry type: " + entryType );
            }
            if( ! bool_list.empty() )
               parameters.addParameter< std::vector< bool > >( option, bool_list );
            if( ! integer_list.empty() )
               parameters.addParameter< std::vector< Integer > >( option, integer_list );
            if( ! unsigned_integer_list.empty() )
               parameters.addParameter< std::vector< UnsignedInteger > >( option, unsigned_integer_list );
            if( ! real_list.empty() )
               parameters.addParameter< std::vector< double > >( option, real_list );
            if( ! string_list.empty() )
               parameters.addParameter< std::vector< std::string > >( option, string_list );
         }
         else {
            if( entryType == "bool" ) {
               const bool v = matob( value, option );
               parameters.addParameter< bool >( option, v );
            }
            else if( entryType == "integer" ) {
               const Integer v = convertStringValue< Integer >( value, option );
               const auto& entry = dynamic_cast< const ConfigEntry< Integer >& >( *entryBase );
               checkEnumValues( entry, option, v );
               parameters.addParameter< Integer >( option, v );
            }
            else if( entryType == "unsigned integer" ) {
               const auto v = convertStringValue< UnsignedInteger >( value, option );
               const auto& entry = dynamic_cast< const ConfigEntry< UnsignedInteger >& >( *entryBase );
               checkEnumValues( entry, option, v );
               parameters.addParameter< UnsignedInteger >( option, v );
            }
            else if( entryType == "real" ) {
               const double v = convertStringValue< double >( value, option );
               const auto& entry = dynamic_cast< const ConfigEntry< double >& >( *entryBase );
               checkEnumValues( entry, option, v );
               parameters.addParameter< double >( option, v );
            }
            else if( entryType == "string" ) {
               const auto& entry = dynamic_cast< const ConfigEntry< std::string >& >( *entryBase );
               checkEnumValues( entry, option, (std::string) value );
               parameters.addParameter< std::string >( option, value );
            }
            else
               // this will not happen if all entry types are handled above
               throw std::runtime_error( "Function parseCommandLine encountered unsupported entry type: " + entryType );
         }
      }
   }
   catch( const Exceptions::ConfigError& e ) {
      std::cerr << argv[ 0 ] << ": failed to parse the command line due to the following error:\n" << e.what() << std::endl;
      if( printUsage )
         Config::printUsage( config_description, argv[ 0 ] );
      return false;
   }
   addDefaultValues( config_description, parameters );
   return checkMissingEntries( config_description, parameters, printUsage, argv[ 0 ] );
}

}  // namespace Config
}  // namespace noa::TNL
