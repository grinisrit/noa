// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <set>

#include <noa/3rdparty/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Config/parseCommandLine.h>    // for addDefaultValues and checkEnumValues

#include <Leksys/iniparser.hpp>

namespace noa::TNL {
namespace Config {

ParameterContainer
parseINIConfigFile( const std::string& configPath,
                    const ConfigDescription& description )
{
    ParameterContainer parameters;

    INI::File file;
    if( ! file.Load( configPath ) )
        throw std::runtime_error( file.LastResult().GetErrorDesc() );

    // aggregated sets
    std::set< std::string > undefinedOptions;

    // iterate over all entries in the global section
    const INI::Section* section = file.GetSection( "" );
    for( auto e = section->ValuesBegin(); e != section->ValuesEnd(); e++ ) {
        const std::string name = e->first;
        const auto* entryBase = description.getEntry( name );
        if( entryBase == nullptr ) {
            undefinedOptions.insert( name );
            continue;
        }
        const std::string entryType = entryBase->getUIEntryType();
        const auto value = e->second;
        if( value.AsString().empty() )
            throw std::runtime_error( "Missing value for the parameter " + name + "." );

        if( entryType == "bool" ) {
            parameters.addParameter< bool >( name, value.AsBool() );
            continue;
        }
        else if( entryType == "integer" ) {
            const auto v = value.AsT< Integer >();
            const ConfigEntry< Integer >& entry = dynamic_cast< const ConfigEntry< Integer >& >( *entryBase );
            checkEnumValues( entry, name, v );
            parameters.addParameter< Integer >( name, v );
        }
        else if( entryType == "unsigned integer" ) {
            const auto v = value.AsT< UnsignedInteger >();
            const ConfigEntry< UnsignedInteger >& entry = dynamic_cast< const ConfigEntry< UnsignedInteger >& >( *entryBase );
            checkEnumValues( entry, name, v );
            parameters.addParameter< UnsignedInteger >( name, v );
        }
        else if( entryType == "real" ) {
            const auto v = value.AsDouble();
            const ConfigEntry< double >& entry = dynamic_cast< const ConfigEntry< double >& >( *entryBase );
            checkEnumValues( entry, name, v );
            parameters.addParameter< double >( name, v );
        }
        else if( entryType == "string" ) {
            const auto v = value.AsString();
            const ConfigEntry< std::string >& entry = dynamic_cast< const ConfigEntry< std::string >& >( *entryBase );
            checkEnumValues( entry, name, v );
            parameters.addParameter< std::string >( name, v );
            continue;
        }
        else if( entryType == "list of bool" ) {
            const auto list = value.AsArray().ToVector< bool >();
            parameters.addList< bool >( name, list );
        }
        else if( entryType == "list of integer" ) {
            const auto list = value.AsArray().ToVector< Integer >();
            const ConfigEntry< Integer >& entry = dynamic_cast< const ConfigEntry< Integer >& >( *entryBase );
            checkEnumValues( entry, name, list );
            parameters.addList< Integer >( name, list );
        }
        else if( entryType == "list of unsigned integer" ) {
            const auto list = value.AsArray().ToVector< UnsignedInteger >();
            const ConfigEntry< UnsignedInteger >& entry = dynamic_cast< const ConfigEntry< UnsignedInteger >& >( *entryBase );
            checkEnumValues( entry, name, list );
            parameters.addList< UnsignedInteger >( name, list );
        }
        else if( entryType == "list of real" ) {
            const auto list = value.AsArray().ToVector< double >();
            const ConfigEntry< double >& entry = dynamic_cast< const ConfigEntry< double >& >( *entryBase );
            checkEnumValues( entry, name, list );
            parameters.addList< double >( name, list );
        }
        else if( entryType == "list of string" ) {
            const auto list = value.AsArray().ToVector< std::string >();
            const ConfigEntry< std::string >& entry = dynamic_cast< const ConfigEntry< std::string >& >( *entryBase );
            checkEnumValues( entry, name, list );
            parameters.addList< std::string >( name, list );
        }
        else
            // this will not happen if all entry types are handled above
            throw std::runtime_error( "Function parseINIConfigFil encountered unsupported entry type: " + entryType );
    }

    if( ! undefinedOptions.empty() ) {
        std::string msg = "The configuration file contains the following options which are not defined in the program:\n";
        for( auto option : undefinedOptions )
            msg += " - " + option + "\n";
        throw std::runtime_error( msg );
    }

    // add default values to the parameters
    addDefaultValues( description, parameters );

    return parameters;
}

} // namespace Config
} // namespace noa::TNL
