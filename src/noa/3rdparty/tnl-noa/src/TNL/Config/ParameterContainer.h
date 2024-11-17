// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <unordered_map>
#include <vector>

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ConfigEntryType.h>
#include <noa/3rdparty/tnl-noa/src/TNL/TypeInfo.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/ConfigError.h>

namespace noa::TNL {
namespace Config {

class ParameterContainer
{
public:
   /**
    * \brief Adds new parameter to the ParameterContainer.
    *
    * \param name Name of the new parameter.
    * \param value Value assigned to the parameter.
    */
   template< class T >
   void
   addParameter( const std::string& name, const T& value )
   {
      parameters.emplace( name, ParameterTypeCoercion< T >::convert( value ) );
   }

   /**
    * \brief Adds new parameter to the ParameterContainer.
    *
    * \param name Name of the new parameter.
    * \param value Value assigned to the parameter.
    */
   template< class T >
   void
   addParameter( const std::string& name, const std::vector< T >& value )
   {
      parameters.emplace( name, ParameterTypeCoercion< std::vector< T > >::convert( value ) );
   }

   /**
    * \brief Adds new list parameter to the ParameterContainer.
    *
    * \param name Name of the new parameter.
    * \param value Vector of values assigned to the parameter.
    */
   template< class T >
   void
   addList( const std::string& name, const std::vector< T >& value )
   {
      addParameter( name, value );
   }

   /**
    * \brief Adds new list parameter to the ParameterContainer.
    *
    * \param name Name of the new parameter.
    * \param value First value assigned to the parameter.
    * \param values Other values assigned to the parameter.
    */
   template< class T, class... Ts >
   void
   addList( const std::string& name, const T& value, const Ts&... values )
   {
      addParameter( name, std::vector< T >{ value, values... } );
   }

   /**
    * \brief Checks if the ParameterContainer contains a parameter specified by its name.
    *
    * \param name Name of the parameter.
    */
   bool
   checkParameter( const std::string& name ) const
   {
      return parameters.count( name ) > 0;
   }

   /**
    * \brief Checks if the ParameterContainer contains all specified parameter names.
    *
    * \param names List of the parameter names.
    */
   bool
   checkParameters( std::initializer_list< std::string > names ) const
   {
      for( const auto& name : names )
         if( ! checkParameter( name ) )
            return false;
      return true;
   }

   /**
    * \brief Checks if the parameter \e name already exists in ParameterContainer and holds a value of type \e T.
    *
    * \param name Name of the parameter.
    */
   template< class T >
   bool
   checkParameterType( const std::string& name ) const
   {
      using CoercedType = typename ParameterTypeCoercion< T >::type;
      auto search = parameters.find( name );
      if( search != parameters.end() ) {
         if( holds_alternative< CoercedType >( search->second ) )
            return true;
      }
      return false;
   }

   /**
    * \brief Assigns new \e value to the parameter \e name.
    *
    * \tparam T Type of the parameter value.
    * \param name Name of parameter.
    * \param value Value of type T assigned to the parameter.
    */
   template< class T >
   void
   setParameter( const std::string& name, const T& value )
   {
      using CoercedType = typename ParameterTypeCoercion< T >::type;
      auto search = parameters.find( name );
      if( search != parameters.end() ) {
         if( holds_alternative< CoercedType >( search->second ) ) {
            search->second = (CoercedType) value;
         }
         else {
            throw std::logic_error( "Parameter " + name
                                    + " already exists with different type. "
                                      "Current type index is "
                                    + std::to_string( search->second.index() ) + " (variant type is "
                                    + std::string( TNL::getType< Parameter >() )
                                    + "), "
                                      "tried to set value of type "
                                    + std::string( TNL::getType< T >() ) + "." );
         }
      }
      addParameter< T >( name, value );
   }

   /**
    * \brief Returns parameter value.
    *
    * \param name Name of the parameter.
    */
   template< class T >
   T
   getParameter( const std::string& name ) const
   {
      using CoercedType = typename ParameterTypeCoercion< T >::type;
      auto search = parameters.find( name );
      if( search != parameters.end() ) {
         if( holds_alternative< CoercedType >( search->second ) )
            return ParameterTypeCoercion< T >::template convert_back< T >( get< CoercedType >( search->second ) );
         else
            throw Exceptions::ConfigError( "Parameter " + name
                                           + " holds a value of different type than requested. "
                                             "Current type index is "
                                           + std::to_string( search->second.index() ) + " (variant type is "
                                           + std::string( TNL::getType< Parameter >() )
                                           + "), "
                                             "tried to get value of type "
                                           + std::string( TNL::getType< T >() ) + "." );
      }
      throw Exceptions::ConfigError( "The program attempts to get unknown parameter " + name + "." );
   }

   /**
    * \brief Returns parameter list.
    *
    * \param name Name of the parameter list.
    */
   template< class T >
   std::vector< T >
   getList( const std::string& name ) const
   {
      return getParameter< std::vector< T > >( name );
   }

   /**
    * \brief Returns up to three parameter values as a StaticArray.
    *
    * \param prefix Prefix of the parameter names. The names are \e prefix-x,
    *               \e prefix-y and \e prefix-z.
    */
   template< class StaticArray >
   StaticArray
   getXyz( const std::string& prefix ) const
   {
      StaticArray result;
      result[ 0 ] = getParameter< typename StaticArray::ValueType >( prefix + "-x" );
      if( StaticArray::getSize() >= 2 )
         result[ 1 ] = getParameter< typename StaticArray::ValueType >( prefix + "-y" );
      if( StaticArray::getSize() >= 3 )
         result[ 2 ] = getParameter< typename StaticArray::ValueType >( prefix + "-z" );
      return result;
   }

protected:
   std::unordered_map< std::string, Parameter > parameters;
};

}  // namespace Config
}  // namespace noa::TNL
