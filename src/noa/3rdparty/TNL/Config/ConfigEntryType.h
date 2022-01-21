// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Jakub Klinkovský

#pragma once

#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <string>
#include <stdexcept>
#include <variant>   // backport of std::variant from C++17

namespace TNL {
namespace Config {

using std::variant;
using std::get;
using std::monostate;
using std::holds_alternative;

// aliases for integer types
using UnsignedInteger = std::size_t;
using Integer = std::make_signed_t< std::size_t >;

using Parameter = variant< monostate,
                           bool,
                           Integer,
                           UnsignedInteger,
                           double,
                           std::string,
                           std::vector< bool >,
                           std::vector< Integer >,
                           std::vector< UnsignedInteger >,
                           std::vector< double >,
                           std::vector< std::string >
                         >;

template< typename T >
struct ParameterTypeCoercion
{
   using type =
      std::conditional_t< std::is_same< T, bool >::value, bool,
         std::conditional_t< std::is_integral< T >::value && std::is_signed< T >::value, Integer,
            std::conditional_t< std::is_integral< T >::value && std::is_unsigned< T >::value, UnsignedInteger,
               std::conditional_t< std::is_floating_point< T >::value, double,
                  std::conditional_t< std::is_base_of< std::string, T >::value, std::string,
                     std::conditional_t< std::is_same< std::decay_t< T >, const char* >::value, std::string,
                        T
                     >
                  >
               >
            >
         >
      >;

   static type convert( const T& v ) { return v; }
   template< typename Result >
   static Result convert_back( const type& v ) { return v; }
};

template< typename T >
struct ParameterTypeCoercion< std::vector< T > >
{
   using type = std::vector< typename ParameterTypeCoercion< T >::type >;

   static type convert( const std::vector< T >& vec )
   {
      type new_vec;
      for( auto value : vec )
         new_vec.push_back( value );
      return new_vec;
   }

   template< typename Result >
   static Result convert_back( const type& vec )
   {
      Result new_vec;
      for( auto value : vec )
         new_vec.push_back( value );
      return new_vec;
   }
};

template< typename EntryType >
std::string getUIEntryType() { throw std::logic_error( "getUIEntryType called with unknown type." ); };

template<> inline std::string getUIEntryType< bool >()             { return "bool"; };
template<> inline std::string getUIEntryType< Integer >()          { return "integer"; };
template<> inline std::string getUIEntryType< UnsignedInteger >()  { return "unsigned integer"; };
template<> inline std::string getUIEntryType< double >()           { return "real"; };
template<> inline std::string getUIEntryType< std::string >()      { return "string"; };

template<> inline std::string getUIEntryType< std::vector< bool > >()             { return "list of bool"; };
template<> inline std::string getUIEntryType< std::vector< Integer > >()          { return "list of integer"; };
template<> inline std::string getUIEntryType< std::vector< UnsignedInteger > >()  { return "list of unsigned integer"; };
template<> inline std::string getUIEntryType< std::vector< double > >()           { return "list of real"; };
template<> inline std::string getUIEntryType< std::vector< std::string > >()      { return "list of string"; };

} // namespace Config
} // namespace TNL
