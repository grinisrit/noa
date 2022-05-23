// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>

namespace noa::TNL {

inline int
String::getLength() const
{
   return getSize();
}

inline int
String::getSize() const
{
   return this->size();
}

inline int
String::getAllocatedSize() const
{
   return this->capacity();
}

inline void
String::setSize( int size )
{
   TNL_ASSERT_GE( size, 0, "string size must be non-negative" );
   this->resize( size );
}

inline const char*
String::getString() const
{
   return this->c_str();
}

inline const char*
String::getData() const
{
   return data();
}

inline char*
String::getData()
{
   // NOTE: std::string::data is non-const only since C++17
   return const_cast< char* >( data() );
}

inline const char&
String::operator[]( int i ) const
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, getSize(), "Element index is out of bounds." );
   return std::string::operator[]( i );
}

inline char&
String::operator[]( int i )
{
   TNL_ASSERT_GE( i, 0, "Element index must be non-negative." );
   TNL_ASSERT_LT( i, getSize(), "Element index is out of bounds." );
   return std::string::operator[]( i );
}

/****
 * Operators for single characters
 */
inline String&
String::operator+=( char str )
{
   std::string::operator+=( str );
   return *this;
}

inline String
String::operator+( char str ) const
{
   return String( *this ) += str;
}

inline bool
String::operator==( char str ) const
{
   return std::string( *this ) == std::string( 1, str );
}

inline bool
String::operator!=( char str ) const
{
   return ! operator==( str );
}

/****
 * Operators for C strings
 */
inline String&
String::operator+=( const char* str )
{
   std::string::operator+=( str );
   return *this;
}

inline String
String::operator+( const char* str ) const
{
   return String( *this ) += str;
}

inline bool
String::operator==( const char* str ) const
{
   return std::string( *this ) == str;
}

inline bool
String::operator!=( const char* str ) const
{
   return ! operator==( str );
}

/****
 * Operators for std::string
 */
inline String&
String::operator+=( const std::string& str )
{
   std::string::operator+=( str );
   return *this;
}

inline String
String::operator+( const std::string& str ) const
{
   return String( *this ) += str;
}

inline bool
String::operator==( const std::string& str ) const
{
   return std::string( *this ) == str;
}

inline bool
String::operator!=( const std::string& str ) const
{
   return ! operator==( str );
}

/****
 * Operators for String
 */
inline String&
String::operator+=( const String& str )
{
   std::string::operator+=( str );
   return *this;
}

inline String
String::operator+( const String& str ) const
{
   return String( *this ) += str;
}

inline bool
String::operator==( const String& str ) const
{
   return std::string( *this ) == str;
}

inline bool
String::operator!=( const String& str ) const
{
   return ! operator==( str );
}

inline String::operator bool() const
{
   return ! empty();
}

inline bool
String::operator!() const
{
   return ! operator bool();
}

inline String
String::replace( const String& pattern, const String& replaceWith, int count ) const
{
   std::string newString = *this;

   std::size_t index = 0;
   for( int i = 0; i < count || count == 0; i++ ) {
      // locate the substring to replace
      index = newString.find( pattern, index );
      if( index == std::string::npos )
         break;

      // make the replacement
      newString.replace( index, pattern.getLength(), replaceWith );
      index += replaceWith.getLength();
   }

   return newString;
}

inline String
String::strip( char strip ) const
{
   int prefix_cut_off = 0;
   int sufix_cut_off = 0;

   while( prefix_cut_off < getLength() && ( *this )[ prefix_cut_off ] == strip )
      prefix_cut_off++;

   while( sufix_cut_off < getLength() && ( *this )[ getLength() - 1 - sufix_cut_off ] == strip )
      sufix_cut_off++;

   if( prefix_cut_off + sufix_cut_off < getLength() )
      return substr( prefix_cut_off, getLength() - prefix_cut_off - sufix_cut_off );
   return "";
}

inline std::vector< String >
String::split( char separator, SplitSkip skip ) const
{
   std::vector< String > parts;
   String s;
   for( int i = 0; i < this->getLength(); i++ ) {
      if( ( *this )[ i ] == separator ) {
         if( skip != SplitSkip::SkipEmpty || ! s.empty() )
            parts.push_back( s );
         s = "";
      }
      else
         s += ( *this )[ i ];
   }
   if( skip != SplitSkip::SkipEmpty || ! s.empty() )
      parts.push_back( s );
   return parts;
}

inline bool
String::startsWith( const String& prefix ) const
{
   if( prefix.getSize() > getSize() )
      return false;
   return std::equal( prefix.begin(), prefix.end(), begin() );
}

inline bool
String::endsWith( const String& suffix ) const
{
   if( suffix.getSize() > getSize() )
      return false;
   return std::equal( suffix.rbegin(), suffix.rend(), rbegin() );
}

inline String
operator+( char string1, const String& string2 )
{
   return convertToString( string1 ) + string2;
}

inline String
operator+( const char* string1, const String& string2 )
{
   return String( string1 ) + string2;
}

inline String
operator+( const std::string& string1, const String& string2 )
{
   return String( string1 ) + string2;
}

}  // namespace noa::TNL
