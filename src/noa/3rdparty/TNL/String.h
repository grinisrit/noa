// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

namespace noa::TNL {

/**
 * \brief Class for managing strings.
 *
 * The following example shows common use of String.
 *
 * \par Example
 * \include StringExample.cpp
 * \par Output
 * \include StringExample.out
 *
 * In addition to methods of this class, check the following related functions:
 *
 * \ref convertToString
 *
 * \ref operator+
 */
class String
: public std::string
{
   public:

      /**
       * \brief This enum defines how the operation split of string is to be performed.
       */
      enum class SplitSkip
      {
         NoSkip,    ///< Do not skip empty characters
         SkipEmpty  ///< Skip empty characters.
      };

      /**
       * \brief Default constructor.
       *
       * Constructs an empty string object.
       */
      String() = default;

      /**
       * \brief Default copy constructor.
       */
      String( const String& ) = default;

      /**
       * \brief Default move constructor.
       */
      String( String&& ) = default;

      /**
       * \brief Initialization by \e std::string.
       */
      String( const std::string& str ) : std::string( str ) {}

      /**
       * \brief Default copy assignment operator.
       */
      String& operator=( const String& ) = default;

      /**
       * \brief Default move assignment operator.
       */
      String& operator=( String&& ) = default;

      /**
       * \brief Inherited constructors.
       */
      using std::string::string;

      /**
       * \brief Inherited assignment operators.
       */
      using std::string::operator=;

      /**
       * \brief Returns the number of characters in given string. Equivalent to \ref getSize.
       */
      int getLength() const;

      /**
       * \brief Returns the number of characters in given string.
       */
      int getSize() const;

      /**
       *  \brief Returns size of allocated storage for given string.
       *
       * \par Example
       * \include StringExampleGetAllocatedSize.cpp
       * \par Output
       * \include StringExampleGetAllocatedSize.out
       */
      int getAllocatedSize() const;

      /**
       * \brief Reserves space for given \e size.
       *
       * Requests to allocate storage space of given \e size to avoid memory reallocation.
       * It allocates one more byte for the terminating 0.
       * @param size Number of characters.
       *
       * \par Example
       * \include StringExampleSetSize.cpp
       * \par Output
       * \include StringExampleSetSize.out
       */
      void setSize( int size );

      /**
       * \brief Returns pointer to data.
       *
       * It returns the content of the given string as a constant pointer to char.
       */
      const char* getString() const;

      /**
       * \brief Returns pointer to data. Alias of \ref std::string::data.
       */
      const char* getData() const;

      /**
       * \brief Returns pointer to data. Alias of \ref std::string::data.
       */
      char* getData();

      /**
       * \brief Operator for accessing particular chars of the string.
       *
       * This function overloads \ref operator[]. It returns a reference to
       * the character at position \e i in given string.
       * The character can not be changed be user.
       */
      const char& operator[]( int i ) const;

      /**
       *  \brief Operator for accessing particular chars of the string.
       *
       * It returns the character at the position \e i in given string as
       * a modifiable reference.
       */
      char& operator[]( int i );

      /**
       * Operators for single characters.
       */

      /**
       * \brief This function overloads \ref operator+=.
       *
       * Appends character \e str to this string.
       */
      String& operator+=( char str );

      /**
       * \brief This function concatenates strings and returns a newly constructed string object.
       */
      String operator+( char str ) const;

      /**
       * \brief This function checks whether the given string is equal to \e str.
       *
       * It returns \e true when the given string is equal to \e str. Otherwise it returns \e false.
       */
      bool operator==( char str ) const;

      /**
       * \brief This function overloads \ref operator!=.
       *
       * It returns \e true when the given string is NOT equal to \e str. Otherwise it returns \e true.
       */
      bool operator!=( char str ) const;

      /**
       * Operators for C strings.
       */

      /**
       * \brief This function overloads \ref operator+=.
       *
       * It appends the C string \e str to this string.
       */
      String& operator+=( const char* str );

      /**
       * \brief This function concatenates C strings \e str and returns a newly constructed string object.
       */
      String operator+( const char* str ) const;

      /**
       * \brief This function overloads \ref operator==.
       *
       * It returns \e true when the given string is equal to \e str. Otherwise it returns \e false.
       */
      bool operator==( const char* str ) const;

      /**
       * \brief This function overloads \ref operator!=.
       *
       * It returns \e true when the given string is NOT equal to \e str. Otherwise it returns \e true.
       */
      bool operator!=( const char* str ) const;

      /**
       * Operators for std::string.
       */

      /**
       * \brief This function overloads \ref operator+=.
       *
       * It appends the C string \e str to this string.
       */
      String& operator+=( const std::string& str );

      /**
       * \brief This function concatenates C strings \e str and returns a newly constructed string object.
       */
      String operator+( const std::string& str ) const;

      /**
       * \brief This function overloads \ref operator==.
       *
       * It returns \e true when the given string is equal to \e str. Otherwise it returns \e false.
       */
      bool operator==( const std::string& str ) const;

      /**
       * \brief This function overloads \ref operator!=.
       *
       * It returns \e true when the given string is NOT equal to \e str. Otherwise it returns \e true.
       */
      bool operator!=( const std::string& str ) const;

      /**
       * Operators for String.
       */

      /**
       * \brief This function overloads \ref operator+=.
       *
       * It appends the C string \e str to this string.
       */
      String& operator+=( const String& str );

      /**
       * \brief This function concatenates C strings \e str and returns a newly constructed string object.
       */
      String operator+( const String& str ) const;

      /**
       * \brief This function overloads \ref operator==.
       *
       * It returns \e true when the given string is equal to \e str. Otherwise it returns \e false.
       */
      bool operator==( const String& str ) const;

      /**
       * \brief This function overloads \ref operator!=.
       *
       * It returns \e true when the given string is NOT equal to \e str. Otherwise it returns \e true.
       */
      bool operator!=( const String& str ) const;

      /**
       *  \brief Cast to bool operator.
       *
       * This operator converts string to boolean expression (true or false).
       * It returns \e true if the string is NOT empty. Otherwise it returns \e false.
       */
      operator bool() const;

      /** \brief Cast to bool with negation operator.
       *
       * This operator converts string to boolean expression (false or true).
       * It returns \e true if the string is empty. Otherwise it returns \e false.
       */
      bool operator!() const;

      /**
       * \brief This method replaces part of the string.
       *
       * It replaces \e pattern in this string with a string \e replaceWith.
       * If parameter \e count is defined, the function makes replacement only count occurrences,
       * of the given pattern. If \e count is zero, all pattern occurrences are replaced.
       *
       * @param pattern to be replaced.
       * @param replaceWith string the \e pattern will be replaced with.
       * @param count number of occurrences to be replaced. All occurrences are replaced if \e count is zero..
       *
       * \par Example
       * \include StringExampleReplace.cpp
       * \par Output
       * \include StringExampleReplace.out
       */
      String replace( const String& pattern,
                      const String& replaceWith,
                      int count = 0 ) const;

      /**
       * \brief Trims/strips this string.
       *
       * Removes all 'spaces' from given string except for single 'spaces' between words.
       *
       * @param strip can be used to change the character to be removed.
       *
       * \par Example
       * \include StringExampleStrip.cpp
       * \par Output
       * \include StringExampleStrip.out
       */
      String strip( char strip = ' ' ) const;

      /**
       *  \brief Splits string into list of strings with respect to given \e separator.
       *
       * This method splits the string into sequence of substrings divided by occurrences of \e separator.
       * It returns the list of those strings via std::vector. When \e separator does not appear
       * anywhere in the given string, this function returns a single-element list
       * containing given sting. If \e skipEmpty equals \e SkipEmpty no empty substrings are
       * inserted into the resulting container.
       *
       * @param separator is a character separating substrings in given string.
       * @param skipEmpty
       *
       * \par Example
       * \include StringExampleSplit.cpp
       * \par Output
       * \include StringExampleSplit.out
       */
      std::vector< String > split( const char separator = ' ', SplitSkip skipEmpty = SplitSkip::NoSkip ) const;

      /**
       * \brief Checks if the string starts with given prefix.
       */
      bool startsWith( const String& prefix ) const;

      /**
       * \brief Checks if the string ends with given suffix.
       */
      bool endsWith( const String& suffix ) const;
};

/**
 * \brief Returns concatenation of \e string1 and \e string2.
 */
String operator+( char string1, const String& string2 );

/**
 * \brief Returns concatenation of \e string1 and \e string2.
 */
String operator+( const char* string1, const String& string2 );

/**
 * \brief Returns concatenation of \e string1 and \e string2.
 */
String operator+( const std::string& string1, const String& string2 );

/**
 * \brief Converts \e value of type \e T to a String.
 *
 * \tparam T can be any type fir which operator << is defined.
 */
template< typename T >
String convertToString( const T& value )
{
   std::stringstream str;
   str << value;
   return String( str.str().data() );
}

/**
 * \brief Specialization of function \ref convertToString for boolean.
 *
 * The boolean type is converted to 'true' or 'false'.
 */
template<> inline String convertToString( const bool& b )
{
   if( b ) return "true";
   return "false";
}

} // namespace noa::TNL

#include <noa/3rdparty/TNL/String.hpp>
