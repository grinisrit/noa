// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/String.h>
#include <noa/3rdparty/TNL/File.h>

namespace noaTNL {
namespace Containers {

/**
 * \brief Array with constant size.
 *
 * \tparam Size Size of static array. Number of its elements.
 * \tparam Value Type of the values in static array.
 */
template< int Size, typename Value >
class StaticArray
{
public:

   /**
    * \brief Type of elements stored in this array.
    */
   using ValueType = Value;

   /**
    * \brief Type being used for the array elements indexing.
    */
   using IndexType = int;

   /**
    * \brief Gets size of this array.
    */
   __cuda_callable__
   static constexpr int getSize();

   /**
    * \brief Default constructor.
    */
   __cuda_callable__
   StaticArray();

   /**
    * \brief Constructor from static array.
    *
    * \param v Input array.
    */
   // Note: the template avoids ambiguity of overloaded functions with literal 0 and pointer
   // reference: https://stackoverflow.com/q/4610503
   template< typename _unused = void >
   __cuda_callable__
   StaticArray( const Value v[ Size ] );

   /**
    * \brief Constructor that sets all array components to value \e v.
    *
    * \param v Reference to a value.
    */
   __cuda_callable__
   StaticArray( const Value& v );

   /**
    * \brief Copy constructor.
    *
    * Constructs a copy of another static array \e v.
    */
   __cuda_callable__
   StaticArray( const StaticArray< Size, Value >& v );

   /**
    * \brief Constructor which initializes the array by copying elements from
    * \ref std::initializer_list, e.g. `{...}`.
    *
    * The initializer list size must larger or equal to \e Size.
    *
    * @param elems input initializer list
    */
   __cuda_callable__
   StaticArray( const std::initializer_list< Value > &elems );

   /**
    * \brief Constructor that sets components of arrays with Size = 2.
    *
    * \param v1 Value of the first array component.
    * \param v2 Value of the second array component.
    */
   __cuda_callable__
   StaticArray( const Value& v1, const Value& v2 );

   /**
    * \brief Constructor that sets components of arrays with Size = 3.
    *
    * \param v1 Value of the first array component.
    * \param v2 Value of the second array component.
    * \param v3 Value of the third array component.
    */
   __cuda_callable__
   StaticArray( const Value& v1, const Value& v2, const Value& v3 );


   /**
    * \brief Gets pointer to data of this static array.
    */
   __cuda_callable__
   Value* getData();

   /**
    * \brief Gets constant pointer to data of this static array.
    */
   __cuda_callable__
   const Value* getData() const;

   /**
    * \brief Accesses specified element at the position \e i and returns a constant reference to its value.
    *
    * \param i Index position of an element.
    */
   __cuda_callable__
   const Value& operator[]( int i ) const;

   /**
    * \brief Accesses specified element at the position \e i and returns a reference to its value.
    *
    * \param i Index position of an element.
    */
   __cuda_callable__
   Value& operator[]( int i );

   /**
    * \brief Accesses specified element at the position \e i and returns a constant reference to its value.
    *
    * Equivalent to \ref operator[].
    */
   __cuda_callable__
   const Value& operator()( int i ) const;

   /**
    * \brief Accesses specified element at the position \e i and returns a reference to its value.
    *
    * Equivalent to \ref operator[].
    */
   __cuda_callable__
   Value& operator()( int i );

   /**
    * \brief Returns reference to the first coordinate.
    */
   __cuda_callable__
   Value& x();

   /**
    * \brief Returns constant reference to the first coordinate.
    */
   __cuda_callable__
   const Value& x() const;

   /**
    * \brief Returns reference to the second coordinate for arrays with Size >= 2.
    */
   __cuda_callable__
   Value& y();

   /**
    * \brief Returns constant reference to the second coordinate for arrays with Size >= 2.
    */
   __cuda_callable__
   const Value& y() const;

   /**
    * \brief Returns reference to the third coordinate for arrays with Size >= 3.
    */
   __cuda_callable__
   Value& z();

   /**
    * \brief Returns constant reference to the third coordinate for arrays with Size >= 3.
    */
   __cuda_callable__
   const Value& z() const;

   /**
    * \brief Assigns another static \e array to this array.
    */
   __cuda_callable__
   StaticArray< Size, Value >& operator=( const StaticArray< Size, Value >& array );

   /**
    * \brief Assigns an object \e v of type \e T.
    *
    * T can be:
    *
    * 1. Static linear container implementing operator[] and having the same size.
    * In this case, \e v is copied to this array elementwise.
    * 2. An object that can be converted to \e Value type. In this case all elements
    * are set to \e v.
    */
   template< typename T >
   __cuda_callable__
   StaticArray< Size, Value >& operator=( const T& v );

   /**
    * \brief This function checks whether this static array is equal to another \e array.
    *
    * Return \e true if the arrays are equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   __cuda_callable__
   bool operator==( const Array& array ) const;

   /**
    * \brief This function checks whether this static array is not equal to another \e array.
    *
    * Return \e true if the arrays are not equal in size. Otherwise returns \e false.
    */
   template< typename Array >
   __cuda_callable__
   bool operator!=( const Array& array ) const;

   /**
    * \brief Cast operator for changing of the \e Value type.
    *
    * Returns static array having \e ValueType set to \e OtherValue, i.e.
    * StaticArray< Size, OtherValue >.
    *
    * \tparam OtherValue is the \e Value type of the static array the casting
    * will be performed to.
    *
    * \return instance of StaticArray< Size, OtherValue >
    */
   template< typename OtherValue >
   __cuda_callable__
   operator StaticArray< Size, OtherValue >() const;

   /**
    * \brief Sets all values of this static array to \e val.
    */
   __cuda_callable__
   void setValue( const ValueType& val );

   /**
    * \brief Saves this static array into the \e file.
    * \param file Reference to a file.
    */
   bool save( File& file ) const;

   /**
    * \brief Loads data from the \e file to this static array.
    * \param file Reference to a file.
    */
   bool load( File& file);

   /**
    * \brief Sorts the elements in this static array in ascending order.
    */
   void sort();

   /**
    * \brief Writes the array values into stream \e str with specified \e separator.
    *
    * @param str Reference to a stream.
    * @param separator Character separating the array values in the stream \e str.
    * Is set to " " by default.
    */
   std::ostream& write( std::ostream& str, const char* separator = " " ) const;

protected:
   Value data[ Size ];
};

template< int Size, typename Value >
std::ostream& operator<<( std::ostream& str, const StaticArray< Size, Value >& a );

/**
 * \brief Serialization of static arrays into binary files.
 *
 * \param file output file
 * \param array is an array to be written into the output file.
 */
template< int Size, typename Value >
File& operator<<( File& file, const StaticArray< Size, Value >& array );

/**
 * \brief Serialization of static arrays into binary files.
 *
 * \param file output file
 * \param array is an array to be written into the output file.
 */
template< int Size, typename Value >
File& operator<<( File&& file, const StaticArray< Size, Value >& array );

/**
 * \brief Deserialization of static arrays from binary files.
 *
 * \param file input file
 * \param array is an array to be read from the input file.
 */
template< int Size, typename Value >
File& operator>>( File& file, StaticArray< Size, Value >& array );

/**
 * \brief Deserialization of static arrays from binary files.
 *
 * \param file input file
 * \param array is an array to be read from the input file.
 */
template< int Size, typename Value >
File& operator>>( File&& file, StaticArray< Size, Value >& array );


} // namespace Containers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Containers/StaticArray.hpp>
