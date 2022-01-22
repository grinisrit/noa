// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <list>
#include <vector>

#include <noa/3rdparty/TNL/File.h>
#include <noa/3rdparty/TNL/TypeTraits.h>
#include <noa/3rdparty/TNL/Allocators/Default.h>
#include <noa/3rdparty/TNL/Containers/ArrayView.h>

namespace noaTNL {
/**
 * \brief Namespace for TNL containers.
 */
namespace Containers {

template< int, typename > class StaticArray;

/**
 * \brief \e Array is responsible for memory management, access to array
 * elements, and general array operations.
 *
 * \tparam Value  The type of array elements.
 * \tparam Device The device to be used for the execution of array operations.
 *                It can be either \ref Devices::Host or \ref Devices::Cuda.
 * \tparam Index  The indexing type.
 * \tparam Allocator The type of the allocator used for the allocation and
 *                   deallocation of memory used by the array. By default,
 *                   an appropriate allocator for the specified \e Device
 *                   is selected with \ref Allocators::Default.
 *
 * Memory management handled by constructors and destructors according to the
 * [RAII](https://en.wikipedia.org/wiki/RAII) principle and by methods
 * \ref resize \ref setSize, \ref setLike, \ref swap, and \ref reset. You can
 * also use methods \ref getSize and \ref empty to check the current array size
 * and \ref getData to access the raw pointer.
 *
 * Methods annotated as \ref \_\_cuda_callable\_\_ can be called either from
 * host or from kernels executing on a device according to the \e Device
 * parameter. One of these methods is the \ref operator[] which provides direct
 * access to the array elements. However, it cannot be called from the host if
 * the array was allocated in a memory space which is not directly accessible
 * by the host. If the host needs access to individual array elements which are
 * allocated in a different memory space, they have to be accessed by the
 * \ref setElement or \ref getElement method. However, these methods imply an
 * explicit data transfer which is not buffered, so it can be very slow.
 *
 * Other methods, such as \ref operator=, \ref operator==, \ref operator!=,
 * \ref setValue, and \ref evaluate, provide various operations on whole arrays.
 *
 * See also \ref ArrayView, \ref Vector, \ref VectorView.
 *
 * \par Example
 * \include Containers/ArrayExample.cpp
 * \par Output
 * \include ArrayExample.out
 */
template< typename Value,
          typename Device = Devices::Host,
          typename Index = int,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Value > >
class Array
{
   static_assert( std::is_same< std::remove_cv_t< Value>, std::remove_cv_t< typename Allocator::value_type > >::value, "Mismatch of Array::Value and Allocator::value_type. The type must be the same." );
   public:
      /**
       * \brief Type of elements stored in this array.
       */
      using ValueType = Value;

      /**
       * \brief Device where the array is allocated.
       *
       * See \ref Devices::Host or \ref Devices::Cuda.
       */
      using DeviceType = Device;

      /**
       * \brief Type being used for the array elements indexing.
       */
      using IndexType = Index;

      /**
       * \brief Allocator type used for allocating this array.
       *
       * See \ref Allocators::Cuda, \ref Allocators::CudaHost, \ref Allocators::CudaManaged, \ref Allocators::Host or \ref Allocators:Default.
       */
      using AllocatorType = Allocator;

      /**
       * \brief Compatible ArrayView type.
       */
      using ViewType = ArrayView< Value, Device, Index >;

      /**
       * \brief Compatible constant ArrayView type.
       */
      using ConstViewType = ArrayView< std::add_const_t< Value >, Device, Index >;

      /**
       * \brief A template which allows to quickly obtain an \ref Array type with changed template parameters.
       */
      template< typename _Value,
                typename _Device = Device,
                typename _Index = Index,
                typename _Allocator = typename Allocators::Default< _Device >::template Allocator< _Value > >
      using Self = Array< _Value, _Device, _Index, _Allocator >;


      /**
       * \brief Constructs an empty array with zero size.
       */
      Array() = default;

      /**
       * \brief Constructs an empty array and sets the provided allocator.
       *
       * \param allocator The allocator to be associated with this array.
       */
      explicit Array( const AllocatorType& allocator );

      /**
       * \brief Constructs an array with given size.
       *
       * \param size The number of array elements to be allocated.
       * \param allocator The allocator to be associated with this array.
       */
      explicit Array( IndexType size, const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Constructs an array with given size and value.
       *
       * \param size The number of array elements to be allocated.
       * \param value The value all elements will be set to.
       * \param allocator The allocator to be associated with this array.
       */
      explicit Array( IndexType size, ValueType value, const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Constructs an array with given size and copies data from given
       * pointer.
       *
       * \param data The pointer to the data to be copied to the array.
       * \param size The number of array elements to be copied to the array.
       * \param allocator The allocator to be associated with this array.
       */
      Array( ValueType* data,
             IndexType size,
             const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Copy constructor (makes a deep copy).
       *
       * \param array The array to be copied.
       */
      explicit Array( const Array& array );

      /**
       * \brief Copy constructor with a specific allocator (makes a deep copy).
       *
       * \param array The array to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      explicit Array( const Array& array, const AllocatorType& allocator );

      /**
       * \brief Copy constructor (makes a deep copy).
       *
       * \param array The array to be copied.
       * \param begin The first index which should be copied.
       * \param size The number of elements that should be copied.
       * \param allocator The allocator to be associated with this array.
       */
      Array( const Array& array,
             IndexType begin,
             IndexType size = 0,
             const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Move constructor for initialization from \e rvalues.
       *
       * \param array The array to be moved.
       */
      Array( Array&& array );


      /**
       * \brief Copy constructor from array with different template parameters.
       *
       * \tparam Value_ Value type of the input array.
       * \tparam Device_ Device type of the input array.
       * \tparam Index_ Index type of the input array.
       * \tparam Allocator_ Allocator type of the input array.
       * \param a the input array.
       */
      template< typename Value_,
                typename Device_,
                typename Index_,
                typename Allocator_ >
      explicit Array( const Array< Value_, Device_, Index_, Allocator_ >& a );

      /**
       * \brief Constructor which initializes the array by copying elements from
       * \ref std::initializer_list, e.g. `{...}`.
       *
       * \param list The initializer list containing elements to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      template< typename InValue >
      Array( const std::initializer_list< InValue >& list,
             const AllocatorType& allocator = AllocatorType() );

      /**
       * \brief Constructor which initializes the array by copying elements from
       * \ref std::list.
       *
       * \param list The STL list containing elements to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      template< typename InValue >
      Array( const std::list< InValue >& list,
             const AllocatorType& allocator = AllocatorType() );


      /**
       * \brief Constructor which initializes the array by copying elements from
       * \ref std::vector.
       *
       * \param vector The STL vector containing elements to be copied.
       * \param allocator The allocator to be associated with this array.
       */
      template< typename InValue >
      Array( const std::vector< InValue >& vector,
             const AllocatorType& allocator = AllocatorType() );


      /**
       * \brief Returns the allocator associated with the array.
       */
      AllocatorType getAllocator() const;

      /**
       * \brief Returns a \ref String representation of the array type in C++ style,
       * with a placeholder in place of \e Device and \e Allocator.
       */
      static String getSerializationType();

      /**
       * \brief Returns a \ref String representation of the array type in C++ style,
       * with a placeholder in place of \e Device and \e Allocator.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Method for resizing the array.
       *
       * The method resizes the array to the given size:
       *
       * - If the current size is greater than count, the array is reduced to
       *   its first \e size elements.
       * - If the current size is less than \e size, additional elements are
       *   appended (see the note below on initialization).
       * - If the current size is equal to \e size, nothing happens.
       *
       * If the array size changes, the current data will be deallocated, thus
       * all pointers and views to the array alements will become invalid.
       *
       * Note that this method differs from \ref std::vector::resize with respect
       * to the initialization of array elements:
       *
       * - if \e ValueType is a [fundamental type](https://en.cppreference.com/w/cpp/types/is_fundamental),
       *   the elements are [default-initialized](https://en.cppreference.com/w/cpp/language/default_initialization)
       *   (i.e., the elements are initialized to indeterminate values).
       * - otherwise, the elements are [value-initialized](https://en.cppreference.com/w/cpp/language/value_initialization)
       *   (like in \ref std::vector::resize).
       *
       * \param size The new size of the array.
       */
      void resize( IndexType size );

      /**
       * \brief Method for resizing the array with an initial value.
       *
       * The method resizes the array to the given size:
       *
       * - If the current size is greater than count, the array is reduced to
       *   its first \e size elements.
       * - If the current size is less than \e size, additional copies of
       *   \e value are appended.
       * - If the current size is equal to \e size, nothing happens.
       *
       * If the array size changes, the current data will be deallocated, thus
       * all pointers and views to the array alements will become invalid.
       *
       * \param size The new size of the array.
       * \param value The value to initialize new elements with.
       */
      void resize( IndexType size, ValueType value );

      /**
       * \brief Method for setting the array size.
       *
       * This method behaves almost like \ref resize, but when the array size
       * is changed, old elements are not copied to the new memory location.
       * Hence, this is a shortcut for deallocating the array with \e resize(0)
       * followed by setting the new size with \e resize(size)
       *
       * \param size The new size of the array.
       */
      void setSize( IndexType size );

      /**
       * \brief Returns the current array size.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ IndexType getSize() const;

      /**
       * \brief Sets the same size as the size of an existing array.
       *
       * If the array size changes, the current data will be deallocated, thus
       * all pointers and views to the array alements will become invalid.
       *
       * Note that this method uses \ref setSize rather than \ref resize.
       *
       * \tparam ArrayT The type of the parameter can be any type which provides
       *         the method \ref getSize() with the same signature as \e Array.
       * \param array The array whose size is to be taken.
       */
      template< typename ArrayT >
      void setLike( const ArrayT& array );

      /**
       * \brief Returns a modifiable view of the array.
       *
       * By default, a view for the whole array is returned. If \e begin or
       * \e end is set to a non-zero value, a view only for the sub-interval
       * `[begin, end)` is returned.
       *
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      ViewType getView( IndexType begin = 0, IndexType end = 0 );

      /**
       * \brief Returns a non-modifiable view of the array.
       *
       * By default, a view for the whole array is returned. If \e begin or
       * \e end is set to a non-zero value, a view only for the sub-interval
       * `[begin, end)` is returned.
       *
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      ConstViewType getConstView( IndexType begin = 0, IndexType end = 0 ) const;

      /**
       * \brief Conversion operator to a modifiable view of the array.
       */
      operator ViewType();

      /**
       * \brief Conversion operator to a non-modifiable view of the array.
       */
      operator ConstViewType() const;

      /**
       * \brief Swaps this array with another.
       *
       * Swapping is done in a shallow way, i.e. only pointers and sizes are
       * swapped.
       *
       * \param array The array to be swapped with this array.
       */
      void swap( Array& array );

      /**
       * \brief Resets the array to the empty state.
       *
       * The current data will be deallocated, thus all pointers and views to
       * the array elements will become invalid.
       */
      void reset();

      /**
       * \brief Returns \e true if the current array size is zero.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__
      bool empty() const;

      /**
       * \brief Returns a \e const-qualified raw pointer to the data.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ const Value* getData() const;

      /**
       * \brief Returns a raw pointer to the data.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ Value* getData();

      /**
       * \brief Returns a \e const-qualified raw pointer to the data.
       *
       * Use this method in algorithms where you want to emphasize that
       * C-style array pointer is required.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ const Value* getArrayData() const;

      /**
       * \brief Returns a raw pointer to the data.
       *
       * Use this method in algorithms where you want to emphasize that
       * C-style array pointer is required.
       *
       * This method can be called from device kernels.
       */
      __cuda_callable__ Value* getArrayData();


      /**
       * \brief Sets the value of the \e i-th element to \e v.
       *
       * This method can be called from both the host system and the device
       * where the array is allocated.
       *
       * \param i The index of the element to be set.
       * \param value The new value of the element.
       */
      __cuda_callable__
      void setElement( IndexType i, ValueType value );

      /**
       * \brief Returns the value of the \e i-th element.
       *
       * This method can be called from both the host system and the device
       * where the array is allocated.
       *
       * \param i The index of the element to be returned.
       */
      __cuda_callable__
      ValueType getElement( IndexType i ) const;

      /**
       * \brief Accesses the \e i-th element of the array.
       *
       * This method can be called only from the device which has direct access
       * to the memory space where the array was allocated. For example, if the
       * array was allocated in the host memory, it can be called only from
       * host, and if the array was allocated in the device memory, it can be
       * called only from device kernels. If NDEBUG is not defined, assertions
       * inside this methods performs runtime checks for cross-device memory
       * accesses which lead to segmentation fault. If you need to do just a
       * pointer arithmetics use \e getData instead.
       *
       * \param i The index of the element to be accessed.
       * \return Reference to the \e i-th element.
       */
      __cuda_callable__ Value& operator[]( IndexType i );

      /**
       * \brief Accesses the \e i-th element of the array.
       *
       * This method can be called only from the device which has direct access
       * to the memory space where the array was allocated. For example, if the
       * array was allocated in the host memory, it can be called only from
       * host, and if the array was allocated in the device memory, it can be
       * called only from device kernels. If NDEBUG is not defined, assertions
       * inside this methods performs runtime checks for cross-device memory
       * accesses which lead to segmentation fault. If you need to do just a
       * pointer arithmetics use \e getData instead.
       *
       * \param i The index of the element to be accessed.
       * \return Constant reference to the \e i-th element.
       */
      __cuda_callable__ const Value& operator[]( IndexType i ) const;

      /**
       * \brief Accesses the \e i-th element of the array.
       *
       * Equivalent to \ref operator[], with the same notes and caveats.
       */
      __cuda_callable__ Value& operator()( IndexType i );

      /**
       * \brief Accesses the \e i-th element of the array.
       *
       * Equivalent to \ref operator[], with the same notes and caveats.
       */
      __cuda_callable__ const Value& operator()( IndexType i ) const;

      /**
       * \brief Copy-assignment operator for copying data from another array.
       *
       * \param array Reference to the source array.
       * \return Reference to this array.
       */
      Array& operator=( const Array& array );

      /**
       * \brief Move-assignment operator for acquiring data from \e rvalues.
       *
       * \param array Reference to the source array.
       * \return Reference to this array.
       */
      Array& operator=( Array&& array );

      /**
       * \brief Assigns either array-like container or a single value.
       *
       * If \e T is an array type, e.g. \ref Array, \ref ArrayView,
       * \ref StaticArray, \ref Vector, \ref VectorView, or \ref StaticVector,
       * the elements from \e data are copied into this array. Otherwise, if it
       * is a type convertible to \ref ValueType, all array elements are set to
       * the value \e data.
       *
       * \tparam T The type of the source array or value.
       * \param data Reference to the source array or value.
       * \return Reference to this array.
       */
      template< typename T,
                typename...,
                typename = std::enable_if_t< std::is_convertible< T, ValueType >::value || IsArrayType< T >::value > >
      Array& operator=( const T& data );

      /**
       * \brief Copies elements from \ref std::list to this array.
       *
       * \param list The STL list containing elements to be copied.
       * \return Reference to this array.
       */
      template< typename InValue >
      Array& operator=( const std::list< InValue >& list );

      /**
       * \brief Copies elements from \ref std::vector to this array.
       *
       * \param list The STL vector containing elements to be copied.
       * \return Reference to this array.
       */
      template< typename InValue >
      Array& operator=( const std::vector< InValue >& vector );

      /**
       * \brief Compares the array with another array-like container.
       *
       * \tparam ArrayT The type of the parameter can be any array-like
       *         container, e.g. \ref Array, \ref ArrayView, \ref Vector,
       *         \ref VectorView, etc.
       * \param array Reference to the array-like container.
       * \return `true` if both arrays are element-wise equal and `false`
       *         otherwise.
       */
      template< typename ArrayT >
      bool operator==( const ArrayT& array ) const;

      /**
       * \brief Compares the array with another array-like container.
       *
       * \tparam ArrayT The type of the parameter can be any array-like
       *         container, e.g. \ref Array, \ref ArrayView, \ref Vector,
       *         \ref VectorView, etc.
       * \param array Reference to the array-like container.
       * \return The negated result of \ref operator==.
       */
      template< typename ArrayT >
      bool operator!=( const ArrayT& array ) const;

      /**
       * \brief Sets elements of the array to given value.
       *
       * By default, all array elements are set to the given value. If \e begin
       * or \e end is set to a non-zero value, only elements in the sub-interval
       * `[begin, end)` are set.
       *
       * \param value The new value for the array elements.
       * \param begin The beginning of the array sub-interval. It is 0 by
       *              default.
       * \param end The end of the array sub-interval. The default value is 0
       *            which is, however, replaced with the array size.
       */
      void setValue( ValueType value,
                     IndexType begin = 0,
                     IndexType end = 0 );

      /**
       * \brief Process the lambda function \e f for each array element in interval [ \e begin, \e end).
       *
       * The lambda function is supposed to be declared as
       *
       * ```
       * f( IndexType elementIdx, ValueType& elementValue )
       * ```
       *
       * where
       *
       * - \e elementIdx is an index of the array element being currently processed
       * - \e elementValue is a value of the array element being currently processed
       *
       * This is performed at the same place where the array is allocated,
       * i.e. it is efficient even on GPU.
       *
       * \param begin The beginning of the array elements interval.
       * \param end The end of the array elements interval.
       * \param f The lambda function to be processed.
       *
       * \par Example
       * \include Containers/ArrayExample_forElements.cpp
       * \par Output
       * \include ArrayExample_forElements.out
       *
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& f );

      /**
       * \brief Process the lambda function \e f for each array element in interval [ \e begin, \e end) for constant instances of the array.
       *
       * The lambda function is supposed to be declared as
       *
       * ```
       * f( IndexType elementIdx, const ValueType& elementValue )
       * ```
       *
       * where
       *
       * - \e elementIdx is an index of the array element being currently processed
       * - \e elementValue is a value of the array element being currently processed
       *
       * This is performed at the same place where the array is allocated,
       * i.e. it is efficient even on GPU.
       *
       * \param begin The beginning of the array elements interval.
       * \param end The end of the array elements interval.
       * \param f The lambda function to be processed.
       *
       * \par Example
       * \include Containers/ArrayExample_forElements.cpp
       * \par Output
       * \include ArrayExample_forElements.out
       *
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& f ) const;

      /**
       * \brief Process the lambda function \e f for each array element.
       *
       * The lambda function is supposed to be declared as
       *
       * ```
       * f( IndexType elementIdx, ValueType& elementValue )
       * ```
       *
       * where
       *
       * - \e elementIdx is an index of the array element being currently processed
       * - \e elementValue is a value of the array element being currently processed
       *
       * This is performed at the same place where the array is allocated,
       * i.e. it is efficient even on GPU.
       *
       * \param f The lambda function to be processed.
       *
       * \par Example
       * \include Containers/ArrayExample_forElements.cpp
       * \par Output
       * \include ArrayExample_forElements.out
       *
       */
      template< typename Function >
      void forAllElements( Function&& f );

      /**
       * \brief Process the lambda function \e f for each array element for constant instances.
       *
       * The lambda function is supposed to be declared as
       *
       * ```
       * f( IndexType elementIdx, const ValueType& elementValue )
       * ```
       *
       * where
       *
       * - \e elementIdx is an index of the array element being currently processed
       * - \e elementValue is a value of the array element being currently processed
       *
       * This is performed at the same place where the array is allocated,
       * i.e. it is efficient even on GPU.
       *
       * \param f The lambda function to be processed.
       *
       * \par Example
       * \include Containers/ArrayExample_forElements.cpp
       * \par Output
       * \include ArrayExample_forElements.out
       *
       */
      template< typename Function >
      void forAllElements( Function&& f ) const;

      /**
       * \brief Method for saving the array to a binary file \e fileName.
       *
       * \param fileName The output file name.
       */
      void save( const String& fileName ) const;

      /**
       * \brief Method for loading the array from a binary file \e fileName.
       *
       * \param fileName The input file name.
       */
      void load( const String& fileName );

      /** \brief Destructor. */
      virtual ~Array();

   protected:

      /** \brief Internal method for releasing (deallocating) array data. */
      void releaseData();

      /** \brief Internal method for reallocating array elements. Used only
       * from the two overloads of \ref resize.
       */
      void reallocate( IndexType size );

      /** \brief Pointer to the data. */
      Value* data = nullptr;

      /** \brief Number of elements in the array. */
      IndexType size = 0;

      /**
       * \brief The internal allocator instance.
       */
      Allocator allocator;
};

/**
 * \brief Overloaded insertion operator for printing an array to output stream.
 *
 * \tparam Value is a type of the array elements.
 * \tparam Device is a device where the array is allocated.
 * \tparam Index is a type used for the indexing of the array elements.
 *
 * \param str is a output stream.
 * \param array is the array to be printed.
 *
 * \return a reference on the output stream \ref std::ostream&.
 */
template< typename Value, typename Device, typename Index, typename Allocator >
std::ostream& operator<<( std::ostream& str, const Array< Value, Device, Index, Allocator >& array );

/**
 * \brief Serialization of arrays into binary files.
 */
template< typename Value, typename Device, typename Index, typename Allocator >
File& operator<<( File& file, const Array< Value, Device, Index, Allocator >& array );

template< typename Value, typename Device, typename Index, typename Allocator >
File& operator<<( File&& file, const Array< Value, Device, Index, Allocator >& array );

/**
 * \brief Deserialization of arrays from binary files.
 */
template< typename Value, typename Device, typename Index, typename Allocator >
File& operator>>( File& file, Array< Value, Device, Index, Allocator >& array );

template< typename Value, typename Device, typename Index, typename Allocator >
File& operator>>( File&& file, Array< Value, Device, Index, Allocator >& array );

} // namespace Containers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Containers/Array.hpp>
