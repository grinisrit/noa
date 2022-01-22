// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/Array.h>
#include <noa/3rdparty/TNL/Containers/VectorView.h>

namespace noaTNL {
namespace Containers {

/**
 * \brief \e Vector extends \ref Array with algebraic operations.
 *
 * The template parameters have the same meaning as in \ref Array, with \e Real
 * corresponding to \e Array's \e Value parameter.
 *
 * \tparam Real   An arithmetic type for the vector values, e.g. `float` or
 *                `double`.
 * \tparam Device The device to be used for the execution of vector operations.
 * \tparam Index  The indexing type.
 * \tparam Allocator The type of the allocator used for the allocation and
 *                   deallocation of memory used by the array. By default,
 *                   an appropriate allocator for the specified \e Device
 *                   is selected with \ref Allocators::Default.
 *
 * \par Example
 * \include Containers/VectorExample.cpp
 * \par Output
 * \include VectorExample.out
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Real > >
class Vector
: public Array< Real, Device, Index, Allocator >
{
public:
   /**
    * \brief Type of elements stored in this vector.
    */
   using RealType = Real;

   /**
    * \brief Device where the vector is allocated.
    *
    * See \ref Devices::Host or \ref Devices::Cuda.
    */
   using DeviceType = Device;

   /**
    * \brief Type being used for the vector elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief Allocator type used for allocating this vector.
    *
    * See \ref Allocators::Cuda, \ref Allocators::CudaHost, \ref Allocators::CudaManaged, \ref Allocators::Host or \ref Allocators:Default.
    */
   using AllocatorType = Allocator;

   /**
    * \brief Compatible VectorView type.
    */
   using ViewType = VectorView< Real, Device, Index >;

   /**
    * \brief Compatible constant VectorView type.
    */
   using ConstViewType = VectorView< std::add_const_t< Real >, Device, Index >;

   /**
    * \brief A template which allows to quickly obtain a \ref Vector type with changed template parameters.
    */
   template< typename _Real,
             typename _Device = Device,
             typename _Index = Index,
             typename _Allocator = typename Allocators::Default< _Device >::template Allocator< _Real > >
   using Self = Vector< _Real, _Device, _Index, _Allocator >;


   // constructors and assignment operators inherited from the class Array
   using Array< Real, Device, Index, Allocator >::Array;
   using Array< Real, Device, Index, Allocator >::operator=;

   /**
    * \brief Constructs an empty array with zero size.
    */
   Vector() = default;

   /**
    * \brief Copy constructor (makes a deep copy).
    */
   explicit Vector( const Vector& ) = default;

   /**
    * \brief Copy constructor with a specific allocator (makes a deep copy).
    */
   explicit Vector( const Vector& vector, const AllocatorType& allocator );

   /**
    * \brief Default move constructor.
    */
   Vector( Vector&& ) = default;

   /**
    * \brief Constructor from expression template
    *
    * @param expression input expression template
    */
   template< typename VectorExpression,
             typename...,
             typename = std::enable_if_t< Expressions::HasEnabledExpressionTemplates< VectorExpression >::value && ! IsArrayType< VectorExpression >::value > >
   explicit Vector( const VectorExpression& expression );

   /**
    * \brief Copy-assignment operator for copying data from another vector.
    */
   Vector& operator=( const Vector& ) = default;

   /**
    * \brief Move-assignment operator for acquiring data from \e rvalues.
    */
   Vector& operator=( Vector&& ) = default;

   /**
    * \brief Returns a modifiable view of the vector.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   ViewType getView( IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Returns a non-modifiable view of the vector.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   ConstViewType getConstView( IndexType begin = 0, IndexType end = 0 ) const;

   /**
    * \brief Conversion operator to a modifiable view of the vector.
    */
   operator ViewType();

   /**
    * \brief Conversion operator to a non-modifiable view of the vector.
    */
   operator ConstViewType() const;

   /**
    * \brief Assigns a vector expression to this vector.
    *
    * The assignment is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector. If it evaluates to a vector
    * with a different size than this vector, this vector is reallocated to
    * match the size of the vector expression.
    *
    * \param expression The vector expression to be evaluated and assigned to
    *                   this vector.
    * \return Reference to this vector.
    */
   template< typename VectorExpression,
             typename...,
             typename = std::enable_if_t< Expressions::HasEnabledExpressionTemplates< VectorExpression >::value && ! IsArrayType< VectorExpression >::value > >
   Vector& operator=( const VectorExpression& expression );

   /**
    * \brief Assigns a value or an array - same as \ref Array::operator=.
    *
    * \return Reference to this vector.
    */
   // operator= from the base class should be hidden according to the C++14 standard,
   // although GCC does not do that - see https://stackoverflow.com/q/57322624
#if !defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11
   template< typename T,
             typename...,
             typename = std::enable_if_t< std::is_convertible< T, Real >::value || IsArrayType< T >::value > >
   Array< Real, Device, Index, Allocator >&
   operator=( const T& data )
   {
      return Array< Real, Device, Index, Allocator >::operator=(data);
   }
#endif

   /**
    * \brief Adds elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The addition is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator+=( const VectorExpression& expression );

   /**
    * \brief Subtracts elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The subtraction is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator-=( const VectorExpression& expression );

   /**
    * \brief Multiplies elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The multiplication is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator*=( const VectorExpression& expression );

   /**
    * \brief Divides elements of this vector and a vector expression and
    * stores the result in this vector.
    *
    * The division is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator/=( const VectorExpression& expression );

   /**
    * \brief Modulo assignment operator for vector and a vector expression.
    *
    * The division is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   Vector& operator%=( const VectorExpression& expression );
};

// Enable expression templates for Vector
namespace Expressions {
   template< typename Real, typename Device, typename Index, typename Allocator >
   struct HasEnabledExpressionTemplates< Vector< Real, Device, Index, Allocator > >
   : std::true_type
   {};
} // namespace Expressions

} // namespace Containers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Containers/Vector.hpp>
