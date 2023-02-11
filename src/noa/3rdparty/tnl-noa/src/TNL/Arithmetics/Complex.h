// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>

namespace noa::TNL {
namespace Arithmetics {
/**
 * \brief Implementation of complex types.
 *
 * Warning: Only basic algebraic operations like addition, subtraction, multiplication and division are implemented currently.
 *
 * \tparam Value represents arithmetics of real numbers used for the construction of the complex number.
 */
template< typename Value = double >
struct Complex
{
   using ValueType = Value;

   __cuda_callable__
   constexpr Complex();

   __cuda_callable__
   constexpr Complex( const Value& re );

   __cuda_callable__
   constexpr Complex( const Value& re, const Value& im );

   __cuda_callable__
   constexpr Complex( const Complex< Value >& c );

   template< typename Value_ >
   __cuda_callable__
   constexpr Complex( const Complex< Value_ >& c );

   template< typename Value_ >
   constexpr Complex( const std::complex< Value_ >& c );

   __cuda_callable__
   constexpr Complex&
   operator=( const Value& v );

   __cuda_callable__
   constexpr Complex&
   operator=( const Complex< Value >& c );

   template< typename Value_ >
   __cuda_callable__
   constexpr Complex&
   operator=( const Value_& v );

   template< typename Value_ >
   __cuda_callable__
   constexpr Complex&
   operator=( const Complex< Value_ >& c );

   constexpr Complex&
   operator=( const std::complex< Value >& c );

   template< typename Value_ >
   constexpr Complex&
   operator=( const std::complex< Value_ >& c );

   __cuda_callable__
   Complex&
   operator+=( const Value& v );

   __cuda_callable__
   Complex&
   operator+=( const Complex< Value >& c );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator+=( const Value_& v );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator+=( const Complex< Value_ >& c );

   Complex&
   operator+=( const std::complex< Value >& c );

   template< typename Value_ >
   Complex&
   operator+=( const std::complex< Value_ >& c );

   __cuda_callable__
   Complex&
   operator-=( const Value& v );

   __cuda_callable__
   Complex&
   operator-=( const Complex< Value >& c );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator-=( const Value_& v );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator-=( const Complex< Value_ >& c );

   Complex&
   operator-=( const std::complex< Value >& c );

   template< typename Value_ >
   Complex&
   operator-=( const std::complex< Value_ >& c );

   __cuda_callable__
   Complex&
   operator*=( const Value& v );

   __cuda_callable__
   Complex&
   operator*=( const Complex< Value >& c );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator*=( const Value_& v );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator*=( const Complex< Value_ >& c );

   Complex&
   operator*=( const std::complex< Value >& c );

   template< typename Value_ >
   Complex&
   operator*=( const std::complex< Value_ >& c );

   __cuda_callable__
   Complex&
   operator/=( const Value& v );

   __cuda_callable__
   Complex&
   operator/=( const Complex< Value >& c );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator/=( const Value_& v );

   template< typename Value_ >
   __cuda_callable__
   Complex&
   operator/=( const Complex< Value_ >& c );

   Complex&
   operator/=( const std::complex< Value >& c );

   template< typename Value_ >
   Complex&
   operator/=( const std::complex< Value_ >& c );

   __cuda_callable__
   bool
   operator==( const Value& v ) const;

   __cuda_callable__
   bool
   operator==( const Complex< Value >& c ) const;

   template< typename Value_ >
   __cuda_callable__
   bool
   operator==( const Value_& v ) const;

   template< typename Value_ >
   __cuda_callable__
   bool
   operator==( const Complex< Value_ >& c ) const;

   bool
   operator==( const std::complex< Value >& c ) const;

   template< typename Value_ >
   bool
   operator==( const std::complex< Value_ >& c ) const;

   __cuda_callable__
   bool
   operator!=( const Value& v ) const;

   __cuda_callable__
   bool
   operator!=( const Complex< Value >& c ) const;

   template< typename Value_ >
   __cuda_callable__
   bool
   operator!=( const Value_& v ) const;

   template< typename Value_ >
   __cuda_callable__
   bool
   operator!=( const Complex< Value_ >& c ) const;

   bool
   operator!=( const std::complex< Value >& c ) const;

   template< typename Value_ >
   bool
   operator!=( const std::complex< Value_ >& c ) const;

   __cuda_callable__
   Complex
   operator-() const;

   __cuda_callable__
   Complex
   operator+() const;

   __cuda_callable__
   const Value&
   real() const volatile;

   __cuda_callable__
   const Value&
   imag() const volatile;

   __cuda_callable__
   const Value&
   real() const;

   __cuda_callable__
   const Value&
   imag() const;

   __cuda_callable__
   Value&
   real() volatile;

   __cuda_callable__
   Value&
   imag() volatile;

   __cuda_callable__
   Value&
   real();

   __cuda_callable__
   Value&
   imag();

protected:
   Value real_, imag_;
};

////
// Addition operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator+( const Complex< Value >& c, const Value_& v );

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator+( const Value& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator+( const Complex< Value >& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
Complex< Value >
operator+( const std::complex< Value >& c1, const Complex< Value >& c2 );

template< typename Value, typename Value_ >
Complex< Value >
operator+( const Complex< Value >& c1, const std::complex< Value_ >& c2 );

////
// Subtraction operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator-( const Complex< Value >& c, const Value_& v );

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator-( const Value& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator-( const Complex< Value >& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
Complex< Value >
operator-( const std::complex< Value >& c1, const Complex< Value >& c2 );

template< typename Value, typename Value_ >
Complex< Value >
operator-( const Complex< Value >& c1, const std::complex< Value_ >& c2 );

////
// Multiplication operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator*( const Complex< Value >& c, const Value_& v );

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator*( const Value& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator*( const Complex< Value >& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
Complex< Value >
operator*( const std::complex< Value >& c1, const Complex< Value >& c2 );

template< typename Value, typename Value_ >
Complex< Value >
operator*( const Complex< Value >& c1, const std::complex< Value_ >& c2 );

////
// Division operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator/( const Complex< Value >& c, const Value_& v );

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator/( const Value& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator/( const Complex< Value >& v, const Complex< Value_ >& c );

template< typename Value, typename Value_ >
Complex< Value >
operator/( const std::complex< Value >& c1, const Complex< Value >& c2 );

template< typename Value, typename Value_ >
Complex< Value >
operator/( const Complex< Value >& c1, const std::complex< Value_ >& c2 );

template< typename Value >
__cuda_callable__
Value
norm( const Complex< Value >& c );

template< typename Value >
__cuda_callable__
Value
abs( const Complex< Value >& c );

template< typename Value >
__cuda_callable__
Value
arg( const Complex< Value >& c );

template< typename Value >
__cuda_callable__
Complex< Value >
conj( const Complex< Value >& c );

template< typename Value >
std::ostream&
operator<<( std::ostream& str, const Complex< Value >& c );

}  // namespace Arithmetics
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Arithmetics/Complex.hpp>
