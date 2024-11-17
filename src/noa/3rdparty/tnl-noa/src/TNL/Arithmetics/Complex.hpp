// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Arithmetics {

template< typename Value >
__cuda_callable__
constexpr Complex< Value >::Complex()
{
   this->real_ = (ValueType) 0.0;
   this->imag_ = (ValueType) 0.0;
}

template< typename Value >
__cuda_callable__
constexpr Complex< Value >::Complex( const Value& re )
{
   this->real_ = re;
   this->imag_ = (ValueType) 0.0;
}

template< typename Value >
__cuda_callable__
constexpr Complex< Value >::Complex( const Value& re, const Value& im )
{
   this->real_ = re;
   this->imag_ = im;
}

template< typename Value >
__cuda_callable__
constexpr Complex< Value >::Complex( const Complex< Value >& c )
{
   this->real_ = c.real();
   this->imag_ = c.imag();
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
constexpr Complex< Value >::Complex( const Complex< Value_ >& c )
{
   this->real_ = c.real();
   this->imag_ = c.imag();
}

template< typename Value >
template< typename Value_ >
constexpr Complex< Value >::Complex( const std::complex< Value_ >& c )
{
   this->real_ = c.real();
   this->imag_ = c.imag();
}

template< typename Value >
__cuda_callable__
constexpr Complex< Value >&
Complex< Value >::operator=( const Value& v )
{
   this->real_ = v;
   this->imag_ = (ValueType) 0.0;
   return *this;
}

template< typename Value >
__cuda_callable__
constexpr Complex< Value >&
Complex< Value >::operator=( const Complex< Value >& v )
{
   this->real_ = v.real();
   this->imag_ = v.imag();
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
constexpr Complex< Value >&
Complex< Value >::operator=( const Value_& v )
{
   this->real_ = v;
   this->imag_ = (ValueType) 0.0;
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
constexpr Complex< Value >&
Complex< Value >::operator=( const Complex< Value_ >& v )
{
   this->real_ = v.real();
   this->imag_ = v.imag();
   return *this;
}

template< typename Value >
constexpr Complex< Value >&
Complex< Value >::operator=( const std::complex< Value >& c )
{
   this->real_ = c.real();
   this->imag_ = c.imag();
   return *this;
}

template< typename Value >
template< typename Value_ >
constexpr Complex< Value >&
Complex< Value >::operator=( const std::complex< Value_ >& c )
{
   this->real_ = c.real();
   this->imag_ = c.imag();
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator+=( const Value& v )
{
   this->real_ += v;
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator+=( const Complex< Value >& c )
{
   this->real_ += c.real();
   this->imag_ += c.imag();
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator+=( const Value_& v )
{
   this->real_ += v;
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator+=( const Complex< Value_ >& c )
{
   this->real_ += c.real();
   this->imag_ += c.imag();
   return *this;
}

template< typename Value >
Complex< Value >&
Complex< Value >::operator+=( const std::complex< Value >& c )
{
   this->real_ += c.real();
   this->imag_ += c.imag();
   return *this;
}

template< typename Value >
template< typename Value_ >
Complex< Value >&
Complex< Value >::operator+=( const std::complex< Value_ >& c )
{
   this->real_ += c.real();
   this->imag_ += c.imag();
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator-=( const Value& v )
{
   this->real_ -= v;
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator-=( const Complex< Value >& c )
{
   this->real_ -= c.real();
   this->imag_ -= c.imag();
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator-=( const Value_& v )
{
   this->real_ -= v;
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator-=( const Complex< Value_ >& c )
{
   this->real_ -= c.real();
   this->imag_ -= c.imag();
   return *this;
}

template< typename Value >
Complex< Value >&
Complex< Value >::operator-=( const std::complex< Value >& c )
{
   this->real_ -= c.real();
   this->imag_ -= c.imag();
   return *this;
}

template< typename Value >
template< typename Value_ >
Complex< Value >&
Complex< Value >::operator-=( const std::complex< Value_ >& c )
{
   this->real_ -= c.real();
   this->imag_ -= c.imag();
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator*=( const Value& v )
{
   this->real_ *= v;
   this->imag_ *= v;
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator*=( const Complex< Value >& c )
{
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() - this->imag() * c.imag();
   this->imag_ = real_ * c.imag() + this->imag() * c.real();
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator*=( const Value_& v )
{
   this->real_ *= v;
   this->imag_ *= v;
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator*=( const Complex< Value_ >& c )
{
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() - this->imag() * c.imag();
   this->imag_ = real_ * c.imag() + this->imag() * c.real();
   return *this;
}

template< typename Value >
Complex< Value >&
Complex< Value >::operator*=( const std::complex< Value >& c )
{
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() - this->imag() * c.imag();
   this->imag_ = real_ * c.imag() + this->imag() * c.real();
   return *this;
}

template< typename Value >
template< typename Value_ >
Complex< Value >&
Complex< Value >::operator*=( const std::complex< Value_ >& c )
{
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() - this->imag() * c.imag();
   this->imag_ = real_ * c.imag() + this->imag() * c.real();
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator/=( const Value& v )
{
   this->real_ /= v;
   this->imag_ /= v;
   return *this;
}

template< typename Value >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator/=( const Complex< Value >& c )
{
   // multiply by conjugate of c and divide by norm of c
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() + this->imag() * c.imag();
   this->imag_ = -real_ * c.imag() + this->imag() * c.real();
   *this /= norm( c );
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator/=( const Value_& v )
{
   this->real_ /= v;
   this->imag_ /= v;
   return *this;
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
Complex< Value >&
Complex< Value >::operator/=( const Complex< Value_ >& c )
{
   // multiply by conjugate of c and divide by norm of c
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() + this->imag() * c.imag();
   this->imag_ = -real_ * c.imag() + this->imag() * c.real();
   *this /= c.norm();
   return *this;
}

template< typename Value >
Complex< Value >&
Complex< Value >::operator/=( const std::complex< Value >& c )
{
   // multiply by conjugate of c and divide by norm of c
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() + this->imag() * c.imag();
   this->imag_ = -real_ * c.imag() + this->imag() * c.real();
   *this /= std::norm( c );
   return *this;
}

template< typename Value >
template< typename Value_ >
Complex< Value >&
Complex< Value >::operator/=( const std::complex< Value_ >& c )
{
   // multiply by conjugate of c and divide by norm of c
   Value real_ = this->real_;
   this->real_ = this->real() * c.real() + this->imag() * c.imag();
   this->imag_ = -real_ * c.imag() + this->imag() * c.real();
   *this /= c.norm();
   return *this;
}

template< typename Value >
__cuda_callable__
bool
Complex< Value >::operator==( const Value& v ) const
{
   return ( this->real() == v && this->imag_ == (ValueType) 0.0 );
}

template< typename Value >
__cuda_callable__
bool
Complex< Value >::operator==( const Complex< Value >& c ) const
{
   return ( this->real() == c.real() && this->imag() == c.imag() );
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
bool
Complex< Value >::operator==( const Value_& v ) const
{
   return ( this->real() == v && this->imag() == (ValueType) 0.0 );
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
bool
Complex< Value >::operator==( const Complex< Value_ >& c ) const
{
   return ( this->real() == c.real() && this->imag() == c.imag() );
}

template< typename Value >
bool
Complex< Value >::operator==( const std::complex< Value >& c ) const
{
   return ( this->real() == c.real() && this->imag() == c.imag() );
}

template< typename Value >
template< typename Value_ >
bool
Complex< Value >::operator==( const std::complex< Value_ >& c ) const
{
   return ( this->real() == c.real() && this->imag() == c.imag() );
}

template< typename Value >
__cuda_callable__
bool
Complex< Value >::operator!=( const Value& v ) const
{
   return ! this->operator==( v );
}

template< typename Value >
__cuda_callable__
bool
Complex< Value >::operator!=( const Complex< Value >& c ) const
{
   return ! this->operator==( c );
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
bool
Complex< Value >::operator!=( const Value_& v ) const
{
   return ! this->operator==( v );
}

template< typename Value >
template< typename Value_ >
__cuda_callable__
bool
Complex< Value >::operator!=( const Complex< Value_ >& c ) const
{
   return ! this->operator==( c );
}

template< typename Value >
bool
Complex< Value >::operator!=( const std::complex< Value >& c ) const
{
   return ! this->operator==( c );
}

template< typename Value >
template< typename Value_ >
bool
Complex< Value >::operator!=( const std::complex< Value_ >& c ) const
{
   return ! this->operator==( c );
}

template< typename Value >
__cuda_callable__
Complex< Value >
Complex< Value >::operator-() const
{
   return Complex( -this->real(), -this->imag() );
}

template< typename Value >
__cuda_callable__
Complex< Value >
Complex< Value >::operator+() const
{
   return *this;
}

template< typename Value >
__cuda_callable__
const Value&
Complex< Value >::real() const volatile
{
   return this->real_;
}

template< typename Value >
__cuda_callable__
const Value&
Complex< Value >::imag() const volatile
{
   return this->imag_;
}

template< typename Value >
__cuda_callable__
const Value&
Complex< Value >::real() const
{
   return this->real_;
}

template< typename Value >
__cuda_callable__
const Value&
Complex< Value >::imag() const
{
   return this->imag_;
}

template< typename Value >
__cuda_callable__
Value&
Complex< Value >::real() volatile
{
   return this->real_;
}

template< typename Value >
__cuda_callable__
Value&
Complex< Value >::imag() volatile
{
   return this->imag_;
}

template< typename Value >
__cuda_callable__
Value&
Complex< Value >::real()
{
   return this->real_;
}

template< typename Value >
__cuda_callable__
Value&
Complex< Value >::imag()
{
   return this->imag_;
}

////
// Addition operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator+( const Complex< Value >& c, const Value_& v )
{
   return Complex< Value >( c.real() + v, c.imag() );
}

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator+( const Value& v, const Complex< Value_ >& c )
{
   const Value add = v + c.real();
   return Complex< Value >( add, c.imag() );
}

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator+( const Complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() + c2.real(), c1.imag() + c2.imag() );
}

template< typename Value, typename Value_ >
Complex< Value >
operator+( const std::complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() + c2.real(), c1.imag() + c2.imag() );
}

template< typename Value, typename Value_ >
Complex< Value >
operator+( const Complex< Value >& c1, const std::complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() + c2.real(), c1.imag() + c2.imag() );
}

////
// Subtraction operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator-( const Complex< Value >& c, const Value_& v )
{
   return Complex< Value >( c.real() - v, c.imag() );
}

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator-( const Value& v, const Complex< Value_ >& c )
{
   return Complex< Value >( v - c.real(), c.imag() );
}

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator-( const Complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() - c2.real(), c1.imag() - c2.imag() );
}

template< typename Value, typename Value_ >
Complex< Value >
operator-( const std::complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() - c2.real(), c1.imag() - c2.imag() );
}

template< typename Value, typename Value_ >
Complex< Value >
operator-( const Complex< Value >& c1, const std::complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() - c2.real(), c1.imag() - c2.imag() );
}

////
// Multiplication operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator*( const Complex< Value >& c, const Value_& v )
{
   return Complex< Value >( c.real() * v, c.imag() * v );
}

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator*( const Value& v, const Complex< Value_ >& c )
{
   return Complex< Value >( v * c.real(), v * c.imag() );
}

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator*( const Complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() * c2.real() - c1.imag() * c2.imag(), c1.real() * c2.imag() + c1.imag() * c2.real() );
}

template< typename Value, typename Value_ >
Complex< Value >
operator*( const std::complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() * c2.real() - c1.imag() * c2.imag(), c1.real() * c2.imag() + c1.imag() * c2.real() );
}

template< typename Value, typename Value_ >
Complex< Value >
operator*( const Complex< Value >& c1, const std::complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() * c2.real() - c1.imag() * c2.imag(), c1.real() * c2.imag() + c1.imag() * c2.real() );
}

////
// Division operators
template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value_ >::value, Complex< Value > >
operator/( const Complex< Value >& c, const Value_& v )
{
   return Complex< Value >( c.real() / v, c.imag() / v );
}

template< typename Value, typename Value_ >
__cuda_callable__
std::enable_if_t< IsScalarType< Value >::value, Complex< Value > >
operator/( const Value& v, const Complex< Value_ >& c )
{
   return Complex< Value >( v * c.real(), -v * c.imag() ) / c.norm();
}

template< typename Value, typename Value_ >
__cuda_callable__
Complex< Value >
operator/( const Complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() * c2.real() + c1.imag() * c2.imag(), c1.imag() * c2.real() - c1.real() * c2.imag() )
        / norm( c2 );
}

template< typename Value, typename Value_ >
Complex< Value >
operator/( const std::complex< Value >& c1, const Complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() * c2.real() + c1.imag() * c2.imag(), c1.imag() * c2.real() - c1.real() * c2.imag() )
        / norm( c2 );
}

template< typename Value, typename Value_ >
Complex< Value >
operator/( const Complex< Value >& c1, const std::complex< Value_ >& c2 )
{
   return Complex< Value >( c1.real() * c2.real() + c1.imag() * c2.imag(), c1.imag() * c2.real() - c1.real() * c2.imag() )
        / norm( c2 );
}

template< typename Value >
__cuda_callable__
Value
norm( const Complex< Value >& c )
{
   return c.real() * c.real() + c.imag() * c.imag();
}

template< typename Value >
__cuda_callable__
Value
abs( const Complex< Value >& c )
{
   return TNL::sqrt( norm( c ) );
}

template< typename Value >
__cuda_callable__
Value
arg( const Complex< Value >& c )
{
   return TNL::atan2( c.imag(), c.real() );
}

template< typename Value >
__cuda_callable__
Complex< Value >
conj( const Complex< Value >& c )
{
   return Complex( c.real(), -c.imag() );
}

template< typename Value >
std::ostream&
operator<<( std::ostream& str, const Complex< Value >& c )
{
   str << "(" << c.real() << "," << c.imag() << ")";
   return str;
}

}  // namespace Arithmetics
}  // namespace noa::TNL
