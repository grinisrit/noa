// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <type_traits>
#include <algorithm>

#include <noa/3rdparty/tnl-noa/src/TNL/TypeTraits.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>

namespace noa::TNL {

//! \brief Value of the pi constant
static constexpr double pi = 3.14159265358979323846;

/**
 * \brief This function returns minimum of two numbers.
 *
 * GPU device code uses the functions defined in the CUDA's math_functions.h,
 * host uses the STL functions.
 */
template< typename T1,
          typename T2,
          typename ResultType = std::common_type_t< T1, T2 >,
          // enable_if is necessary to avoid ambiguity in vector expressions
          std::enable_if_t< std::is_arithmetic< T1 >::value && std::is_arithmetic< T2 >::value, bool > = true >
constexpr ResultType
min( const T1& a, const T2& b )
{
   // std::min is constexpr since C++14 so it can be reused directly
   return std::min( (ResultType) a, (ResultType) b );
}

/**
 * \brief This function returns minimum of a variadic number of inputs.
 *
 * The inputs are folded with the binary \e min function from the left to the
 * right.
 */
template< typename T1, typename T2, typename T3, typename... Ts >
constexpr typename std::common_type< T1, T2, T3, Ts... >::type
min( T1&& val1, T2&& val2, T3&& val3, Ts&&... vs )
{
   return min(
      min( std::forward< T1 >( val1 ), std::forward< T2 >( val2 ) ), std::forward< T3 >( val3 ), std::forward< Ts >( vs )... );
}

/**
 * \brief This function returns maximum of two numbers.
 *
 * GPU device code uses the functions defined in the CUDA's math_functions.h,
 * host uses the STL functions.
 */
template< typename T1,
          typename T2,
          typename ResultType = std::common_type_t< T1, T2 >,
          // enable_if is necessary to avoid ambiguity in vector expressions
          std::enable_if_t< std::is_arithmetic< T1 >::value && std::is_arithmetic< T2 >::value, bool > = true >
constexpr ResultType
max( const T1& a, const T2& b )
{
   // std::max is constexpr since C++14 so it can be reused directly
   return std::max( (ResultType) a, (ResultType) b );
}

/**
 * \brief This function returns minimum of a variadic number of inputs.
 *
 * The inputs are folded with the binary \e max function from the left to the
 * right.
 */
template< typename T1, typename T2, typename T3, typename... Ts >
constexpr typename std::common_type< T1, T2, T3, Ts... >::type
max( T1&& val1, T2&& val2, T3&& val3, Ts&&... vs )
{
   return max(
      max( std::forward< T1 >( val1 ), std::forward< T2 >( val2 ) ), std::forward< T3 >( val3 ), std::forward< Ts >( vs )... );
}

/**
 * \brief This function returns absolute value of given number \e n.
 */
template< class T, std::enable_if_t< std::is_arithmetic< T >::value && ! std::is_unsigned< T >::value, bool > = true >
__cuda_callable__
T
abs( const T& n )
{
#if defined( __CUDA_ARCH__ )
   if( std::is_integral< T >::value )
      return ::abs( n );
   else
      return ::fabs( n );
#else
   return std::abs( n );
#endif
}

/**
 * \brief This function returns the absolute value of given unsigned number \e n, i.e. \e n.
 */
template< class T, std::enable_if_t< std::is_unsigned< T >::value, bool > = true >
__cuda_callable__
T
abs( const T& n )
{
   return n;
}

/***
 * \brief This function returns argument of minimum of two numbers.
 */
template< typename T1, typename T2, typename ResultType = std::common_type_t< T1, T2 > >
constexpr ResultType
argMin( const T1& a, const T2& b )
{
   return ( a < b ) ? a : b;
}

/***
 * \brief This function returns argument of maximum of two numbers.
 */
template< typename T1, typename T2, typename ResultType = std::common_type_t< T1, T2 > >
constexpr ResultType
argMax( const T1& a, const T2& b )
{
   return ( a > b ) ? a : b;
}

/***
 * \brief This function returns argument of minimum of absolute values of two numbers.
 */
template< typename T1, typename T2, typename ResultType = std::common_type_t< T1, T2 > >
__cuda_callable__
ResultType
argAbsMin( const T1& a, const T2& b )
{
   return ( TNL::abs( a ) < TNL::abs( b ) ) ? a : b;
}

/***
 * \brief This function returns argument of maximum of absolute values of two numbers.
 */
template< typename T1, typename T2, typename ResultType = std::common_type_t< T1, T2 > >
__cuda_callable__
ResultType
argAbsMax( const T1& a, const T2& b )
{
   return ( TNL::abs( a ) > TNL::abs( b ) ) ? a : b;
}

/**
 * \brief This function returns the result of \e base to the power of \e exp.
 */
template< typename T1,
          typename T2,
          typename ResultType = std::common_type_t< T1, T2 >,
          // enable_if is necessary to avoid ambiguity in vector expressions
          std::enable_if_t< std::is_arithmetic< T1 >::value && std::is_arithmetic< T2 >::value, bool > = true >
__cuda_callable__
ResultType
pow( const T1& base, const T2& exp )
{
#if defined( __CUDA_ARCH__ )
   return ::pow( base, exp );
#else
   return std::pow( base, exp );
#endif
}

/**
 * \brief This function returns the base-e exponential of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
exp( const T& value ) -> decltype( std::exp( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::exp( value );
#else
   return std::exp( value );
#endif
}

/**
 * \brief This function returns the square the given \e value.
 */
template< typename T >
__cuda_callable__
std::enable_if_t< IsScalarType< T >::value, T >
sqr( const T& value )
{
   if constexpr( std::is_same_v< T, bool > )
      return value;
   else
      return value * value;
}

/**
 * \brief This function returns square root of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
sqrt( const T& value ) -> decltype( std::sqrt( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::sqrt( value );
#else
   return std::sqrt( value );
#endif
}

/**
 * \brief This function returns cubic root of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
cbrt( const T& value ) -> decltype( std::cbrt( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::cbrt( value );
#else
   return std::cbrt( value );
#endif
}

/**
 * \brief This function returns the natural logarithm of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
log( const T& value ) -> decltype( std::log( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::log( value );
#else
   return std::log( value );
#endif
}

/**
 * \brief This function returns the common logarithm of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
log10( const T& value ) -> decltype( std::log10( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::log10( value );
#else
   return std::log10( value );
#endif
}

/**
 * \brief This function returns the binary logarithm of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
log2( const T& value ) -> decltype( std::log2( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::log2( value );
#else
   return std::log2( value );
#endif
}

/**
 * \brief This function returns sine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
sin( const T& value ) -> decltype( std::sin( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::sin( value );
#else
   return std::sin( value );
#endif
}

/**
 * \brief This function returns cosine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
cos( const T& value ) -> decltype( std::cos( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::cos( value );
#else
   return std::cos( value );
#endif
}

/**
 * \brief This function returns tangent of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
tan( const T& value ) -> decltype( std::tan( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::tan( value );
#else
   return std::tan( value );
#endif
}

/**
 * \brief This function returns the arc sine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
asin( const T& value ) -> decltype( std::asin( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::asin( value );
#else
   return std::asin( value );
#endif
}

/**
 * \brief This function returns the arc cosine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
acos( const T& value ) -> decltype( std::acos( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::acos( value );
#else
   return std::acos( value );
#endif
}

/**
 * \brief This function returns the arc tangent of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
atan( const T& value ) -> decltype( std::atan( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::atan( value );
#else
   return std::atan( value );
#endif
}

/**
 * \brief This function returns the arc tangent of \e y/x using the signs of
 * arguments to determine the correct quadrant.
 */
template< typename T >
__cuda_callable__
auto
atan2( const T& y, const T& x ) -> decltype( std::atan2( y, x ) )
{
#if defined( __CUDA_ARCH__ )
   return ::atan2( y, x );
#else
   return std::atan2( y, x );
#endif
}

/**
 * \brief This function returns the hyperbolic sine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
sinh( const T& value ) -> decltype( std::sinh( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::sinh( value );
#else
   return std::sinh( value );
#endif
}

/**
 * \brief This function returns the hyperbolic cosine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
cosh( const T& value ) -> decltype( std::cosh( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::cosh( value );
#else
   return std::cosh( value );
#endif
}

/**
 * \brief This function returns the hyperbolic tangent of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
tanh( const T& value ) -> decltype( std::tanh( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::tanh( value );
#else
   return std::tanh( value );
#endif
}

/**
 * \brief This function returns the inverse hyperbolic sine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
asinh( const T& value ) -> decltype( std::asinh( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::asinh( value );
#else
   return std::asinh( value );
#endif
}

/**
 * \brief This function returns the inverse hyperbolic cosine of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
acosh( const T& value ) -> decltype( std::acosh( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::acosh( value );
#else
   return std::acosh( value );
#endif
}

/**
 * \brief This function returns the inverse hyperbolic tangent of the given \e value.
 */
template< typename T >
__cuda_callable__
auto
atanh( const T& value ) -> decltype( std::atanh( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::atanh( value );
#else
   return std::atanh( value );
#endif
}

/**
 * \brief This function returns largest integer value not greater than the given \e value.
 */
template< typename T >
__cuda_callable__
auto
floor( const T& value ) -> decltype( std::floor( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::floor( value );
#else
   return std::floor( value );
#endif
}

/**
 * \brief This function returns the smallest integer value not less than the given \e value.
 */
template< typename T >
__cuda_callable__
auto
ceil( const T& value ) -> decltype( std::ceil( value ) )
{
#if defined( __CUDA_ARCH__ )
   return ::ceil( value );
#else
   return std::ceil( value );
#endif
}

/**
 * \brief This function swaps values of two parameters.
 *
 * It assigns the value of \e a to the parameter \e b and vice versa.
 */
template< typename Type >
__cuda_callable__
constexpr void
swap( Type& a, Type& b )
{
   Type tmp( a );
   a = b;
   b = tmp;
}

/**
 * \brief This function represents the signum function.
 *
 * It extracts the sign of the number \e a. In other words, the signum function projects
 * negative numbers to value -1, positive numbers to value 1 and zero to value 0.
 */
template< class T,
          // enable_if is necessary to avoid ambiguity in vector expressions
          std::enable_if_t< ! HasSubscriptOperator< T >::value, bool > = true >
constexpr T
sign( const T& a )
{
   if( a < (T) 0 )
      return (T) -1;
   if( a == (T) 0 )
      return (T) 0;
   return (T) 1;
}

/**
 * \brief This function tests whether the given real number is small.
 *
 * It tests whether the number \e v is in \e tolerance, in other words, whether
 * \e v in absolute value is less then or equal to \e tolerance.
 * \param v Real number.
 * \param tolerance Critical value which is set to 0.00001 by defalt.
 */
template< typename Real >
constexpr bool
isSmall( const Real& v, const Real& tolerance = 1.0e-5 )
{
   return ( -tolerance <= v && v <= tolerance );
}

/**
 * \brief This function divides \e num by \e div and rounds up the result.
 *
 * \param num An integer considered as dividend.
 * \param div An integer considered as divisor.
 */
constexpr int
roundUpDivision( const int num, const int div )
{
   return num / div + static_cast< int >( num % div != 0 );
}

/**
 * \brief This function rounds up \e number to the nearest multiple of number \e multiple.
 *
 * \param number Integer we want to round.
 * \param multiple Integer.
 */
constexpr int
roundToMultiple( int number, int multiple )
{
   return multiple * ( number / multiple + static_cast< int >( number % multiple != 0 ) );
}

/**
 * \brief This function checks if \e x is an integral power of two.
 *
 * Returns \e true if \e x is a power of two. Otherwise returns \e false.
 * \param x Integer.
 */
constexpr bool
isPow2( int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

/**
 * \brief This function checks if \e x is an integral power of two.
 *
 * Returns \e true if \e x is a power of two. Otherwise returns \e false.
 * \param x Long integer.
 */
constexpr bool
isPow2( long int x )
{
   return ( ( x & ( x - 1 ) ) == 0 );
}

}  // namespace noa::TNL
