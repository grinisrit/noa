// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <utility>

#include <noa/3rdparty/tnl-noa/src/TNL/Functional.h>
#include <noa/3rdparty/tnl-noa/src/TNL/TypeTraits.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/TypeTraits.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/ExpressionVariableType.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/StaticComparison.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/StaticVerticalOperations.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

template< typename T1, typename Operation >
struct StaticUnaryExpressionTemplate;

template< typename T1, typename Operation >
struct HasEnabledStaticExpressionTemplates< StaticUnaryExpressionTemplate< T1, Operation > > : std::true_type
{};

template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct StaticBinaryExpressionTemplate;

template< typename T1, typename T2, typename Operation, ExpressionVariableType T1Type, ExpressionVariableType T2Type >
struct HasEnabledStaticExpressionTemplates< StaticBinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};

////
// Static binary expression template
template< typename T1, typename T2, typename Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using VectorOperandType = T1;
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T1 >::value,
                  "Left-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( IsStaticArrayType< T2 >::value,
                  "Right-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( HasEnabledStaticExpressionTemplates< T1 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the "
                  "left operand." );
   static_assert( HasEnabledStaticExpressionTemplates< T2 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the "
                  "right operand." );
   static_assert( T1::getSize() == T2::getSize(), "Attempt to mix static operands with different sizes." );

   static constexpr int
   getSize()
   {
      return T1::getSize();
   }

   constexpr StaticBinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a ), op2( b ) {}

   constexpr RealType
   operator[]( const int i ) const
   {
      return Operation{}( op1[ i ], op2[ i ] );
   }

   constexpr RealType
   x() const
   {
      return ( *this )[ 0 ];
   }

   constexpr RealType
   y() const
   {
      return ( *this )[ 1 ];
   }

   constexpr RealType
   z() const
   {
      return ( *this )[ 2 ];
   }

protected:
   typename OperandMemberType< T1 >::type op1;
   typename OperandMemberType< T2 >::type op2;
};

template< typename T1, typename T2, typename Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using VectorOperandType = T1;
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >() ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T1 >::value,
                  "Left-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( HasEnabledStaticExpressionTemplates< T1 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the "
                  "left operand." );

   static constexpr int
   getSize()
   {
      return T1::getSize();
   }

   constexpr StaticBinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a ), op2( b ) {}

   constexpr RealType
   operator[]( const int i ) const
   {
      return Operation{}( op1[ i ], op2 );
   }

   constexpr RealType
   x() const
   {
      return ( *this )[ 0 ];
   }

   constexpr RealType
   y() const
   {
      return ( *this )[ 1 ];
   }

   constexpr RealType
   z() const
   {
      return ( *this )[ 2 ];
   }

protected:
   typename OperandMemberType< T1 >::type op1;
   typename OperandMemberType< T2 >::type op2;
};

template< typename T1, typename T2, typename Operation >
struct StaticBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using VectorOperandType = T2;
   using RealType = decltype( Operation{}( std::declval< T1 >(), std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T2 >::value,
                  "Right-hand side operand of static expression is not static, i.e. based on static vector." );
   static_assert( HasEnabledStaticExpressionTemplates< T2 >::value,
                  "Invalid operand in static binary expression templates - static expression templates are not enabled for the "
                  "right operand." );

   static constexpr int
   getSize()
   {
      return T2::getSize();
   }

   constexpr StaticBinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a ), op2( b ) {}

   constexpr RealType
   operator[]( const int i ) const
   {
      return Operation{}( op1, op2[ i ] );
   }

   constexpr RealType
   x() const
   {
      return ( *this )[ 0 ];
   }

   constexpr RealType
   y() const
   {
      return ( *this )[ 1 ];
   }

   constexpr RealType
   z() const
   {
      return ( *this )[ 2 ];
   }

protected:
   typename OperandMemberType< T1 >::type op1;
   typename OperandMemberType< T2 >::type op2;
};

////
// Static unary expression template
template< typename T1, typename Operation >
struct StaticUnaryExpressionTemplate
{
   using VectorOperandType = T1;
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ] ) );
   using ValueType = RealType;

   static_assert( IsStaticArrayType< T1 >::value,
                  "The operand of static expression is not static, i.e. based on static vector." );
   static_assert(
      HasEnabledStaticExpressionTemplates< T1 >::value,
      "Invalid operand in static unary expression templates - static expression templates are not enabled for the operand." );

   static constexpr int
   getSize()
   {
      return T1::getSize();
   }

   constexpr StaticUnaryExpressionTemplate( const T1& a ) : operand( a ) {}

   constexpr RealType
   operator[]( const int i ) const
   {
      return Operation{}( operand[ i ] );
   }

   constexpr RealType
   x() const
   {
      return ( *this )[ 0 ];
   }

   constexpr RealType
   y() const
   {
      return ( *this )[ 1 ];
   }

   constexpr RealType
   z() const
   {
      return ( *this )[ 2 ];
   }

protected:
   typename OperandMemberType< T1 >::type operand;
};

#ifndef DOXYGEN_ONLY

   #define TNL_MAKE_STATIC_UNARY_EXPRESSION( marker, fname, functor )                            \
      template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true > \
      marker auto fname( const ET1& a )                                                          \
      {                                                                                          \
         return StaticUnaryExpressionTemplate< ET1, functor >( a );                              \
      }

   #define TNL_MAKE_STATIC_BINARY_EXPRESSION( marker, fname, functor )                                               \
      template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true > \
      marker auto fname( const ET1& a, const ET2& b )                                                                \
      {                                                                                                              \
         return StaticBinaryExpressionTemplate< ET1, ET2, functor >( a, b );                                         \
      }

TNL_MAKE_STATIC_BINARY_EXPRESSION( constexpr, operator+, TNL::Plus )
TNL_MAKE_STATIC_BINARY_EXPRESSION( constexpr, operator-, TNL::Minus )
TNL_MAKE_STATIC_BINARY_EXPRESSION( constexpr, operator*, TNL::Multiplies )
TNL_MAKE_STATIC_BINARY_EXPRESSION( constexpr, operator/, TNL::Divides )
TNL_MAKE_STATIC_BINARY_EXPRESSION( constexpr, operator%, TNL::Modulus )
TNL_MAKE_STATIC_BINARY_EXPRESSION( constexpr, min, TNL::Min )
TNL_MAKE_STATIC_BINARY_EXPRESSION( constexpr, max, TNL::Max )

TNL_MAKE_STATIC_UNARY_EXPRESSION( constexpr, operator+, TNL::UnaryPlus )
TNL_MAKE_STATIC_UNARY_EXPRESSION( constexpr, operator-, TNL::UnaryMinus )
TNL_MAKE_STATIC_UNARY_EXPRESSION( constexpr, operator!, TNL::LogicalNot )
TNL_MAKE_STATIC_UNARY_EXPRESSION( constexpr, operator~, TNL::BitNot )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, abs, TNL::Abs )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, exp, TNL::Exp )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, sqrt, TNL::Sqrt )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, cbrt, TNL::Cbrt )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, log, TNL::Log )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, log10, TNL::Log10 )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, log2, TNL::Log2 )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, sin, TNL::Sin )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, cos, TNL::Cos )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, tan, TNL::Tan )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, asin, TNL::Asin )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, acos, TNL::Acos )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, atan, TNL::Atan )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, sinh, TNL::Sinh )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, cosh, TNL::Cosh )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, tanh, TNL::Tanh )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, asinh, TNL::Asinh )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, acosh, TNL::Acosh )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, atanh, TNL::Atanh )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, floor, TNL::Floor )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, ceil, TNL::Ceil )
TNL_MAKE_STATIC_UNARY_EXPRESSION( __cuda_callable__, sign, TNL::Sign )

   #undef TNL_MAKE_STATIC_UNARY_EXPRESSION
   #undef TNL_MAKE_STATIC_BINARY_EXPRESSION

////
// Pow
template< typename ET1, typename Real, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
__cuda_callable__
auto
pow( const ET1& a, const Real& exp )
{
   return StaticBinaryExpressionTemplate< ET1, Real, Pow >( a, exp );
}

////
// Cast
template< typename ResultType, typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
cast( const ET1& a )
{
   using CastOperation = typename Cast< ResultType >::Operation;
   return StaticUnaryExpressionTemplate< ET1, CastOperation >( a );
}

////
// Comparison operator ==
template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr bool
operator==( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::EQ( a, b );
}

////
// Comparison operator !=
template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr bool
operator!=( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::NE( a, b );
}

////
// Comparison operator <
template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr bool
operator<( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::LT( a, b );
}

////
// Comparison operator <=
template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr bool
operator<=( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::LE( a, b );
}

////
// Comparison operator >
template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr bool
operator>( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::GT( a, b );
}

////
// Comparison operator >=
template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr bool
operator>=( const ET1& a, const ET2& b )
{
   return StaticComparison< ET1, ET2 >::GE( a, b );
}

////
// Scalar product
template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr auto
operator,( const ET1& a, const ET2& b )
{
   return StaticExpressionSum( a * b );
}

template< typename ET1, typename ET2, typename..., EnableIfStaticBinaryExpression_t< ET1, ET2, bool > = true >
constexpr auto
dot( const ET1& a, const ET2& b )
{
   return ( a, b );
}

////
// Vertical operations
template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
min( const ET1& a )
{
   return StaticExpressionMin( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
argMin( const ET1& a )
{
   return StaticExpressionArgMin( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
max( const ET1& a )
{
   return StaticExpressionMax( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
argMax( const ET1& a )
{
   return StaticExpressionArgMax( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
sum( const ET1& a )
{
   return StaticExpressionSum( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
maxNorm( const ET1& a )
{
   return max( abs( a ) );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
l1Norm( const ET1& a )
{
   return sum( abs( a ) );
}

template< typename ET1,
          typename...,
          EnableIfStaticUnaryExpression_t< ET1, bool > = true,
          std::enable_if_t< ( ET1::getSize() > 1 ), bool > = true >
__cuda_callable__
auto
l2Norm( const ET1& a )
{
   using TNL::sqrt;
   return sqrt( sum( a * a ) );
}

template< typename ET1,
          typename...,
          EnableIfStaticUnaryExpression_t< ET1, bool > = true,
          std::enable_if_t< ET1::getSize() == 1, bool > = true >
__cuda_callable__
auto
l2Norm( const ET1& a )
{
   // avoid sqrt for 1D vectors (l1 and l2 norms are identical in 1D)
   return l1Norm( a );
}

template< typename ET1,
          typename Real,
          typename...,
          EnableIfStaticUnaryExpression_t< ET1, bool > = true,
          std::enable_if_t< ( ET1::getSize() > 1 ), bool > = true >
__cuda_callable__
auto
lpNorm( const ET1& a, const Real& p )
   // since (1.0 / p) has type double, TNL::pow returns double
   -> double
//-> RemoveET< decltype(pow( StaticExpressionLpNorm( a, p ), 1.0 / p )) >
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   using TNL::pow;
   return pow( sum( pow( abs( a ), p ) ), 1.0 / p );
}

template< typename ET1,
          typename Real,
          typename...,
          EnableIfStaticUnaryExpression_t< ET1, bool > = true,
          std::enable_if_t< ET1::getSize() == 1, bool > = true >
__cuda_callable__
auto
lpNorm( const ET1& a, const Real& p )
{
   // avoid sqrt and pow for 1D vectors (all lp norms are identical in 1D)
   return l1Norm( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
product( const ET1& a )
{
   return StaticExpressionProduct( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
logicalAnd( const ET1& a )
{
   return StaticExpressionLogicalAnd( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
logicalOr( const ET1& a )
{
   return StaticExpressionLogicalOr( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
binaryAnd( const ET1& a )
{
   return StaticExpressionBinaryAnd( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
binaryOr( const ET1& a )
{
   return StaticExpressionBinaryOr( a );
}

template< typename ET1, typename..., EnableIfStaticUnaryExpression_t< ET1, bool > = true >
constexpr auto
binaryXor( const ET1& a )
{
   return StaticExpressionBinaryXor( a );
}

#endif  // DOXYGEN_ONLY

////
// Output stream
template< typename T1, typename T2, typename Operation >
std::ostream&
operator<<( std::ostream& str, const StaticBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}

template< typename T, typename Operation >
std::ostream&
operator<<( std::ostream& str, const StaticUnaryExpressionTemplate< T, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression[ i ] << ", ";
   str << expression[ expression.getSize() - 1 ] << " ]";
   return str;
}

}  // namespace Expressions

// Make all operators visible in the TNL::Containers namespace to be considered
// even for StaticVector
using Expressions::operator!;
using Expressions::operator~;
using Expressions::operator+;
using Expressions::operator-;
using Expressions::operator*;
using Expressions::operator/;
using Expressions::operator%;
using Expressions::operator, ;
using Expressions::operator==;
using Expressions::operator!=;
using Expressions::operator<;
using Expressions::operator<=;
using Expressions::operator>;
using Expressions::operator>=;

// Make all functions visible in the TNL::Containers namespace
using Expressions::abs;
using Expressions::acos;
using Expressions::acosh;
using Expressions::argMax;
using Expressions::argMin;
using Expressions::asin;
using Expressions::asinh;
using Expressions::atan;
using Expressions::atanh;
using Expressions::binaryAnd;
using Expressions::binaryOr;
using Expressions::cast;
using Expressions::cbrt;
using Expressions::ceil;
using Expressions::cos;
using Expressions::cosh;
using Expressions::dot;
using Expressions::exp;
using Expressions::floor;
using Expressions::l1Norm;
using Expressions::l2Norm;
using Expressions::log;
using Expressions::log10;
using Expressions::log2;
using Expressions::logicalAnd;
using Expressions::logicalOr;
using Expressions::lpNorm;
using Expressions::max;
using Expressions::maxNorm;
using Expressions::min;
using Expressions::pow;
using Expressions::product;
using Expressions::sign;
using Expressions::sin;
using Expressions::sinh;
using Expressions::sqrt;
using Expressions::sum;
using Expressions::tan;
using Expressions::tanh;

}  // namespace Containers

// Make all functions visible in the main TNL namespace
using Containers::abs;
using Containers::acos;
using Containers::acosh;
using Containers::argMax;
using Containers::argMin;
using Containers::asin;
using Containers::asinh;
using Containers::atan;
using Containers::atanh;
using Containers::binaryAnd;
using Containers::binaryOr;
using Containers::cast;
using Containers::cbrt;
using Containers::ceil;
using Containers::cos;
using Containers::cosh;
using Containers::dot;
using Containers::exp;
using Containers::floor;
using Containers::l1Norm;
using Containers::l2Norm;
using Containers::log;
using Containers::log10;
using Containers::log2;
using Containers::logicalAnd;
using Containers::logicalOr;
using Containers::lpNorm;
using Containers::max;
using Containers::maxNorm;
using Containers::min;
using Containers::pow;
using Containers::product;
using Containers::sign;
using Containers::sin;
using Containers::sinh;
using Containers::sqrt;
using Containers::sum;
using Containers::tan;
using Containers::tanh;

////
// Evaluation with reduction
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
__cuda_callable__
Result
evaluateAndReduce( Vector& lhs,
                   const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
                   const Reduction& reduction,
                   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      result = reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
__cuda_callable__
Result
evaluateAndReduce( Vector& lhs,
                   const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
                   const Reduction& reduction,
                   const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ )
      result = reduction( result, lhs[ i ] = expression[ i ] );
   return result;
}

////
// Addition with reduction
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
__cuda_callable__
Result
addAndReduce( Vector& lhs,
              const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
              const Reduction& reduction,
              const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, aux );
   }
   return result;
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
__cuda_callable__
Result
addAndReduce( Vector& lhs,
              const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
              const Reduction& reduction,
              const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, aux );
   }
   return result;
}

////
// Addition with reduction of abs
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
__cuda_callable__
Result
addAndReduceAbs( Vector& lhs,
                 const Containers::Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expression,
                 const Reduction& reduction,
                 const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, TNL::abs( aux ) );
   }
   return result;
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
__cuda_callable__
Result
addAndReduceAbs( Vector& lhs,
                 const Containers::Expressions::StaticUnaryExpressionTemplate< T1, Operation >& expression,
                 const Reduction& reduction,
                 const Result& zero )
{
   Result result( zero );
   for( int i = 0; i < Vector::getSize(); i++ ) {
      const Result aux = expression[ i ];
      lhs[ i ] += aux;
      result = reduction( result, TNL::abs( aux ) );
   }
   return result;
}

}  // namespace noa::TNL
