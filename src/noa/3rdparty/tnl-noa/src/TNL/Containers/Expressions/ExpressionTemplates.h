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
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/Comparison.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/reduce.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

template< typename T1, typename Operation >
struct UnaryExpressionTemplate;

template< typename T1, typename Operation >
struct HasEnabledExpressionTemplates< UnaryExpressionTemplate< T1, Operation > > : std::true_type
{};

template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct BinaryExpressionTemplate;

template< typename T1, typename T2, typename Operation, ExpressionVariableType T1Type, ExpressionVariableType T2Type >
struct HasEnabledExpressionTemplates< BinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > > : std::true_type
{};

////
// Non-static binary expression template
template< typename T1, typename T2, typename Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using ConstViewType = BinaryExpressionTemplate;

   static_assert(
      HasEnabledExpressionTemplates< T1 >::value,
      "Invalid operand in binary expression templates - expression templates are not enabled for the left operand." );
   static_assert(
      HasEnabledExpressionTemplates< T2 >::value,
      "Invalid operand in binary expression templates - expression templates are not enabled for the right operand." );
   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value,
                  "Attempt to mix operands which have different DeviceType." );

   BinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a.getConstView() ), op2( b.getConstView() )
   {
      TNL_ASSERT_EQ( op1.getSize(), op2.getSize(), "Attempt to mix operands with different sizes." );
   }

   RealType
   getElement( const IndexType i ) const
   {
      return Operation{}( op1.getElement( i ), op2.getElement( i ) );
   }

   __cuda_callable__
   RealType
   operator[]( const IndexType i ) const
   {
      return Operation{}( op1[ i ], op2[ i ] );
   }

   __cuda_callable__
   RealType
   operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType
   getSize() const
   {
      return op1.getSize();
   }

   ConstViewType
   getConstView() const
   {
      return *this;
   }

protected:
   const typename T1::ConstViewType op1;
   const typename T2::ConstViewType op2;
};

template< typename T1, typename T2, typename Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >() ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using ConstViewType = BinaryExpressionTemplate;

   static_assert(
      HasEnabledExpressionTemplates< T1 >::value,
      "Invalid operand in binary expression templates - expression templates are not enabled for the left operand." );

   BinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a.getConstView() ), op2( b ) {}

   RealType
   getElement( const IndexType i ) const
   {
      return Operation{}( op1.getElement( i ), op2 );
   }

   __cuda_callable__
   RealType
   operator[]( const IndexType i ) const
   {
      return Operation{}( op1[ i ], op2 );
   }

   __cuda_callable__
   RealType
   operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType
   getSize() const
   {
      return op1.getSize();
   }

   ConstViewType
   getConstView() const
   {
      return *this;
   }

protected:
   const typename T1::ConstViewType op1;
   const T2 op2;
};

template< typename T1, typename T2, typename Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >(), std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using ConstViewType = BinaryExpressionTemplate;

   static_assert(
      HasEnabledExpressionTemplates< T2 >::value,
      "Invalid operand in binary expression templates - expression templates are not enabled for the right operand." );

   BinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a ), op2( b.getConstView() ) {}

   RealType
   getElement( const IndexType i ) const
   {
      return Operation{}( op1, op2.getElement( i ) );
   }

   __cuda_callable__
   RealType
   operator[]( const IndexType i ) const
   {
      return Operation{}( op1, op2[ i ] );
   }

   __cuda_callable__
   RealType
   operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType
   getSize() const
   {
      return op2.getSize();
   }

   ConstViewType
   getConstView() const
   {
      return *this;
   }

protected:
   const T1 op1;
   const typename T2::ConstViewType op2;
};

////
// Non-static unary expression template
template< typename T1, typename Operation >
struct UnaryExpressionTemplate
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using ConstViewType = UnaryExpressionTemplate;

   static_assert( HasEnabledExpressionTemplates< T1 >::value,
                  "Invalid operand in unary expression templates - expression templates are not enabled for the operand." );

   UnaryExpressionTemplate( const T1& a ) : operand( a.getConstView() ) {}

   RealType
   getElement( const IndexType i ) const
   {
      return Operation{}( operand.getElement( i ) );
   }

   __cuda_callable__
   RealType
   operator[]( const IndexType i ) const
   {
      return Operation{}( operand[ i ] );
   }

   __cuda_callable__
   RealType
   operator()( const IndexType i ) const
   {
      return operator[]( i );
   }

   __cuda_callable__
   IndexType
   getSize() const
   {
      return operand.getSize();
   }

   ConstViewType
   getConstView() const
   {
      return *this;
   }

protected:
   const typename T1::ConstViewType operand;
};

#ifndef DOXYGEN_ONLY

   #define TNL_MAKE_UNARY_EXPRESSION( fname, functor )                                     \
      template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true > \
      auto fname( const ET1& a )                                                           \
      {                                                                                    \
         return UnaryExpressionTemplate< ET1, functor >( a );                              \
      }

   #define TNL_MAKE_BINARY_EXPRESSION( fname, functor )                                                        \
      template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true > \
      auto fname( const ET1& a, const ET2& b )                                                                 \
      {                                                                                                        \
         return BinaryExpressionTemplate< ET1, ET2, functor >( a, b );                                         \
      }

TNL_MAKE_BINARY_EXPRESSION( operator+, TNL::Plus )
TNL_MAKE_BINARY_EXPRESSION( operator-, TNL::Minus )
TNL_MAKE_BINARY_EXPRESSION( operator*, TNL::Multiplies )
TNL_MAKE_BINARY_EXPRESSION( operator/, TNL::Divides )
TNL_MAKE_BINARY_EXPRESSION( operator%, TNL::Modulus )
TNL_MAKE_BINARY_EXPRESSION( min, TNL::Min )
TNL_MAKE_BINARY_EXPRESSION( max, TNL::Max )

TNL_MAKE_UNARY_EXPRESSION( operator+, TNL::UnaryPlus )
TNL_MAKE_UNARY_EXPRESSION( operator-, TNL::UnaryMinus )
TNL_MAKE_UNARY_EXPRESSION( operator!, TNL::LogicalNot )
TNL_MAKE_UNARY_EXPRESSION( operator~, TNL::BitNot )
TNL_MAKE_UNARY_EXPRESSION( abs, TNL::Abs )
TNL_MAKE_UNARY_EXPRESSION( exp, TNL::Exp )
TNL_MAKE_UNARY_EXPRESSION( sqr, TNL::Sqr )
TNL_MAKE_UNARY_EXPRESSION( sqrt, TNL::Sqrt )
TNL_MAKE_UNARY_EXPRESSION( cbrt, TNL::Cbrt )
TNL_MAKE_UNARY_EXPRESSION( log, TNL::Log )
TNL_MAKE_UNARY_EXPRESSION( log10, TNL::Log10 )
TNL_MAKE_UNARY_EXPRESSION( log2, TNL::Log2 )
TNL_MAKE_UNARY_EXPRESSION( sin, TNL::Sin )
TNL_MAKE_UNARY_EXPRESSION( cos, TNL::Cos )
TNL_MAKE_UNARY_EXPRESSION( tan, TNL::Tan )
TNL_MAKE_UNARY_EXPRESSION( asin, TNL::Asin )
TNL_MAKE_UNARY_EXPRESSION( acos, TNL::Acos )
TNL_MAKE_UNARY_EXPRESSION( atan, TNL::Atan )
TNL_MAKE_UNARY_EXPRESSION( sinh, TNL::Sinh )
TNL_MAKE_UNARY_EXPRESSION( cosh, TNL::Cosh )
TNL_MAKE_UNARY_EXPRESSION( tanh, TNL::Tanh )
TNL_MAKE_UNARY_EXPRESSION( asinh, TNL::Asinh )
TNL_MAKE_UNARY_EXPRESSION( acosh, TNL::Acosh )
TNL_MAKE_UNARY_EXPRESSION( atanh, TNL::Atanh )
TNL_MAKE_UNARY_EXPRESSION( floor, TNL::Floor )
TNL_MAKE_UNARY_EXPRESSION( ceil, TNL::Ceil )
TNL_MAKE_UNARY_EXPRESSION( sign, TNL::Sign )

   #undef TNL_MAKE_UNARY_EXPRESSION
   #undef TNL_MAKE_BINARY_EXPRESSION

////
// Pow
template< typename ET1, typename Real, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
pow( const ET1& a, const Real& exp )
{
   return BinaryExpressionTemplate< ET1, Real, Pow >( a, exp );
}

////
// Cast
template< typename ResultType, typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
cast( const ET1& a )
{
   using CastOperation = typename Cast< ResultType >::Operation;
   return UnaryExpressionTemplate< ET1, CastOperation >( a );
}

////
// Comparison operator ==
template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator==( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::EQ( a, b );
}

////
// Comparison operator !=
template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator!=( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::NE( a, b );
}

////
// Comparison operator <
template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator<( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::LT( a, b );
}

////
// Comparison operator <=
template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator<=( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::LE( a, b );
}

////
// Comparison operator >
template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator>( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::GT( a, b );
}

////
// Comparison operator >=
template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator>=( const ET1& a, const ET2& b )
{
   return Comparison< ET1, ET2 >::GE( a, b );
}

////
// Scalar product
template< typename ET1, typename ET2,
          typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
auto
operator,( const ET1& a, const ET2& b )
{
   return Algorithms::reduce( a * b, TNL::Plus{} );
}

template< typename ET1, typename ET2, typename..., EnableIfBinaryExpression_t< ET1, ET2, bool > = true >
auto
dot( const ET1& a, const ET2& b )
{
   return ( a, b );
}

////
// Vertical operations
template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
min( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Min{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
argMin( const ET1& a )
{
   return Algorithms::reduceWithArgument( a, TNL::MinWithArg{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
max( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Max{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
argMax( const ET1& a )
{
   return Algorithms::reduceWithArgument( a, TNL::MaxWithArg{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
sum( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Plus{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
maxNorm( const ET1& a )
{
   return max( abs( a ) );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
l1Norm( const ET1& a )
{
   return sum( abs( a ) );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
l2Norm( const ET1& a )
{
   using TNL::sqrt;
   return sqrt( sum( sqr( a ) ) );
}

template< typename ET1, typename Real, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
lpNorm( const ET1& a, const Real& p )
   // since (1.0 / p) has type double, TNL::pow returns double
   -> double
{
   if( p == 1.0 )
      return l1Norm( a );
   if( p == 2.0 )
      return l2Norm( a );
   using TNL::pow;
   return pow( sum( pow( abs( a ), p ) ), 1.0 / p );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
product( const ET1& a )
{
   return Algorithms::reduce( a, TNL::Multiplies{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
logicalAnd( const ET1& a )
{
   return Algorithms::reduce( a, TNL::LogicalAnd{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
logicalOr( const ET1& a )
{
   return Algorithms::reduce( a, TNL::LogicalOr{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
binaryAnd( const ET1& a )
{
   return Algorithms::reduce( a, TNL::BitAnd{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
binaryOr( const ET1& a )
{
   return Algorithms::reduce( a, TNL::BitOr{} );
}

template< typename ET1, typename..., EnableIfUnaryExpression_t< ET1, bool > = true >
auto
binaryXor( const ET1& a )
{
   return Algorithms::reduce( a, TNL::BitXor{} );
}

#endif  // DOXYGEN_ONLY

////
// Output stream
template< typename T1, typename T2, typename Operation >
std::ostream&
operator<<( std::ostream& str, const BinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

template< typename T, typename Operation >
std::ostream&
operator<<( std::ostream& str, const UnaryExpressionTemplate< T, Operation >& expression )
{
   str << "[ ";
   for( int i = 0; i < expression.getSize() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( expression.getSize() - 1 ) << " ]";
   return str;
}

}  // namespace Expressions

// Make all operators visible in the TNL::Containers namespace to be considered
// even for Vector and VectorView
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
using Expressions::sqr;
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
using Containers::sqr;
using Containers::sqrt;
using Containers::sum;
using Containers::tan;
using Containers::tanh;

////
// Evaluation with reduction
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
Result
evaluateAndReduce( Vector& lhs,
                   const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
                   const Reduction& reduction,
                   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> RealType
   {
      return ( lhs_data[ i ] = expression[ i ] );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
Result
evaluateAndReduce( Vector& lhs,
                   const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
                   const Reduction& reduction,
                   const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> RealType
   {
      return ( lhs_data[ i ] = expression[ i ] );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, lhs.getSize(), fetch, reduction, zero );
}

////
// Addition and reduction
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
Result
addAndReduce( Vector& lhs,
              const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
              const Reduction& reduction,
              const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> RealType
   {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
Result
addAndReduce( Vector& lhs,
              const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
              const Reduction& reduction,
              const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> RealType
   {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return aux;
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, lhs.getSize(), fetch, reduction, zero );
}

////
// Addition and reduction
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
Result
addAndReduceAbs( Vector& lhs,
                 const Containers::Expressions::BinaryExpressionTemplate< T1, T2, Operation >& expression,
                 const Reduction& reduction,
                 const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> RealType
   {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
Result
addAndReduceAbs( Vector& lhs,
                 const Containers::Expressions::UnaryExpressionTemplate< T1, Operation >& expression,
                 const Reduction& reduction,
                 const Result& zero )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   using DeviceType = typename Vector::DeviceType;

   RealType* lhs_data = lhs.getData();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> RealType
   {
      const RealType aux = expression[ i ];
      lhs_data[ i ] += aux;
      return TNL::abs( aux );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, lhs.getSize(), fetch, reduction, zero );
}

}  // namespace noa::TNL
