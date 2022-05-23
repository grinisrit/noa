// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <memory>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/ExpressionTemplates.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/DistributedComparison.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/DistributedVerticalOperations.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

////
// Distributed unary expression template
template< typename T1, typename Operation >
struct DistributedUnaryExpressionTemplate;

template< typename T1, typename Operation >
struct HasEnabledDistributedExpressionTemplates< DistributedUnaryExpressionTemplate< T1, Operation > > : std::true_type
{};

////
// Distributed binary expression template
template< typename T1,
          typename T2,
          typename Operation,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct DistributedBinaryExpressionTemplate
{};

template< typename T1, typename T2, typename Operation, ExpressionVariableType T1Type, ExpressionVariableType T2Type >
struct HasEnabledDistributedExpressionTemplates< DistributedBinaryExpressionTemplate< T1, T2, Operation, T1Type, T2Type > >
: std::true_type
{};

template< typename T1, typename T2, typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType =
      BinaryExpressionTemplate< typename T1::ConstLocalViewType, typename T2::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the left operand." );
   static_assert( HasEnabledDistributedExpressionTemplates< T2 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the right operand." );
   static_assert( std::is_same< typename T1::DeviceType, typename T2::DeviceType >::value,
                  "Attempt to mix operands which have different DeviceType." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a ), op2( b )
   {
      TNL_ASSERT_EQ( op1.getSize(), op2.getSize(), "Attempt to mix operands with different sizes." );
      TNL_ASSERT_EQ( op1.getLocalRange(),
                     op2.getLocalRange(),
                     "Distributed expressions are supported only on vectors which are distributed the same way." );
      TNL_ASSERT_EQ( op1.getGhosts(),
                     op2.getGhosts(),
                     "Distributed expressions are supported only on vectors which are distributed the same way." );
      TNL_ASSERT_EQ( op1.getCommunicator(),
                     op2.getCommunicator(),
                     "Distributed expressions are supported only on vectors within the same communicator." );
   }

   RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType
   getSize() const
   {
      return op1.getSize();
   }

   LocalRangeType
   getLocalRange() const
   {
      return op1.getLocalRange();
   }

   IndexType
   getGhosts() const
   {
      return op1.getGhosts();
   }

   const MPI::Comm&
   getCommunicator() const
   {
      return op1.getCommunicator();
   }

   ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2.getConstLocalView() );
   }

   ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1.getConstLocalViewWithGhosts(), op2.getConstLocalViewWithGhosts() );
   }

   std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return op1.getSynchronizer();
   }

   int
   getValuesPerElement() const
   {
      return op1.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      op1.waitForSynchronization();
      op2.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2& op2;
};

template< typename T1, typename T2, typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, VectorExpressionVariable, ArithmeticVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ], std::declval< T2 >() ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< typename T1::ConstLocalViewType, T2, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the left operand." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a ), op2( b ) {}

   RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType
   getSize() const
   {
      return op1.getSize();
   }

   LocalRangeType
   getLocalRange() const
   {
      return op1.getLocalRange();
   }

   IndexType
   getGhosts() const
   {
      return op1.getGhosts();
   }

   const MPI::Comm&
   getCommunicator() const
   {
      return op1.getCommunicator();
   }

   ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( op1.getConstLocalView(), op2 );
   }

   ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1.getConstLocalViewWithGhosts(), op2 );
   }

   std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return op1.getSynchronizer();
   }

   int
   getValuesPerElement() const
   {
      return op1.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      op1.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2& op2;
};

template< typename T1, typename T2, typename Operation >
struct DistributedBinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorExpressionVariable >
{
   using RealType = decltype( Operation{}( std::declval< T1 >(), std::declval< T2 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T2::DeviceType;
   using IndexType = typename T2::IndexType;
   using LocalRangeType = typename T2::LocalRangeType;
   using ConstLocalViewType = BinaryExpressionTemplate< T1, typename T2::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T2::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T2 >::value,
                  "Invalid operand in distributed binary expression templates - distributed expression templates are not "
                  "enabled for the right operand." );

   DistributedBinaryExpressionTemplate( const T1& a, const T2& b ) : op1( a ), op2( b ) {}

   RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType
   getSize() const
   {
      return op2.getSize();
   }

   LocalRangeType
   getLocalRange() const
   {
      return op2.getLocalRange();
   }

   IndexType
   getGhosts() const
   {
      return op2.getGhosts();
   }

   const MPI::Comm&
   getCommunicator() const
   {
      return op2.getCommunicator();
   }

   ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( op1, op2.getConstLocalView() );
   }

   ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( op1, op2.getConstLocalViewWithGhosts() );
   }

   std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return op2.getSynchronizer();
   }

   int
   getValuesPerElement() const
   {
      return op2.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      op2.waitForSynchronization();
   }

protected:
   const T1& op1;
   const T2& op2;
};

////
// Distributed unary expression template
template< typename T1, typename Operation >
struct DistributedUnaryExpressionTemplate
{
   using RealType = decltype( Operation{}( std::declval< T1 >()[ 0 ] ) );
   using ValueType = RealType;
   using DeviceType = typename T1::DeviceType;
   using IndexType = typename T1::IndexType;
   using LocalRangeType = typename T1::LocalRangeType;
   using ConstLocalViewType = UnaryExpressionTemplate< typename T1::ConstLocalViewType, Operation >;
   using SynchronizerType = typename T1::SynchronizerType;

   static_assert( HasEnabledDistributedExpressionTemplates< T1 >::value,
                  "Invalid operand in distributed unary expression templates - distributed expression templates are not "
                  "enabled for the operand." );

   DistributedUnaryExpressionTemplate( const T1& a ) : operand( a ) {}

   RealType
   getElement( const IndexType i ) const
   {
      const IndexType li = getLocalRange().getLocalIndex( i );
      return getConstLocalView().getElement( li );
   }

   // this is actually never executed, but needed for proper ExpressionVariableTypeGetter
   // selection via HasSubscriptOperator type trait
   RealType
   operator[]( const IndexType i ) const
   {
      return getConstLocalView()[ i ];
   }

   IndexType
   getSize() const
   {
      return operand.getSize();
   }

   LocalRangeType
   getLocalRange() const
   {
      return operand.getLocalRange();
   }

   IndexType
   getGhosts() const
   {
      return operand.getGhosts();
   }

   const MPI::Comm&
   getCommunicator() const
   {
      return operand.getCommunicator();
   }

   ConstLocalViewType
   getConstLocalView() const
   {
      return ConstLocalViewType( operand.getConstLocalView() );
   }

   ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return ConstLocalViewType( operand.getConstLocalViewWithGhosts() );
   }

   std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return operand.getSynchronizer();
   }

   int
   getValuesPerElement() const
   {
      return operand.getValuesPerElement();
   }

   void
   waitForSynchronization() const
   {
      operand.waitForSynchronization();
   }

protected:
   const T1& operand;
};

#ifndef DOXYGEN_ONLY

   #define TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( fname, functor )                                    \
      template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true > \
      auto fname( const ET1& a )                                                                      \
      {                                                                                               \
         return DistributedUnaryExpressionTemplate< ET1, functor >( a );                              \
      }

   #define TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( fname, functor )                                                       \
      template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true > \
      auto fname( const ET1& a, const ET2& b )                                                                            \
      {                                                                                                                   \
         return DistributedBinaryExpressionTemplate< ET1, ET2, functor >( a, b );                                         \
      }

TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator+, TNL::Plus )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator-, TNL::Minus )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator*, TNL::Multiplies )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator/, TNL::Divides )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( operator%, TNL::Modulus )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( min, TNL::Min )
TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION( max, TNL::Max )

TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( operator+, TNL::UnaryPlus )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( operator-, TNL::UnaryMinus )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( abs, TNL::Abs )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( exp, TNL::Exp )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sqrt, TNL::Sqrt )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( cbrt, TNL::Cbrt )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( log, TNL::Log )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( log10, TNL::Log10 )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( log2, TNL::Log2 )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sin, TNL::Sin )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( cos, TNL::Cos )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( tan, TNL::Tan )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( asin, TNL::Asin )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( acos, TNL::Acos )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( atan, TNL::Atan )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sinh, TNL::Sinh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( cosh, TNL::Cosh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( tanh, TNL::Tanh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( asinh, TNL::Asinh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( acosh, TNL::Acosh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( atanh, TNL::Atanh )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( floor, TNL::Floor )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( ceil, TNL::Ceil )
TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION( sign, TNL::Sign )

   #undef TNL_MAKE_DISTRIBUTED_UNARY_EXPRESSION
   #undef TNL_MAKE_DISTRIBUTED_BINARY_EXPRESSION

////
// Pow
template< typename ET1, typename Real, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
pow( const ET1& a, const Real& exp )
{
   return DistributedBinaryExpressionTemplate< ET1, Real, Pow >( a, exp );
}

////
// Cast
template< typename ResultType, typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
cast( const ET1& a )
{
   using CastOperation = typename Cast< ResultType >::Operation;
   return DistributedUnaryExpressionTemplate< ET1, CastOperation >( a );
}

////
// Comparison operator ==
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator==( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::EQ( a, b );
}

////
// Comparison operator !=
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator!=( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::NE( a, b );
}

////
// Comparison operator <
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator<( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::LT( a, b );
}

////
// Comparison operator <=
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator<=( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::LE( a, b );
}

////
// Comparison operator >
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator>( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::GT( a, b );
}

////
// Comparison operator >=
template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
bool
operator>=( const ET1& a, const ET2& b )
{
   return DistributedComparison< ET1, ET2 >::GE( a, b );
}

////
// Scalar product
template< typename ET1, typename ET2,
          typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
auto
operator,( const ET1& a, const ET2& b )
{
   return DistributedExpressionSum( a * b );
}

template< typename ET1, typename ET2, typename..., EnableIfDistributedBinaryExpression_t< ET1, ET2, bool > = true >
auto
dot( const ET1& a, const ET2& b )
{
   return ( a, b );
}

////
// Vertical operations
template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
min( const ET1& a )
{
   return DistributedExpressionMin( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
argMin( const ET1& a )
{
   return DistributedExpressionArgMin( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
max( const ET1& a )
{
   return DistributedExpressionMax( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
argMax( const ET1& a )
{
   return DistributedExpressionArgMax( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
sum( const ET1& a )
{
   return DistributedExpressionSum( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
maxNorm( const ET1& a )
{
   return max( abs( a ) );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
l1Norm( const ET1& a )
{
   return sum( abs( a ) );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
l2Norm( const ET1& a )
{
   using TNL::sqrt;
   return sqrt( sum( a * a ) );
}

template< typename ET1, typename Real, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
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

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
product( const ET1& a )
{
   return DistributedExpressionProduct( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
logicalAnd( const ET1& a )
{
   return DistributedExpressionLogicalAnd( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
logicalOr( const ET1& a )
{
   return DistributedExpressionLogicalOr( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
binaryAnd( const ET1& a )
{
   return DistributedExpressionBinaryAnd( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
binaryOr( const ET1& a )
{
   return DistributedExpressionBinaryOr( a );
}

template< typename ET1, typename..., EnableIfDistributedUnaryExpression_t< ET1, bool > = true >
auto
binaryXor( const ET1& a )
{
   return DistributedExpressionBinaryXor( a );
}

////
// Output stream
template< typename T1, typename T2, typename Operation >
std::ostream&
operator<<( std::ostream& str, const DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression )
{
   const auto localRange = expression.getLocalRange();
   str << "[ ";
   for( int i = localRange.getBegin(); i < localRange.getEnd() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( localRange.getEnd() - 1 );
   if( expression.getGhosts() > 0 ) {
      str << " | ";
      const auto localView = expression.getConstLocalViewWithGhosts();
      for( int i = localRange.getSize(); i < localView.getSize() - 1; i++ )
         str << localView.getElement( i ) << ", ";
      str << localView.getElement( localView.getSize() - 1 );
   }
   str << " ]";
   return str;
}

template< typename T, typename Operation >
std::ostream&
operator<<( std::ostream& str, const DistributedUnaryExpressionTemplate< T, Operation >& expression )
{
   const auto localRange = expression.getLocalRange();
   str << "[ ";
   for( int i = localRange.getBegin(); i < localRange.getEnd() - 1; i++ )
      str << expression.getElement( i ) << ", ";
   str << expression.getElement( localRange.getEnd() - 1 );
   if( expression.getGhosts() > 0 ) {
      str << " | ";
      const auto localView = expression.getConstLocalViewWithGhosts();
      for( int i = localRange.getSize(); i < localView.getSize() - 1; i++ )
         str << localView.getElement( i ) << ", ";
      str << localView.getElement( localView.getSize() - 1 );
   }
   str << " ]";
   return str;
}

#endif  // DOXYGEN_ONLY

}  // namespace Expressions

// Make all operators visible in the TNL::Containers namespace to be considered
// even for DistributedVector and DistributedVectorView
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
Result
evaluateAndReduce( Vector& lhs,
                   const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
Result
evaluateAndReduce( Vector& lhs,
                   const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( lhs.getSize(), fetch, reduction, zero );
}

////
// Addition and reduction
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
Result
addAndReduce( Vector& lhs,
              const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
Result
addAndReduce( Vector& lhs,
              const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( lhs.getSize(), fetch, reduction, zero );
}

////
// Addition and reduction
template< typename Vector, typename T1, typename T2, typename Operation, typename Reduction, typename Result >
Result
addAndReduceAbs( Vector& lhs,
                 const Containers::Expressions::DistributedBinaryExpressionTemplate< T1, T2, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( lhs.getSize(), fetch, reduction, zero );
}

template< typename Vector, typename T1, typename Operation, typename Reduction, typename Result >
Result
addAndReduceAbs( Vector& lhs,
                 const Containers::Expressions::DistributedUnaryExpressionTemplate< T1, Operation >& expression,
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
   return Algorithms::reduce< DeviceType >( lhs.getSize(), fetch, reduction, zero );
}

}  // namespace noa::TNL
