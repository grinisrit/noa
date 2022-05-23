// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/TypeTraits.h>

////
// By vertical operations we mean those applied across vector elements or
// vector expression elements. It means for example minim/maximum of all
// vector elements etc.
namespace noa::TNL {
namespace Containers {
namespace Expressions {

template< typename Expression >
__cuda_callable__
auto
StaticExpressionMin( const Expression& expression )
{
   // use argument-dependent lookup and make TNL::min available for unqualified calls
   using TNL::min;
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = min( aux, expression[ i ] );
   return aux;
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionArgMin( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   int arg = 0;
   ResultType value = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ ) {
      if( expression[ i ] < value ) {
         value = expression[ i ];
         arg = i;
      }
   }
   return std::make_pair( value, arg );
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionMax( const Expression& expression )
{
   // use argument-dependent lookup and make TNL::max available for unqualified calls
   using TNL::max;
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = max( aux, expression[ i ] );
   return aux;
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionArgMax( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   int arg = 0;
   ResultType value = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ ) {
      if( expression[ i ] > value ) {
         value = expression[ i ];
         arg = i;
      }
   }
   return std::make_pair( value, arg );
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionSum( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux += expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionProduct( const Expression& expression )
{
   using ResultType = RemoveET< typename Expression::RealType >;
   ResultType aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux *= expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
bool
StaticExpressionLogicalAnd( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux && expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
bool
StaticExpressionLogicalOr( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux || expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionBinaryAnd( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux & expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionBinaryOr( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux | expression[ i ];
   return aux;
}

template< typename Expression >
__cuda_callable__
auto
StaticExpressionBinaryXor( const Expression& expression )
{
   auto aux = expression[ 0 ];
   for( int i = 1; i < expression.getSize(); i++ )
      aux = aux ^ expression[ i ];
   return aux;
}

}  // namespace Expressions
}  // namespace Containers
}  // namespace noa::TNL
