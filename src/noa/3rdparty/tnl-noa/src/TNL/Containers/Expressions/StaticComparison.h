// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/ExpressionVariableType.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

template< typename T1,
          typename T2,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct StaticComparison;

/////
// Static comparison of vector expressions
template< typename T1, typename T2 >
struct StaticComparison< T1, T2, VectorExpressionVariable, VectorExpressionVariable >
{
   static constexpr bool
   EQ( const T1& a, const T2& b )
   {
      if( a.getSize() != b.getSize() )
         return false;
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] == b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   static constexpr bool
   GT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] > b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   GE( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] >= b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   LT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] < b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   LE( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not fit." );
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] <= b[ i ] ) )
            return false;
      return true;
   }
};

/////
// Static comparison of number and vector expressions
template< typename T1, typename T2 >
struct StaticComparison< T1, T2, ArithmeticVariable, VectorExpressionVariable >
{
   static constexpr bool
   EQ( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( ! ( a == b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   static constexpr bool
   GT( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( ! ( a > b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   GE( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( ! ( a >= b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   LT( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( ! ( a < b[ i ] ) )
            return false;
      return true;
   }

   static constexpr bool
   LE( const T1& a, const T2& b )
   {
      for( int i = 0; i < b.getSize(); i++ )
         if( ! ( a <= b[ i ] ) )
            return false;
      return true;
   }
};

/////
// Static comparison of vector expressions and number
template< typename T1, typename T2 >
struct StaticComparison< T1, T2, VectorExpressionVariable, ArithmeticVariable >
{
   static constexpr bool
   EQ( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] == b ) )
            return false;
      return true;
   }

   static constexpr bool
   NE( const T1& a, const T2& b )
   {
      return ! EQ( a, b );
   }

   static constexpr bool
   GT( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] > b ) )
            return false;
      return true;
   }

   static constexpr bool
   GE( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] >= b ) )
            return false;
      return true;
   }

   static constexpr bool
   LT( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] < b ) )
            return false;
      return true;
   }

   static constexpr bool
   LE( const T1& a, const T2& b )
   {
      for( int i = 0; i < a.getSize(); i++ )
         if( ! ( a[ i ] <= b ) )
            return false;
      return true;
   }
};

}  // namespace Expressions
}  // namespace Containers
}  // namespace noa::TNL
