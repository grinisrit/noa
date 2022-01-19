// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Expressions/ExpressionVariableType.h>
#include <TNL/MPI/Wrappers.h>

namespace TNL {
namespace Containers {
namespace Expressions {

////
// Non-static comparison
template< typename T1,
          typename T2,
          ExpressionVariableType T1Type = getExpressionVariableType< T1, T2 >(),
          ExpressionVariableType T2Type = getExpressionVariableType< T2, T1 >() >
struct DistributedComparison;

/////
// Distributed comparison of two vector expressions
template< typename T1,
          typename T2 >
struct DistributedComparison< T1, T2, VectorExpressionVariable, VectorExpressionVariable >
{
   static bool EQ( const T1& a, const T2& b )
   {
      // we can't run allreduce if the communicators are different
      if( a.getCommunicator() != b.getCommunicator() )
         return false;
      const bool localResult =
            a.getLocalRange() == b.getLocalRange() &&
            a.getGhosts() == b.getGhosts() &&
            a.getSize() == b.getSize() &&
            // compare without ghosts
            a.getConstLocalView() == b.getConstLocalView();
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool NE( const T1& a, const T2& b )
   {
      return ! DistributedComparison::EQ( a, b );
   }

   static bool LT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getLocalRange(), b.getLocalRange(), "Local ranges of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getGhosts(), b.getGhosts(), "Ghosts of expressions to be compared do not match." );

      // we can't run allreduce if the communicators are different
      if( a.getCommunicator() != b.getCommunicator() )
         return false;
      const bool localResult = a.getConstLocalView() < b.getConstLocalView();
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool LE( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getLocalRange(), b.getLocalRange(), "Local ranges of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getGhosts(), b.getGhosts(), "Ghosts of expressions to be compared do not match." );

      // we can't run allreduce if the communicators are different
      if( a.getCommunicator() != b.getCommunicator() )
         return false;
      const bool localResult = a.getConstLocalView() <= b.getConstLocalView();
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool GT( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getLocalRange(), b.getLocalRange(), "Local ranges of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getGhosts(), b.getGhosts(), "Ghosts of expressions to be compared do not match." );

      // we can't run allreduce if the communicators are different
      if( a.getCommunicator() != b.getCommunicator() )
         return false;
      const bool localResult = a.getConstLocalView() > b.getConstLocalView();
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool GE( const T1& a, const T2& b )
   {
      TNL_ASSERT_EQ( a.getSize(), b.getSize(), "Sizes of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getLocalRange(), b.getLocalRange(), "Local ranges of expressions to be compared do not match." );
      TNL_ASSERT_EQ( a.getGhosts(), b.getGhosts(), "Ghosts of expressions to be compared do not match." );

      // we can't run allreduce if the communicators are different
      if( a.getCommunicator() != b.getCommunicator() )
         return false;
      const bool localResult = a.getConstLocalView() >= b.getConstLocalView();
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }
};

/////
// Distributed comparison of number and vector expression
template< typename T1,
          typename T2 >
struct DistributedComparison< T1, T2, ArithmeticVariable, VectorExpressionVariable >
{
   static bool EQ( const T1& a, const T2& b )
   {
      const bool localResult = a == b.getConstLocalView();
      bool result = true;
      if( b.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, b.getCommunicator() );
      return result;
   }

   static bool NE( const T1& a, const T2& b )
   {
      return ! DistributedComparison::EQ( a, b );
   }

   static bool LT( const T1& a, const T2& b )
   {
      const bool localResult = a < b.getConstLocalView();
      bool result = true;
      if( b.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, b.getCommunicator() );
      return result;
   }

   static bool LE( const T1& a, const T2& b )
   {
      const bool localResult = a <= b.getConstLocalView();
      bool result = true;
      if( b.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, b.getCommunicator() );
      return result;
   }

   static bool GT( const T1& a, const T2& b )
   {
      const bool localResult = a > b.getConstLocalView();
      bool result = true;
      if( b.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, b.getCommunicator() );
      return result;
   }

   static bool GE( const T1& a, const T2& b )
   {
      const bool localResult = a >= b.getConstLocalView();
      bool result = true;
      if( b.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, b.getCommunicator() );
      return result;
   }
};

/////
// Distributed comparison of vector expressions and number
template< typename T1,
          typename T2 >
struct DistributedComparison< T1, T2, VectorExpressionVariable, ArithmeticVariable >
{
   static bool EQ( const T1& a, const T2& b )
   {
      const bool localResult = a.getConstLocalView() == b;
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool NE( const T1& a, const T2& b )
   {
      return ! DistributedComparison::EQ( a, b );
   }

   static bool LT( const T1& a, const T2& b )
   {
      const bool localResult = a.getConstLocalView() < b;
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool LE( const T1& a, const T2& b )
   {
      const bool localResult = a.getConstLocalView() <= b;
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool GT( const T1& a, const T2& b )
   {
      const bool localResult = a.getConstLocalView() > b;
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }

   static bool GE( const T1& a, const T2& b )
   {
      const bool localResult = a.getConstLocalView() >= b;
      bool result = true;
      if( a.getCommunicator() != MPI_COMM_NULL )
         MPI::Allreduce( &localResult, &result, 1, MPI_LAND, a.getCommunicator() );
      return result;
   }
};

} // namespace Expressions
} // namespace Containers
} // namespace TNL
