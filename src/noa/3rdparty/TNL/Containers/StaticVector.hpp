// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/TNL/Containers/detail/VectorAssignment.h>

namespace noaTNL {
namespace Containers {

template< int Size, typename Real >
   template< typename T1,
             typename T2,
             typename Operation >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expr )
{
   detail::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation > >::assignStatic( *this, expr );
}

template< int Size,
          typename Real >
   template< typename T,
             typename Operation >
__cuda_callable__
StaticVector< Size, Real >::StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& expr )
{
   detail::VectorAssignment< StaticVector< Size, Real >, Expressions::StaticUnaryExpressionTemplate< T, Operation > >::assignStatic( *this, expr );
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >&
StaticVector< Size, Real >::operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< StaticVector< Size, Real >, VectorExpression >::assignStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::additionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::subtractionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::multiplicationStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::divisionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename VectorExpression >
__cuda_callable__
StaticVector< Size, Real >& StaticVector< Size, Real >::operator%=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::moduloStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
   template< typename OtherReal >
__cuda_callable__
StaticVector< Size, Real >::
operator StaticVector< Size, OtherReal >() const
{
   StaticVector< Size, OtherReal > aux;
   aux.operator=( *this );
   return aux;
}

} // namespace Containers
} // namespace noaTNL
