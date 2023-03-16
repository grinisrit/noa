// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/detail/VectorAssignment.h>

namespace noa::TNL {
namespace Containers {

template< int Size, typename Real >
template< typename T1, typename T2, typename Operation >
constexpr StaticVector< Size, Real >::StaticVector(
   const Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation >& expr )
: StaticArray< Size, Real >()
{
   detail::VectorAssignment< StaticVector< Size, Real >,
                             Expressions::StaticBinaryExpressionTemplate< T1, T2, Operation > >::assignStatic( *this, expr );
}

template< int Size, typename Real >
template< typename T, typename Operation >
constexpr StaticVector< Size, Real >::StaticVector( const Expressions::StaticUnaryExpressionTemplate< T, Operation >& expr )
: StaticArray< Size, Real >()
{
   detail::VectorAssignment< StaticVector< Size, Real >,
                             Expressions::StaticUnaryExpressionTemplate< T, Operation > >::assignStatic( *this, expr );
}

template< int Size, typename Real >
template< typename VectorExpression >
constexpr StaticVector< Size, Real >&
StaticVector< Size, Real >::operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< StaticVector< Size, Real >, VectorExpression >::assignStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
template< typename VectorExpression >
constexpr StaticVector< Size, Real >&
StaticVector< Size, Real >::operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::additionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
template< typename VectorExpression >
constexpr StaticVector< Size, Real >&
StaticVector< Size, Real >::operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::subtractionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
template< typename VectorExpression >
constexpr StaticVector< Size, Real >&
StaticVector< Size, Real >::operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::multiplicationStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
template< typename VectorExpression >
constexpr StaticVector< Size, Real >&
StaticVector< Size, Real >::operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::divisionStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
template< typename VectorExpression >
constexpr StaticVector< Size, Real >&
StaticVector< Size, Real >::operator%=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< StaticVector, VectorExpression >::moduloStatic( *this, expression );
   return *this;
}

template< int Size, typename Real >
template< typename OtherReal >
// NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
__cuda_callable__
constexpr StaticVector< Size, Real >::operator StaticVector< Size, OtherReal >() const
{
   StaticVector< Size, OtherReal > aux;
   aux.operator=( *this );
   return aux;
}

}  // namespace Containers
}  // namespace noa::TNL
