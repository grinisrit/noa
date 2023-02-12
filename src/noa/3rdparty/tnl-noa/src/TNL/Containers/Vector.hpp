// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>

namespace noa::TNL {
namespace Containers {

template< typename Real, typename Device, typename Index, typename Allocator >
Vector< Real, Device, Index, Allocator >::Vector( const Vector& vector, const AllocatorType& allocator )
: Array< Real, Device, Index, Allocator >( vector, allocator )
{}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression, typename..., typename >
Vector< Real, Device, Index, Allocator >::Vector( const VectorExpression& expression )
{
   detail::VectorAssignment< Vector, VectorExpression >::resize( *this, expression );
   detail::VectorAssignment< Vector, VectorExpression >::assign( *this, expression );
}

template< typename Real, typename Device, typename Index, typename Allocator >
typename Vector< Real, Device, Index, Allocator >::ViewType
Vector< Real, Device, Index, Allocator >::getView( IndexType begin, IndexType end )
{
   TNL_ASSERT_GE( begin, (Index) 0, "Parameter 'begin' must be non-negative." );
   TNL_ASSERT_LE( begin, this->getSize(), "Parameter 'begin' must be lower or equal to size of the vector." );
   TNL_ASSERT_GE( end, (Index) 0, "Parameter 'end' must be non-negative." );
   TNL_ASSERT_LE( end, this->getSize(), "Parameter 'end' must be lower or equal to size of the vector." );
   TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );

   if( end == 0 )
      end = this->getSize();
   return ViewType( this->getData() + begin, end - begin );
}

template< typename Real, typename Device, typename Index, typename Allocator >
typename Vector< Real, Device, Index, Allocator >::ConstViewType
Vector< Real, Device, Index, Allocator >::getConstView( IndexType begin, IndexType end ) const
{
   TNL_ASSERT_GE( begin, (Index) 0, "Parameter 'begin' must be non-negative." );
   TNL_ASSERT_LE( begin, this->getSize(), "Parameter 'begin' must be lower or equal to size of the vector." );
   TNL_ASSERT_GE( end, (Index) 0, "Parameter 'end' must be non-negative." );
   TNL_ASSERT_LE( end, this->getSize(), "Parameter 'end' must be lower or equal to size of the vector." );
   TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );

   if( end == 0 )
      end = this->getSize();
   return ConstViewType( this->getData() + begin, end - begin );
}

template< typename Real, typename Device, typename Index, typename Allocator >
Vector< Real, Device, Index, Allocator >::operator ViewType()
{
   return getView();
}

template< typename Real, typename Device, typename Index, typename Allocator >
Vector< Real, Device, Index, Allocator >::operator ConstViewType() const
{
   return getConstView();
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression, typename..., typename >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< Vector, VectorExpression >::resize( *this, expression );
   detail::VectorAssignment< Vector, VectorExpression >::assign( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::addition( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::subtraction( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::multiplication( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index, typename Allocator >
template< typename VectorExpression >
Vector< Real, Device, Index, Allocator >&
Vector< Real, Device, Index, Allocator >::operator%=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< Vector, VectorExpression >::modulo( *this, expression );
   return *this;
}

}  // namespace Containers
}  // namespace noa::TNL
