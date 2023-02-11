// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/VectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/detail/VectorAssignment.h>

namespace noa::TNL {
namespace Containers {

template< typename Real, typename Device, typename Index >
__cuda_callable__
typename VectorView< Real, Device, Index >::ViewType
VectorView< Real, Device, Index >::getView( IndexType begin, IndexType end )
{
   TNL_ASSERT_GE( begin, (Index) 0, "Parameter 'begin' must be non-negative." );
   TNL_ASSERT_LE( begin, this->getSize(), "Parameter 'begin' must be lower or equal to size of the vector view." );
   TNL_ASSERT_GE( end, (Index) 0, "Parameter 'end' must be non-negative." );
   TNL_ASSERT_LE( end, this->getSize(), "Parameter 'end' must be lower or equal to size of the vector view." );
   TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );

   if( end == 0 )
      end = this->getSize();
   return ViewType( this->getData() + begin, end - begin );
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
typename VectorView< Real, Device, Index >::ConstViewType
VectorView< Real, Device, Index >::getConstView( const IndexType begin, IndexType end ) const
{
   TNL_ASSERT_GE( begin, (Index) 0, "Parameter 'begin' must be non-negative." );
   TNL_ASSERT_LE( begin, this->getSize(), "Parameter 'begin' must be lower or equal to size of the vector view." );
   TNL_ASSERT_GE( end, (Index) 0, "Parameter 'end' must be non-negative." );
   TNL_ASSERT_LE( end, this->getSize(), "Parameter 'end' must be lower or equal to size of the vector view." );
   TNL_ASSERT_LE( begin, end, "Parameter 'begin' must be lower or equal to the parameter 'end'." );

   if( end == 0 )
      end = this->getSize();
   return ConstViewType( this->getData() + begin, end - begin );
}

template< typename Real, typename Device, typename Index >
template< typename VectorExpression, typename..., typename >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator=( const VectorExpression& expression )
{
   detail::VectorAssignment< VectorView, VectorExpression >::assign( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator+=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::addition( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator-=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::subtraction( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator*=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::multiplication( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator/=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::division( *this, expression );
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename VectorExpression >
VectorView< Real, Device, Index >&
VectorView< Real, Device, Index >::operator%=( const VectorExpression& expression )
{
   detail::VectorAssignmentWithOperation< VectorView, VectorExpression >::modulo( *this, expression );
   return *this;
}

}  // namespace Containers
}  // namespace noa::TNL
