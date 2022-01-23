// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedVector.h"

namespace noa::TNL {
namespace Containers {

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
DistributedVector< Real, Device, Index, Allocator >::
DistributedVector( const DistributedVector& vector, const AllocatorType& allocator )
: BaseType::DistributedArray( vector, allocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::LocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getLocalView()
{
   return BaseType::getLocalView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::ConstLocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getConstLocalView() const
{
   return BaseType::getConstLocalView();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::LocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getLocalViewWithGhosts()
{
   return BaseType::getLocalViewWithGhosts();
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Real, Device, Index, Allocator >::ConstLocalViewType
DistributedVector< Real, Device, Index, Allocator >::
getConstLocalViewWithGhosts() const
{
   return BaseType::getConstLocalViewWithGhosts();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Value, Device, Index, Allocator >::ViewType
DistributedVector< Value, Device, Index, Allocator >::
getView()
{
   return BaseType::getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedVector< Value, Device, Index, Allocator >::ConstViewType
DistributedVector< Value, Device, Index, Allocator >::
getConstView() const
{
   return BaseType::getConstView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedVector< Value, Device, Index, Allocator >::
operator ViewType()
{
   return getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedVector< Value, Device, Index, Allocator >::
operator ConstViewType() const
{
   return getConstView();
}


/*
 * Usual Vector methods follow below.
 */

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator=( const Vector& vector )
{
   this->setLike( vector );
   getView() = vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator+=( const Vector& vector )
{
   getView() += vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator-=( const Vector& vector )
{
   getView() -= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator*=( const Vector& vector )
{
   getView() *= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator/=( const Vector& vector )
{
   getView() /= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Vector, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator%=( const Vector& vector )
{
   getView() %= vector;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator=( Scalar c )
{
   getView() = c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator+=( Scalar c )
{
   getView() += c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator-=( Scalar c )
{
   getView() -= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator*=( Scalar c )
{
   getView() *= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator/=( Scalar c )
{
   getView() /= c;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Scalar, typename..., typename >
DistributedVector< Real, Device, Index, Allocator >&
DistributedVector< Real, Device, Index, Allocator >::
operator%=( Scalar c )
{
   getView() %= c;
   return *this;
}

} // namespace Containers
} // namespace noa::TNL
