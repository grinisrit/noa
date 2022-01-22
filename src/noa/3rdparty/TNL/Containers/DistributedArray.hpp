// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedArray.h"

namespace noaTNL {
namespace Containers {

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >::
~DistributedArray()
{
   // Wait for pending async operation, otherwise the synchronizer would crash
   // if the array goes out of scope.
   waitForSynchronization();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >::
DistributedArray( const Allocator& allocator )
: localData( allocator )
{
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >::
DistributedArray( const DistributedArray& array )
{
   setLike( array );
   view = array;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >::
DistributedArray( const DistributedArray& array, const Allocator& allocator )
: localData( allocator )
{
   setLike( array );
   view = array;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >::
DistributedArray( LocalRangeType localRange, IndexType ghosts, IndexType globalSize, MPI_Comm communicator, const Allocator& allocator )
: localData( allocator )
{
   setDistribution( localRange, ghosts, globalSize, communicator );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
setDistribution( LocalRangeType localRange, IndexType ghosts, IndexType globalSize, MPI_Comm communicator )
{
   TNL_ASSERT_LE( localRange.getEnd(), globalSize, "end of the local range is outside of the global range" );
   if( communicator != MPI_COMM_NULL )
      localData.setSize( localRange.getSize() + ghosts );
   view.bind( localRange, ghosts, globalSize, communicator, localData.getView() );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
const Subrange< Index >&
DistributedArray< Value, Device, Index, Allocator >::
getLocalRange() const
{
   return view.getLocalRange();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
Index
DistributedArray< Value, Device, Index, Allocator >::
getGhosts() const
{
   return view.getGhosts();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
MPI_Comm
DistributedArray< Value, Device, Index, Allocator >::
getCommunicator() const
{
   return view.getCommunicator();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
Allocator
DistributedArray< Value, Device, Index, Allocator >::
getAllocator() const
{
   return localData.getAllocator();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedArray< Value, Device, Index, Allocator >::LocalViewType
DistributedArray< Value, Device, Index, Allocator >::
getLocalView()
{
   return view.getLocalView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedArray< Value, Device, Index, Allocator >::ConstLocalViewType
DistributedArray< Value, Device, Index, Allocator >::
getConstLocalView() const
{
   return view.getConstLocalView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedArray< Value, Device, Index, Allocator >::LocalViewType
DistributedArray< Value, Device, Index, Allocator >::
getLocalViewWithGhosts()
{
   return view.getLocalViewWithGhosts();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedArray< Value, Device, Index, Allocator >::ConstLocalViewType
DistributedArray< Value, Device, Index, Allocator >::
getConstLocalViewWithGhosts() const
{
   return view.getConstLocalViewWithGhosts();
}


template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
copyFromGlobal( ConstLocalViewType globalArray )
{
   view.copyFromGlobal( globalArray );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
setSynchronizer( std::shared_ptr< SynchronizerType > synchronizer, int valuesPerElement )
{
   view.setSynchronizer( synchronizer, valuesPerElement );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
std::shared_ptr< typename DistributedArrayView< Value, Device, Index >::SynchronizerType >
DistributedArray< Value, Device, Index, Allocator >::
getSynchronizer() const
{
   return view.getSynchronizer();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
int
DistributedArray< Value, Device, Index, Allocator >::
getValuesPerElement() const
{
   return view.getValuesPerElement();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
startSynchronization()
{
   view.startSynchronization();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
waitForSynchronization() const
{
   view.waitForSynchronization();
}


/*
 * Usual Array methods follow below.
 */

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedArray< Value, Device, Index, Allocator >::ViewType
DistributedArray< Value, Device, Index, Allocator >::
getView()
{
   return view;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
typename DistributedArray< Value, Device, Index, Allocator >::ConstViewType
DistributedArray< Value, Device, Index, Allocator >::
getConstView() const
{
   return view.getConstView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >::
operator ViewType()
{
   return getView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >::
operator ConstViewType() const
{
   return getConstView();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Array >
void
DistributedArray< Value, Device, Index, Allocator >::
setLike( const Array& array )
{
   localData.setLike( array.getConstLocalViewWithGhosts() );
   view.bind( array.getLocalRange(), array.getGhosts(), array.getSize(), array.getCommunicator(), localData.getView() );
   // set, but do not unset, the synchronizer
   if( array.getSynchronizer() )
      setSynchronizerHelper( view, array );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
reset()
{
   view.reset();
   localData.reset();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
bool
DistributedArray< Value, Device, Index, Allocator >::
empty() const
{
   return view.empty();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
Index
DistributedArray< Value, Device, Index, Allocator >::
getSize() const
{
   return view.getSize();
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
setValue( ValueType value )
{
   view.setValue( value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
void
DistributedArray< Value, Device, Index, Allocator >::
setElement( IndexType i, ValueType value )
{
   view.setElement( i, value );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
Value
DistributedArray< Value, Device, Index, Allocator >::
getElement( IndexType i ) const
{
   return view.getElement( i );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
__cuda_callable__
Value&
DistributedArray< Value, Device, Index, Allocator >::
operator[]( IndexType i )
{
   return view[ i ];
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
__cuda_callable__
const Value&
DistributedArray< Value, Device, Index, Allocator >::
operator[]( IndexType i ) const
{
   return view[ i ];
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
DistributedArray< Value, Device, Index, Allocator >&
DistributedArray< Value, Device, Index, Allocator >::
operator=( const DistributedArray& array )
{
   setLike( array );
   view = array;
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Array, typename..., typename >
DistributedArray< Value, Device, Index, Allocator >&
DistributedArray< Value, Device, Index, Allocator >::
operator=( const Array& array )
{
   setLike( array );
   view = array;
   return *this;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Array >
bool
DistributedArray< Value, Device, Index, Allocator >::
operator==( const Array& array ) const
{
   return view == array;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Array >
bool
DistributedArray< Value, Device, Index, Allocator >::
operator!=( const Array& array ) const
{
   return view != array;
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Function >
void
DistributedArray< Value, Device, Index, Allocator >::
forElements( IndexType begin, IndexType end, Function&& f )
{
   view.forElements( begin, end, f );
}

template< typename Value,
          typename Device,
          typename Index,
          typename Allocator >
   template< typename Function >
void
DistributedArray< Value, Device, Index, Allocator >::
forElements( IndexType begin, IndexType end, Function&& f ) const
{
   view.forElements( begin, end, f );
}

} // namespace Containers
} // namespace noaTNL
