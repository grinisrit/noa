// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedArrayView.h"

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Wrappers.h>

namespace noa::TNL {
namespace Containers {

template< typename Value, typename Device, typename Index >
DistributedArrayView< Value, Device, Index >::~DistributedArrayView()
{
   // Wait for pending async operation, otherwise the synchronizer might crash
   // if the view goes out of scope.
   // (The same thing is done even in DistributedArray, but there might be views
   // bound to an array without a synchronizer, in which case this helps.)
   waitForSynchronization();
}

template< typename Value, typename Device, typename Index >
template< typename Value_ >
DistributedArrayView< Value, Device, Index >::DistributedArrayView( const DistributedArrayView< Value_, Device, Index >& view )
: localRange( view.getLocalRange() ), ghosts( view.getGhosts() ), globalSize( view.getSize() ),
  communicator( view.getCommunicator() ), localData( view.getConstLocalViewWithGhosts() ),
  synchronizer( view.getSynchronizer() ), valuesPerElement( view.getValuesPerElement() )
{}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::bind( const LocalRangeType& localRange,
                                                    IndexType ghosts,
                                                    IndexType globalSize,
                                                    const MPI::Comm& communicator,
                                                    LocalViewType localData )
{
   TNL_ASSERT_EQ( localData.getSize(),
                  localRange.getSize() + ghosts,
                  "The local array size does not match the local range of the distributed array." );
   TNL_ASSERT_GE( ghosts, 0, "The ghosts count must be non-negative." );

   this->localRange = localRange;
   this->ghosts = ghosts;
   this->globalSize = globalSize;
   this->communicator = communicator;
   this->localData.bind( localData );
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::bind( DistributedArrayView view )
{
   localRange = view.getLocalRange();
   ghosts = view.getGhosts();
   globalSize = view.getSize();
   communicator = view.getCommunicator();
   localData.bind( view.getLocalViewWithGhosts() );
   // set, but do not unset, the synchronizer
   if( view.getSynchronizer() )
      setSynchronizer( view.getSynchronizer(), view.getValuesPerElement() );
}

template< typename Value, typename Device, typename Index >
template< typename Value_ >
void
DistributedArrayView< Value, Device, Index >::bind( Value_* data, IndexType localSize )
{
   TNL_ASSERT_EQ( localSize,
                  localRange.getSize() + ghosts,
                  "The local array size does not match the local range of the distributed array." );
   localData.bind( data, localSize );
}

template< typename Value, typename Device, typename Index >
const Subrange< Index >&
DistributedArrayView< Value, Device, Index >::getLocalRange() const
{
   return localRange;
}

template< typename Value, typename Device, typename Index >
Index
DistributedArrayView< Value, Device, Index >::getGhosts() const
{
   return ghosts;
}

template< typename Value, typename Device, typename Index >
const MPI::Comm&
DistributedArrayView< Value, Device, Index >::getCommunicator() const
{
   return communicator;
}

template< typename Value, typename Device, typename Index >
typename DistributedArrayView< Value, Device, Index >::LocalViewType
DistributedArrayView< Value, Device, Index >::getLocalView()
{
   return LocalViewType( localData.getData(), localRange.getSize() );
}

template< typename Value, typename Device, typename Index >
typename DistributedArrayView< Value, Device, Index >::ConstLocalViewType
DistributedArrayView< Value, Device, Index >::getConstLocalView() const
{
   return ConstLocalViewType( localData.getData(), localRange.getSize() );
}

template< typename Value, typename Device, typename Index >
typename DistributedArrayView< Value, Device, Index >::LocalViewType
DistributedArrayView< Value, Device, Index >::getLocalViewWithGhosts()
{
   return localData;
}

template< typename Value, typename Device, typename Index >
typename DistributedArrayView< Value, Device, Index >::ConstLocalViewType
DistributedArrayView< Value, Device, Index >::getConstLocalViewWithGhosts() const
{
   return localData;
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::copyFromGlobal( ConstLocalViewType globalArray )
{
   TNL_ASSERT_EQ( getSize(), globalArray.getSize(), "given global array has different size than the distributed array view" );

   LocalViewType localView = getLocalView();
   const LocalRangeType localRange = getLocalRange();

   auto kernel = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      localView[ i ] = globalArray[ localRange.getGlobalIndex( i ) ];
   };

   Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, localRange.getSize(), kernel );
   startSynchronization();
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::setSynchronizer( std::shared_ptr< SynchronizerType > synchronizer,
                                                               int valuesPerElement )
{
   this->synchronizer = std::move( synchronizer );
   this->valuesPerElement = valuesPerElement;
}

template< typename Value, typename Device, typename Index >
std::shared_ptr< typename DistributedArrayView< Value, Device, Index >::SynchronizerType >
DistributedArrayView< Value, Device, Index >::getSynchronizer() const
{
   return synchronizer;
}

template< typename Value, typename Device, typename Index >
int
DistributedArrayView< Value, Device, Index >::getValuesPerElement() const
{
   return valuesPerElement;
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::startSynchronization()
{
   if( ghosts == 0 )
      return;
   // TODO: assert does not play very nice with automatic synchronizations from operations like
   //       assignment of scalars
   // (Maybe we should just drop all automatic syncs? But that's not nice for high-level codes
   // like linear solvers...)
   TNL_ASSERT_TRUE( synchronizer, "the synchronizer was not set" );

   typename SynchronizerType::ByteArrayView bytes;
   bytes.bind( reinterpret_cast< std::uint8_t* >( localData.getData() ), sizeof( ValueType ) * localData.getSize() );
   synchronizer->synchronizeByteArrayAsync( bytes, sizeof( ValueType ) * valuesPerElement );
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::waitForSynchronization() const
{
   if( synchronizer && synchronizer->async_op.valid() ) {
      synchronizer->async_wait_timer.start();
      synchronizer->async_op.wait();
      synchronizer->async_wait_timer.stop();
   }
}

template< typename Value, typename Device, typename Index >
typename DistributedArrayView< Value, Device, Index >::ViewType
DistributedArrayView< Value, Device, Index >::getView()
{
   return *this;
}

template< typename Value, typename Device, typename Index >
typename DistributedArrayView< Value, Device, Index >::ConstViewType
DistributedArrayView< Value, Device, Index >::getConstView() const
{
   return *this;
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::reset()
{
   localRange.reset();
   ghosts = 0;
   globalSize = 0;
   communicator = MPI_COMM_NULL;
   localData.reset();
}

template< typename Value, typename Device, typename Index >
bool
DistributedArrayView< Value, Device, Index >::empty() const
{
   return getSize() == 0;
}

// TODO: swap

template< typename Value, typename Device, typename Index >
Index
DistributedArrayView< Value, Device, Index >::getSize() const
{
   return globalSize;
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::setValue( ValueType value )
{
   localData.setValue( value );
   startSynchronization();
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::setElement( IndexType i, ValueType value )
{
   const IndexType li = localRange.getLocalIndex( i );
   localData.setElement( li, value );
}

template< typename Value, typename Device, typename Index >
Value
DistributedArrayView< Value, Device, Index >::getElement( IndexType i ) const
{
   const IndexType li = localRange.getLocalIndex( i );
   return localData.getElement( li );
}

template< typename Value, typename Device, typename Index >
__cuda_callable__
Value&
DistributedArrayView< Value, Device, Index >::operator[]( IndexType i )
{
   const IndexType li = localRange.getLocalIndex( i );
   return localData[ li ];
}

template< typename Value, typename Device, typename Index >
__cuda_callable__
const Value&
DistributedArrayView< Value, Device, Index >::operator[]( IndexType i ) const
{
   const IndexType li = localRange.getLocalIndex( i );
   return localData[ li ];
}

template< typename Value, typename Device, typename Index >
DistributedArrayView< Value, Device, Index >&
DistributedArrayView< Value, Device, Index >::operator=( const DistributedArrayView& view )
{
   TNL_ASSERT_EQ( getSize(), view.getSize(), "The sizes of the array views must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getLocalRange(), view.getLocalRange(), "The local ranges must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getGhosts(), view.getGhosts(), "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getCommunicator(), view.getCommunicator(), "The communicators of the array views must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      view.waitForSynchronization();
      getLocalViewWithGhosts() = view.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Value, typename Device, typename Index >
template< typename Array, typename..., typename >
DistributedArrayView< Value, Device, Index >&
DistributedArrayView< Value, Device, Index >::operator=( const Array& array )
{
   TNL_ASSERT_EQ( getSize(), array.getSize(), "The global sizes must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getLocalRange(), array.getLocalRange(), "The local ranges must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getGhosts(), array.getGhosts(), "Ghosts must be equal, views are not resizable." );
   TNL_ASSERT_EQ( getCommunicator(), array.getCommunicator(), "The communicators must be equal." );

   if( this->getCommunicator() != MPI_COMM_NULL ) {
      // TODO: it might be better to split the local and ghost parts and synchronize in the middle
      this->waitForSynchronization();
      array.waitForSynchronization();
      getLocalViewWithGhosts() = array.getConstLocalViewWithGhosts();
   }
   return *this;
}

template< typename Value, typename Device, typename Index >
template< typename Array >
bool
DistributedArrayView< Value, Device, Index >::operator==( const Array& array ) const
{
   // we can't run allreduce if the communicators are different
   if( communicator != array.getCommunicator() )
      return false;
   const bool localResult = localRange == array.getLocalRange() && ghosts == array.getGhosts() && globalSize == array.getSize()
                         &&
                            // compare without ghosts
                            getConstLocalView() == array.getConstLocalView();
   bool result = true;
   if( communicator != MPI_COMM_NULL )
      MPI::Allreduce( &localResult, &result, 1, MPI_LAND, communicator );
   return result;
}

template< typename Value, typename Device, typename Index >
template< typename Array >
bool
DistributedArrayView< Value, Device, Index >::operator!=( const Array& array ) const
{
   return ! ( *this == array );
}

template< typename Value, typename Device, typename Index >
template< typename Function >
void
DistributedArrayView< Value, Device, Index >::forElements( IndexType begin, IndexType end, Function&& f )
{
   // GOTCHA: we can't use localRange.getLocalIndex to calculate localEnd, because localRange.getEnd() does not return a valid
   // local index
   const IndexType localBegin = max( begin, localRange.getBegin() ) - localRange.getBegin();
   const IndexType localEnd = min( end, localRange.getEnd() ) - localRange.getBegin();
   const LocalRangeType localRange = getLocalRange();
   auto local_f = [ = ] __cuda_callable__( IndexType idx, ValueType & value ) mutable
   {
      f( localRange.getGlobalIndex( idx ), value );
   };
   localData.forElements( localBegin, localEnd, local_f );
}

template< typename Value, typename Device, typename Index >
template< typename Function >
void
DistributedArrayView< Value, Device, Index >::forElements( IndexType begin, IndexType end, Function&& f ) const
{
   // GOTCHA: we can't use localRange.getLocalIndex to calculate localEnd, because localRange.getEnd() does not return a valid
   // local index
   const IndexType localBegin = max( begin, localRange.getBegin() ) - localRange.getBegin();
   const IndexType localEnd = min( end, localRange.getEnd() ) - localRange.getBegin();
   const LocalRangeType localRange = getLocalRange();
   auto local_f = [ = ] __cuda_callable__( IndexType idx, const ValueType& value )
   {
      f( localRange.getGlobalIndex( idx ), value );
   };
   localData.forElements( localBegin, localEnd, local_f );
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::loadFromGlobalFile( const String& fileName, bool allowCasting )
{
   File file( fileName, std::ios_base::in );
   loadFromGlobalFile( file, allowCasting );
}

template< typename Value, typename Device, typename Index >
void
DistributedArrayView< Value, Device, Index >::loadFromGlobalFile( File& file, bool allowCasting )
{
   using IO = detail::ArrayIO< Value, Index, typename Allocators::Default< Device >::template Allocator< Value > >;
   const std::string type = getObjectType( file );
   const auto parsedType = parseObjectType( type );

   if( ! allowCasting && type != IO::getSerializationType() )
      throw Exceptions::FileDeserializationError(
         file.getFileName(), "object type does not match (expected " + IO::getSerializationType() + ", found " + type + ")." );

   std::size_t elementsInFile;
   file.load( &elementsInFile );

   if( allowCasting )
      IO::loadSubrange(
         file, elementsInFile, localRange.getBegin(), localData.getData(), localData.getSize(), parsedType[ 1 ] );
   else
      IO::loadSubrange( file, elementsInFile, localRange.getBegin(), localData.getData(), localData.getSize() );
}

}  // namespace Containers
}  // namespace noa::TNL
