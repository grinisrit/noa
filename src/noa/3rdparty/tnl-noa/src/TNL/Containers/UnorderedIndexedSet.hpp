// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/UnorderedIndexedSet.h>

namespace noa::TNL {
namespace Containers {

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
void
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::clear()
{
   map.clear();
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
typename UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::size_type
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::size() const
{
   return map.size();
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
Index
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::insert( const Key& key )
{
   auto iter = map.insert( value_type( key, size() ) ).first;
   return iter->second;
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
Index
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::insert( Key&& key )
{
   auto iter = map.insert( value_type( std::move( key ), size() ) ).first;
   return iter->second;
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
std::pair< Index, bool >
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::try_insert( const Key& key )
{
   auto pair = map.insert( value_type( key, size() ) );
   return std::pair< Index, bool >{ pair.first->second, pair.second };
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
bool
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::find( const Key& key, Index& index ) const
{
   auto iter = map.find( Key( key ) );
   if( iter == map.end() )
      return false;
   index = iter->second;
   return true;
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
void
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::reserve( size_type count )
{
   map.reserve( count );
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
typename UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::size_type
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::count( const Key& key ) const
{
   return map.count( key );
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
typename UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::size_type
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::erase( const Key& key )
{
   return map.erase( key );
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
void
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::print( std::ostream& str ) const
{
   auto iter = map.begin();
   str << iter->second.data;
   iter++;
   while( iter != map.end() ) {
      str << ", " << iter->second.data;
      iter++;
   }
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
typename UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::iterator
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::begin()
{
   return map.begin();
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
typename UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::const_iterator
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::begin() const
{
   return map.begin();
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
typename UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::iterator
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::end()
{
   return map.end();
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
typename UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::const_iterator
UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >::end() const
{
   return map.end();
}

template< class Key, class Index, class Hash, class KeyEqual, class Allocator >
std::ostream&
operator<<( std::ostream& str, UnorderedIndexedSet< Key, Index, Hash, KeyEqual, Allocator >& set )
{
   set.print( str );
   return str;
}

}  // namespace Containers
}  // namespace noa::TNL
