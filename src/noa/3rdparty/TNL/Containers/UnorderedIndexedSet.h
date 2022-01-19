// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_map>
#include <ostream>

namespace TNL {
namespace Containers {

template< class Key,
          class Index,
          class Hash = std::hash< Key >,
          class KeyEqual = std::equal_to< Key >,
          class Allocator = std::allocator< std::pair<const Key, Index> > >
class UnorderedIndexedSet
{
protected:
   using map_type = std::unordered_map< Key, Index, Hash, KeyEqual, Allocator >;
   map_type map;

public:
   using key_type = Key;
   using index_type = Index;
   using value_type = typename map_type::value_type;
   using size_type = typename map_type::size_type;
   using iterator = typename map_type::iterator;
   using const_iterator = typename map_type::const_iterator;
   using hasher = Hash;
   using key_equal = KeyEqual;
   
   void clear();

   size_type size() const;

   Index insert( const Key& key );

   Index insert( Key&& key );

   std::pair< Index, bool > try_insert( const Key& key );

   bool find( const Key& key, Index& index ) const;

   void reserve( size_type count );

   size_type count( const Key& key ) const;

   size_type erase( const Key& key );

   void print( std::ostream& str ) const;

   iterator begin();

   const_iterator begin() const;

   iterator end();

   const_iterator end() const;
};

template< typename Element,
          typename Index >
std::ostream& operator <<( std::ostream& str, UnorderedIndexedSet< Element, Index >& set );

} // namespace Containers
} // namespace TNL

#include <TNL/Containers/UnorderedIndexedSet.hpp>
