// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <ostream>

namespace noa::TNL {
namespace Containers {

template< class Key,
          class Index,
          class Compare = std::less< Key >,
          class Allocator = std::allocator< std::pair< const Key, Index > > >
class IndexedSet
{
protected:
   using map_type = std::map< Key, Index, Compare, Allocator >;
   map_type map;

public:
   using key_type = Key;
   using index_type = Index;
   using value_type = typename map_type::value_type;
   using size_type = typename map_type::size_type;

   void
   clear();

   size_type
   size() const;

   Index
   insert( const Key& key );

   bool
   find( const Key& key, Index& index ) const;

   size_type
   count( const Key& key ) const;

   size_type
   erase( const Key& key );

   void
   print( std::ostream& str ) const;
};

template< typename Element, typename Index >
std::ostream&
operator<<( std::ostream& str, IndexedSet< Element, Index >& set );

}  // namespace Containers
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/IndexedSet_impl.h>
