// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <iostream>

namespace noaTNL {
namespace Containers {

template< typename Value,
          typename Index,
          typename Key >
class IndexedMap
{
public:
   using ValueType = Value;
   using IndexType = Index;
   using KeyType = Key;

   void reset();

   IndexType getSize() const;

   IndexType insert( const ValueType &data );

   bool find( const ValueType &data, IndexType& index ) const;

   template< typename ArrayType >
   void toArray( ArrayType& array ) const;

   const Value& getElement( KeyType key ) const;

   Value& getElement( KeyType key );

   void print( std::ostream& str ) const;

protected:
   struct DataWithIndex
   {
      // This constructor is here only because of bug in g++, we might fix it later.
      // http://stackoverflow.com/questions/22357887/comparing-two-mapiterators-why-does-it-need-the-copy-constructor-of-stdpair
      DataWithIndex(){};

      DataWithIndex( const DataWithIndex& d ) : data( d.data ), index( d.index) {}

      explicit DataWithIndex( const Value data) : data( data ) {}

      DataWithIndex( const Value data,
                     const Index index) : data(data), index(index) {}

      Value data;
      Index index;
   };

   using STDMapType = std::map< Key, DataWithIndex >;
   using STDMapValueType = typename STDMapType::value_type;
   using STDMapIteratorType = typename STDMapType::const_iterator;

   STDMapType map;
};

template< typename Value,
          typename Index,
          typename Key >
std::ostream& operator <<( std::ostream& str, IndexedMap< Value, Index, Key >& set );

} // namespace Containers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Containers/IndexedMap_impl.h>
