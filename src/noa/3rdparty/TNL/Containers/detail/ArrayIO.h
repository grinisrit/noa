// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/TNL/Object.h>
#include <noa/3rdparty/TNL/File.h>
#include <noa/3rdparty/TNL/TypeInfo.h>

namespace noa::TNL {
namespace Containers {
namespace detail {

template< typename Value,
          typename Index,
          typename Allocator,
          bool Elementwise = std::is_base_of< Object, Value >::value >
struct ArrayIO
{};

template< typename Value,
          typename Index,
          typename Allocator >
struct ArrayIO< Value, Index, Allocator, true >
{
   static String getSerializationType()
   {
      return String( "Containers::Array< " ) +
             noa::TNL::getSerializationType< Value >() + ", [any_device], " +
             noa::TNL::getSerializationType< Index >() + ", [any_allocator] >";
   }

   static void save( File& file,
                     const Value* data,
                     const Index elements )
   {
      Index i;
      try
      {
         for( i = 0; i < elements; i++ )
            data[ i ].save( file );
      }
      catch(...)
      {
         throw Exceptions::FileSerializationError( file.getFileName(), "unable to write the " + std::to_string(i) + "-th array element of " + std::to_string(elements) + " into the file." );
      }
   }

   static void load( File& file,
                     Value* data,
                     const Index elements )
   {
      Index i = 0;
      try
      {
         for( i = 0; i < elements; i++ )
            data[ i ].load( file );
      }
      catch(...)
      {
         throw Exceptions::FileDeserializationError( file.getFileName(), "unable to read the " + std::to_string(i) + "-th array element of " + std::to_string(elements) + " from the file." );
      }
   }
};

template< typename Value,
          typename Index,
          typename Allocator >
struct ArrayIO< Value, Index, Allocator, false >
{
   static String getSerializationType()
   {
      return String( "Containers::Array< " ) +
             noa::TNL::getSerializationType< Value >() + ", [any_device], " +
             noa::TNL::getSerializationType< Index >() + ", [any_allocator] >";
   }

   static void save( File& file,
                     const Value* data,
                     const Index elements )
   {
      if( elements == 0 )
         return;
      try
      {
         file.save< Value, Value, Allocator >( data, elements );
      }
      catch(...)
      {
         throw Exceptions::FileSerializationError( file.getFileName(), "unable to write " + std::to_string(elements) + " array elements into the file." );
      }
   }

   static void load( File& file,
                     Value* data,
                     const Index elements )
   {
      if( elements == 0 )
         return;
      try
      {
         file.load< Value, Value, Allocator >( data, elements );
      }
      catch(...)
      {
         throw Exceptions::FileDeserializationError( file.getFileName(), "unable to read " + std::to_string(elements) + " array elements from the file." );
      }
   }
};

} // namespace detail
} // namespace Containers
} // namespace noa::TNL
