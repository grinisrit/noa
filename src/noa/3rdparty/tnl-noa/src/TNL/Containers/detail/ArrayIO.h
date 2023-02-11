// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Object.h>
#include <noa/3rdparty/tnl-noa/src/TNL/File.h>
#include <noa/3rdparty/tnl-noa/src/TNL/TypeInfo.h>

namespace noa::TNL {
namespace Containers {
namespace detail {

template< typename Value, typename Index, typename Allocator, bool Elementwise = std::is_base_of< Object, Value >::value >
struct ArrayIO
{};

template< typename Value, typename Index, typename Allocator >
struct ArrayIO< Value, Index, Allocator, true >
{
   static std::string
   getSerializationType()
   {
      return "TNL::Containers::Array< " + TNL::getSerializationType< Value >() + " >";
   }

   static void
   save( File& file, const Value* data, Index elements )
   {
      Index i;
      try {
         for( i = 0; i < elements; i++ )
            data[ i ].save( file );
      }
      catch( ... ) {
         throw Exceptions::FileSerializationError( file.getFileName(),
                                                   "unable to write the " + std::to_string( i ) + "-th array element of "
                                                      + std::to_string( elements ) + " into the file." );
      }
   }

   static void
   load( File& file, Value* data, Index elements )
   {
      Index i = 0;
      try {
         for( i = 0; i < elements; i++ )
            data[ i ].load( file );
      }
      catch( ... ) {
         throw Exceptions::FileDeserializationError( file.getFileName(),
                                                     "unable to read the " + std::to_string( i ) + "-th array element of "
                                                        + std::to_string( elements ) + " from the file." );
      }
   }

   static void
   loadSubrange( File& file, std::size_t elementsInFile, Index offset, Value* data, Index size )
   {
      if( std::size_t( offset + size ) > elementsInFile )
         throw Exceptions::FileDeserializationError(
            file.getFileName(), "unable to read subrange of array elements: offset + size > elementsInFile" );

      if( size == 0 ) {
         file.ignore< Value >( elementsInFile );
         return;
      }

      try {
         file.ignore< Value >( offset );
         load( file, data, size );
         file.ignore< Value >( elementsInFile - offset - size );
      }
      catch( ... ) {
         throw Exceptions::FileDeserializationError( file.getFileName(),
                                                     "unable to read array elements in the subrange ["
                                                        + std::to_string( offset ) + ", " + std::to_string( offset + size )
                                                        + ") from the file." );
      }
   }
};

template< typename Value, typename Index, typename Allocator >
struct ArrayIO< Value, Index, Allocator, false >
{
   static std::string
   getSerializationType()
   {
      return "TNL::Containers::Array< " + TNL::getSerializationType< Value >() + " >";
   }

   template< typename TargetValue = Value >
   static void
   save( File& file, const Value* data, Index elements )
   {
      if( elements == 0 )
         return;
      try {
         file.save< Value, TargetValue, Allocator >( data, elements );
      }
      catch( ... ) {
         throw Exceptions::FileSerializationError(
            file.getFileName(), "unable to write " + std::to_string( elements ) + " array elements into the file." );
      }
   }

   static void
   save( File& file, const Value* data, Index elements, const std::string& typeInFile )
   {
      if( typeInFile == getType< Value >() )
         save< Value >( file, data, elements );
      // check fundamental types for type casting
      else if( typeInFile == getType< std::int8_t >() )
         save< std::int8_t >( file, data, elements );
      else if( typeInFile == getType< std::uint8_t >() )
         save< std::uint8_t >( file, data, elements );
      else if( typeInFile == getType< std::int16_t >() )
         save< std::int16_t >( file, data, elements );
      else if( typeInFile == getType< std::uint16_t >() )
         save< std::uint16_t >( file, data, elements );
      else if( typeInFile == getType< std::int32_t >() )
         save< std::int32_t >( file, data, elements );
      else if( typeInFile == getType< std::uint32_t >() )
         save< std::uint32_t >( file, data, elements );
      else if( typeInFile == getType< std::int64_t >() )
         save< std::int64_t >( file, data, elements );
      else if( typeInFile == getType< std::uint64_t >() )
         save< std::uint64_t >( file, data, elements );
      else if( typeInFile == getType< float >() )
         save< float >( file, data, elements );
      else if( typeInFile == getType< double >() )
         save< double >( file, data, elements );
      else
         throw Exceptions::FileSerializationError(
            file.getFileName(),
            "value type " + getType< Value >() + " cannot be type-cast to the requested type " + std::string( typeInFile ) );
   }

   template< typename SourceValue = Value >
   static void
   load( File& file, Value* data, Index elements )
   {
      if( elements == 0 )
         return;
      try {
         file.load< Value, SourceValue, Allocator >( data, elements );
      }
      catch( ... ) {
         throw Exceptions::FileDeserializationError(
            file.getFileName(), "unable to read " + std::to_string( elements ) + " array elements from the file." );
      }
   }

   static void
   load( File& file, Value* data, Index elements, const std::string& typeInFile )
   {
      if( typeInFile == getType< Value >() )
         load( file, data, elements );
      // check fundamental types for type casting
      else if( typeInFile == getType< std::int8_t >() )
         load< std::int8_t >( file, data, elements );
      else if( typeInFile == getType< std::uint8_t >() )
         load< std::uint8_t >( file, data, elements );
      else if( typeInFile == getType< std::int16_t >() )
         load< std::int16_t >( file, data, elements );
      else if( typeInFile == getType< std::uint16_t >() )
         load< std::uint16_t >( file, data, elements );
      else if( typeInFile == getType< std::int32_t >() )
         load< std::int32_t >( file, data, elements );
      else if( typeInFile == getType< std::uint32_t >() )
         load< std::uint32_t >( file, data, elements );
      else if( typeInFile == getType< std::int64_t >() )
         load< std::int64_t >( file, data, elements );
      else if( typeInFile == getType< std::uint64_t >() )
         load< std::uint64_t >( file, data, elements );
      else if( typeInFile == getType< float >() )
         load< float >( file, data, elements );
      else if( typeInFile == getType< double >() )
         load< double >( file, data, elements );
      else
         throw Exceptions::FileDeserializationError( file.getFileName(),
                                                     "value type " + std::string( typeInFile )
                                                        + " cannot be type-cast to the requested type " + getType< Value >() );
   }

   template< typename SourceValue = Value >
   static void
   loadSubrange( File& file, std::size_t elementsInFile, Index offset, Value* data, Index size )
   {
      if( std::size_t( offset + size ) > elementsInFile )
         throw Exceptions::FileDeserializationError(
            file.getFileName(),
            "unable to read subrange of array elements: offset + size > elementsInFile: " + std::to_string( offset ) + " + "
               + std::to_string( size ) + " > " + std::to_string( elementsInFile ) );

      if( size == 0 ) {
         file.ignore< SourceValue >( elementsInFile );
         return;
      }

      try {
         file.ignore< SourceValue >( offset );
         load< SourceValue >( file, data, size );
         file.ignore< SourceValue >( elementsInFile - offset - size );
      }
      catch( ... ) {
         throw Exceptions::FileDeserializationError( file.getFileName(),
                                                     "unable to read array elements in the subrange ["
                                                        + std::to_string( offset ) + ", " + std::to_string( offset + size )
                                                        + ") from the file." );
      }
   }

   static void
   loadSubrange( File& file, std::size_t elementsInFile, Index offset, Value* data, Index size, const std::string& typeInFile )
   {
      if( typeInFile == getType< Value >() )
         loadSubrange( file, elementsInFile, offset, data, size );
      // check fundamental types for type casting
      else if( typeInFile == getType< std::int8_t >() )
         loadSubrange< std::int8_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< std::uint8_t >() )
         loadSubrange< std::uint8_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< std::int16_t >() )
         loadSubrange< std::int16_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< std::uint16_t >() )
         loadSubrange< std::uint16_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< std::int32_t >() )
         loadSubrange< std::int32_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< std::uint32_t >() )
         loadSubrange< std::uint32_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< std::int64_t >() )
         loadSubrange< std::int64_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< std::uint64_t >() )
         loadSubrange< std::uint64_t >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< long >() )  // this is for MacOS on ARM
         loadSubrange< long >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< float >() )
         loadSubrange< float >( file, elementsInFile, offset, data, size );
      else if( typeInFile == getType< double >() )
         loadSubrange< double >( file, elementsInFile, offset, data, size );
      else
         throw Exceptions::FileDeserializationError( file.getFileName(),
                                                     "value type " + std::string( typeInFile )
                                                        + " cannot be type-cast to the requested type " + getType< Value >() );
   }
};

}  // namespace detail
}  // namespace Containers
}  // namespace noa::TNL
