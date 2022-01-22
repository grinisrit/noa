// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <cstring>

#include <noa/3rdparty/TNL/Object.h>

namespace noaTNL {

static constexpr char magic_number[] = "TNLMN";

inline String Object::getSerializationType()
{
   return String( "Object" );
}

inline String Object::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

inline void Object::save( File& file ) const
{
   file.save( magic_number, strlen( magic_number ) );
   file << this->getSerializationTypeVirtual();
}

inline void Object::load( File& file )
{
   const String objectType = getObjectType( file );
   if( objectType != this->getSerializationTypeVirtual() )
      throw Exceptions::FileDeserializationError( file.getFileName(), "object type does not match (expected " + this->getSerializationTypeVirtual() + ", found " + objectType + ")." );
}

inline void Object::save( const String& fileName ) const
{
   File file;
   file.open( fileName, std::ios_base::out );
   this->save( file );
}

inline void Object::load( const String& fileName )
{
   File file;
   file.open( fileName, std::ios_base::in );
   this->load( file );
}

inline String getObjectType( File& file )
{
   char mn[ 10 ];
   String type;
   file.load( mn, strlen( magic_number ) );
   if( strncmp( mn, magic_number, 5 ) != 0 )
      throw Exceptions::FileDeserializationError( file.getFileName(), "wrong magic number - file is not in a TNL-compatible format." );
   file >> type;
   return type;
}

inline String getObjectType( const String& fileName )
{
   File binaryFile;
   binaryFile.open( fileName, std::ios_base::in );
   return getObjectType( binaryFile );
}

inline std::vector< String >
parseObjectType( const String& objectType )
{
   std::vector< String > parsedObjectType;
   const int objectTypeLength = objectType.getLength();
   int i = 0;
   /****
    * The object type consists of the following:
    * objectName< param1, param2, param3, .. >.
    * We first extract the objectName by finding the first
    * character '<'.
    */
   while( i < objectTypeLength && objectType[ i ] != '<' )
      i++;
   String objectName = objectType.substr( 0, i );
   parsedObjectType.push_back( objectName );
   i++;

   /****
    * Now, we will extract the parameters.
    * Each parameter can be template, so we must count and pair
    * '<' with '>'.
    */
   int templateBrackets( 0 );
   String buffer( "" );

   while( i < objectTypeLength )
   {
      if( objectType[ i ] == '<' )
         templateBrackets++;
      if( ! templateBrackets )
      {
         if( objectType[ i ] == ',' ||
             objectType[ i ] == '>' )
         {
            if( buffer != "" )
            {
               parsedObjectType.push_back( buffer.strip( ' ' ) );
               buffer.clear();
            }
         }
         else buffer += objectType[ i ];
      }
      else buffer += objectType[ i ];
      if( objectType[ i ] == '>' )
         templateBrackets--;
      i++;
   }

   return parsedObjectType;
}

inline void saveObjectType( File& file, const String& type )
{
   file.save( magic_number, strlen( magic_number ) );
   file << type;
}

} // namespace noaTNL
