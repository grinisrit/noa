// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Images/PNGImage.h>

namespace noa::TNL {
namespace Images {

template< typename Index >
PNGImage< Index >::PNGImage() : fileOpen( false )
{}

template< typename Index >
bool
PNGImage< Index >::readHeader()
{
#ifdef HAVE_PNG_H
   /***
    * Check if it is a PNG image.
    */
   const int headerSize( 8 );
   png_byte header[ headerSize ];
   if( fread( header, sizeof( char ), headerSize, this->file ) != headerSize ) {
      std::cerr << "I am not able to read PNG image header." << std::endl;
      return false;
   }
   bool isPNG = ! png_sig_cmp( header, 0, headerSize );
   if( ! isPNG )
      return false;

   /****
    * Allocate necessary memory
    */
   this->png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr );
   if( ! this->png_ptr )
      return false;

   this->info_ptr = png_create_info_struct( this->png_ptr );
   if( ! this->info_ptr ) {
      png_destroy_read_struct( &this->png_ptr, nullptr, nullptr );
      return false;
   }

   this->end_info = png_create_info_struct( this->png_ptr );
   if( ! this->end_info ) {
      png_destroy_read_struct( &this->png_ptr, &this->info_ptr, nullptr );
      return false;
   }

   /***
    * Prepare the long jump back from libpng.
    */
   if( setjmp( png_jmpbuf( this->png_ptr ) ) ) {
      png_destroy_read_struct( &this->png_ptr, &this->info_ptr, &end_info );
      return false;
   }
   png_init_io( this->png_ptr, this->file );
   png_set_sig_bytes( this->png_ptr, headerSize );

   /****
    * Read the header
    */
   png_read_png( this->png_ptr, this->info_ptr, PNG_TRANSFORM_IDENTITY, nullptr );
   this->height = (Index) png_get_image_height( this->png_ptr, this->info_ptr );
   this->width = (Index) png_get_image_width( this->png_ptr, this->info_ptr );
   this->bit_depth = png_get_bit_depth( this->png_ptr, this->info_ptr );
   this->color_type = png_get_color_type( this->png_ptr, this->info_ptr );
   std::cout << this->height << " x " << this->width << std::endl;
   return true;
#else
   // cerr << "TNL was not compiled with support of PNG. You may still use PGM format." << std::endl;
   return false;
#endif
}

template< typename Index >
bool
PNGImage< Index >::openForRead( const String& fileName )
{
   this->close();
   this->file = fopen( fileName.getString(), "r" );
   if( ! this->file ) {
      std::cerr << "Unable to open the file " << fileName << std::endl;
      return false;
   }
   this->fileOpen = true;
   if( ! readHeader() )
      return false;
   return true;
}

template< typename Index >
template< typename MeshReal, typename Device, typename Real >
bool
PNGImage< Index >::read( const RegionOfInterest< Index > roi,
                         Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function )
{
#ifdef HAVE_PNG_H
   using GridType = Meshes::Grid< 2, MeshReal, Device, Index >;
   const GridType& grid = function.getMesh();
   typename GridType::Cell cell( grid );

   /***
    * Prepare the long jump back from libpng.
    */
   if( setjmp( png_jmpbuf( this->png_ptr ) ) ) {
      png_destroy_read_struct( &this->png_ptr, &this->info_ptr, &this->end_info );
      return false;
   }

   png_bytepp row_pointers = png_get_rows( this->png_ptr, this->info_ptr );

   Index i, j;
   for( i = 0; i < this->height; i++ ) {
      for( j = 0; j < this->width; j++ ) {
         if( ! roi.isIn( i, j ) )
            continue;

         cell.getCoordinates().x() = j - roi.getLeft();
         cell.getCoordinates().y() = roi.getBottom() - 1 - i;
         cell.refresh();
         // Index cellIndex = grid.getCellIndex( CoordinatesType( j - roi.getLeft(),
         //                                      roi.getBottom() - 1 - i ) );
         unsigned char char_color[ 4 ];
         unsigned short int int_color[ 4 ];
         switch( this->color_type ) {
            case PNG_COLOR_TYPE_GRAY:
               if( this->bit_depth == 8 ) {
                  char_color[ 0 ] = row_pointers[ i ][ j ];
                  Real value = char_color[ 0 ] / (Real) 255.0;
                  function.getData().setElement( cell.getIndex(), value );
               }
               if( this->bit_depth == 16 ) {
                  int_color[ 0 ] = row_pointers[ i ][ j ];
                  Real value = int_color[ 0 ] / (Real) 65535.0;
                  function.getData().setElement( cell.getIndex(), value );
               }
               break;
            case PNG_COLOR_TYPE_RGB:
               if( this->bit_depth == 8 ) {
                  unsigned char* row = (unsigned char*) row_pointers[ i ];
                  char_color[ 0 ] = row[ 3 * j ];
                  char_color[ 1 ] = row[ 3 * j + 1 ];
                  char_color[ 2 ] = row[ 3 * j + 2 ];
                  Real r = char_color[ 0 ] / (Real) 255.0;
                  Real g = char_color[ 1 ] / (Real) 255.0;
                  Real b = char_color[ 2 ] / (Real) 255.0;
                  Real value = 0.2989 * r + 0.5870 * g + 0.1140 * b;
                  function.getData().setElement( cell.getIndex(), value );
               }
               if( this->bit_depth == 16 ) {
                  unsigned short int* row = (unsigned short int*) row_pointers[ i ];
                  int_color[ 0 ] = row[ 3 * j ];
                  int_color[ 1 ] = row[ 3 * j + 1 ];
                  int_color[ 2 ] = row[ 3 * j + 2 ];
                  Real r = int_color[ 0 ] / (Real) 65535.0;
                  Real g = int_color[ 1 ] / (Real) 66355.0;
                  Real b = int_color[ 2 ] / (Real) 65535.0;
                  Real value = 0.2989 * r + 0.5870 * g + 0.1140 * b;
                  function.getData().setElement( cell.getIndex(), value );
               }
               break;
            default:
               std::cerr << "Unknown PNG color type." << std::endl;
               return false;
         }
      }
   }
   return true;
#else
   // cerr << "TNL was not compiled with support of PNG. You may still use PGM format." << std::endl;
   return false;
#endif
}

template< typename Index >
template< typename Real, typename Device >
bool
PNGImage< Index >::writeHeader( const Meshes::Grid< 2, Real, Device, Index >& grid )
{
#ifdef HAVE_PNG_H
   this->png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr );
   if( ! png_ptr )
      return false;

   this->info_ptr = png_create_info_struct( this->png_ptr );
   if( ! this->info_ptr ) {
      png_destroy_write_struct( &this->png_ptr, nullptr );
      return false;
   }

   /***
    * Prepare the long jump back from libpng.
    */
   if( setjmp( png_jmpbuf( this->png_ptr ) ) ) {
      png_destroy_read_struct( &this->png_ptr, &this->info_ptr, &this->end_info );
      return false;
   }

   /****
    * Set the zlib compression level
    */
   // png_set_compression_level( this->png_ptr, Z_BEST_COMPRESSION );

   // const int bitDepth( 8 );
   png_set_IHDR( this->png_ptr,
                 this->info_ptr,
                 grid.getDimensions().x(),
                 grid.getDimensions().y(),
                 8,  // bitDepth,
                 PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT );
   png_init_io( this->png_ptr, this->file );
   png_write_info( png_ptr, info_ptr );
   return true;
#else
   std::cerr << "TNL was not compiled with support of PNG. You may still use PGM format." << std::endl;
   return false;
#endif
}

template< typename Index >
template< typename Real, typename Device >
bool
PNGImage< Index >::openForWrite( const String& fileName, Meshes::Grid< 2, Real, Device, Index >& grid )
{
   this->close();
   this->file = fopen( fileName.getString(), "w" );
   if( ! this->file ) {
      std::cerr << "Unable to open the file " << fileName << std::endl;
      return false;
   }
   this->fileOpen = true;
   if( ! writeHeader( grid ) )
      return false;
   return true;
}

template< typename Index >
template< typename Real, typename Device, typename Vector >
bool
PNGImage< Index >::write( const Meshes::Grid< 2, Real, Device, Index >& grid, Vector& vector )
{
#ifdef HAVE_PNG_H
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   typename GridType::Cell cell( grid );

   /***
    * Prepare the long jump back from libpng.
    */
   if( setjmp( png_jmpbuf( this->png_ptr ) ) ) {
      png_destroy_read_struct( &this->png_ptr, &this->info_ptr, &this->end_info );
      return false;
   }

   Index i, j;
   png_bytep row = new png_byte[ 3 * grid.getDimensions().x() ];
   for( i = 0; i < grid.getDimensions().y(); i++ ) {
      for( j = 0; j < grid.getDimensions().x(); j++ ) {
         cell.getCoordinates().x() = j;
         cell.getCoordinates().y() = grid.getDimensions().y() - 1 - i;

         // Index cellIndex = grid.getCellIndex( CoordinatesType( j,
         //                                      grid.getDimensions().y() - 1 - i ) );

         row[ j ] = 255 * vector.getElement( cell.getIndex() );
      }
      png_write_row( this->png_ptr, row );
   }
   delete[] row;
   return true;
#else
   return false;
#endif
}

template< typename Index >
template< typename MeshReal, typename Device, typename Real >
bool
PNGImage< Index >::write( const Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function )
{
#ifdef HAVE_PNG_H
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   const GridType& grid = function.getMesh();
   typename GridType::Cell cell( grid );

   /***
    * Prepare the long jump back from libpng.
    */
   if( setjmp( png_jmpbuf( this->png_ptr ) ) ) {
      png_destroy_read_struct( &this->png_ptr, &this->info_ptr, &this->end_info );
      return false;
   }

   Index i, j;
   png_bytep row = new png_byte[ 3 * grid.getDimensions().x() ];
   for( i = 0; i < grid.getDimensions().y(); i++ ) {
      for( j = 0; j < grid.getDimensions().x(); j++ ) {
         cell.getCoordinates().x() = j;
         cell.getCoordinates().y() = grid.getDimensions().y() - 1 - i;

         // Index cellIndex = grid.getCellIndex( CoordinatesType( j,
         //                                      grid.getDimensions().y() - 1 - i ) );

         row[ j ] = 255 * function.getData().getElement( cell.getIndex() );
      }
      png_write_row( this->png_ptr, row );
   }
   delete[] row;
   return true;
#else
   return false;
#endif
}

template< typename Index >
void
PNGImage< Index >::close()
{
   if( this->fileOpen )
      fclose( file );
   this->fileOpen = false;
}

template< typename Index >
PNGImage< Index >::~PNGImage()
{
   close();
}

}  // namespace Images
}  // namespace noa::TNL
