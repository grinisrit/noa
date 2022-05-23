// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_JPEG_H
   #include <jpeglib.h>
#endif

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/Image.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/RegionOfInterest.h>

#ifdef HAVE_JPEG_H
struct my_error_mgr
{
   jpeg_error_mgr pub;
   jmp_buf setjmp_buffer;
};
#endif

namespace noa::TNL {
namespace Images {

template< typename Index = int >
class JPEGImage : public Image< Index >
{
public:
   using IndexType = Index;

   JPEGImage();

   bool
   openForRead( const String& fileName );

   template< typename MeshReal, typename Device, typename Real >
   bool
   read( RegionOfInterest< Index > roi,
         Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function );

   template< typename Real, typename Device >
   bool
   openForWrite( const String& fileName, Meshes::Grid< 2, Real, Device, Index >& grid );

   // TODO: Obsolete
   template< typename Real, typename Device, typename Vector >
   bool
   write( const Meshes::Grid< 2, Real, Device, Index >& grid, Vector& vector );

   template< typename MeshReal, typename Device, typename Real >
   bool
   write( const Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function );

   void
   close();

   ~JPEGImage();

protected:
   bool
   readHeader();

   template< typename Real, typename Device >
   bool
   writeHeader( const Meshes::Grid< 2, Real, Device, Index >& grid );

   FILE* file;

   bool fileOpen;

#ifdef HAVE_JPEG_H
   my_error_mgr jerr;
   jpeg_decompress_struct decinfo;
   jpeg_compress_struct cinfo;
   int components;
   J_COLOR_SPACE color_space;
#endif
};

}  // namespace Images
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Images/JPEGImage_impl.h>
