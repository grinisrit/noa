// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_PNG_H
   #include <png.h>
#endif

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/Image.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/RegionOfInterest.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/MeshFunction.h>

namespace noa::TNL {
namespace Images {

template< typename Index = int >
class PNGImage : public Image< Index >
{
public:
   using IndexType = Index;

   PNGImage();

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

   ~PNGImage();

protected:
   bool
   readHeader();

   template< typename Real, typename Device >
   bool
   writeHeader( const Meshes::Grid< 2, Real, Device, Index >& grid );

   FILE* file;

   bool fileOpen;

#ifdef HAVE_PNG_H
   png_structp png_ptr;

   png_infop info_ptr, end_info;

   png_byte color_type, bit_depth;
#endif
};

}  // namespace Images
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Images/PNGImage_impl.h>
