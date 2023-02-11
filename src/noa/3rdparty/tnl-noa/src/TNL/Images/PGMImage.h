// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Images/Image.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Images/RegionOfInterest.h>

namespace noa::TNL {
namespace Images {

template< typename Index = int >
class PGMImage : public Image< Index >
{
public:
   using IndexType = Index;

   PGMImage();

   bool
   openForRead( const String& fileName );

   template< typename MeshReal, typename Device, typename Real >
   bool
   read( RegionOfInterest< Index > roi,
         Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function );

   template< typename Real, typename Device >
   bool
   openForWrite( const String& fileName, Meshes::Grid< 2, Real, Device, Index >& grid, bool binary = true );

   // TODO: obsolete
   template< typename Real, typename Device, typename Vector >
   bool
   write( const Meshes::Grid< 2, Real, Device, Index >& grid, Vector& vector );

   template< typename MeshReal, typename Device, typename Real >
   bool
   write( const Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function );

   void
   close();

   ~PGMImage();

protected:
   bool
   readHeader();

   template< typename Real, typename Device >
   bool
   writeHeader( const Meshes::Grid< 2, Real, Device, Index >& grid, bool binary );

   bool binary;

   IndexType maxColors;

   std::fstream file;

   bool fileOpen;
};

}  // namespace Images
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Images/PGMImage_impl.h>
