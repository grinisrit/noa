// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/XMLVTK.h>
#include <cstdint>

namespace noa::TNL {
namespace Meshes {
namespace Readers {

class VTIReader : public XMLVTK
{
#ifdef HAVE_TINYXML2
   void
   readImageData()
   {
      // read the required attributes
      const std::string extent = getAttributeString( datasetElement, "WholeExtent" );
      const std::string origin = getAttributeString( datasetElement, "Origin" );
      const std::string spacing = getAttributeString( datasetElement, "Spacing" );

      // check the <Piece> tag
      using namespace tinyxml2;
      const XMLElement* piece = getChildSafe( datasetElement, "Piece" );
      if( piece->NextSiblingElement( "Piece" ) != nullptr )
         // ambiguity - throw error, we don't know which piece to parse (or all of them?)
         throw MeshReaderError( "VTIReader", "the serial ImageData file contains more than one <Piece> element" );
      const std::string pieceExtent = getAttributeString( piece, "Extent" );
      if( pieceExtent != extent )
         throw MeshReaderError(
            "VTIReader", "the <Piece> element has different extent than <ImageData> (" + pieceExtent + " vs " + extent + ")" );

      // parse the extent
      {
         std::stringstream ss( extent );
         gridExtent.resize( 6, 0 );
         for( int i = 0; i < 6; i++ ) {
            ss >> gridExtent[ i ];
            // check conversion error
            if( ! ss.good() )
               throw MeshReaderError( "VTIReader", "invalid extent: not a number: " + extent );
         }
         // check remaining characters
         std::string suffix;
         ss >> std::ws >> suffix;
         if( ! suffix.empty() )
            throw MeshReaderError( "VTIReader", "invalid extent " + extent + ": trailing characters: " + suffix );
      }

      // parse the origin
      {
         std::stringstream ss( origin );
         gridOrigin.resize( 3 );
         for( int i = 0; i < 3; i++ ) {
            ss >> gridOrigin[ i ];
            // check conversion error
            if( ! ss.good() )
               throw MeshReaderError( "VTIReader", "invalid origin: not a number: " + origin );
         }
         // check remaining characters
         std::string suffix;
         ss >> std::ws >> suffix;
         if( ! suffix.empty() )
            throw MeshReaderError( "VTIReader", "invalid origin " + origin + ": trailing characters: " + suffix );
      }

      // parse the spacing
      {
         std::stringstream ss( spacing );
         gridSpacing.resize( 3 );
         for( int i = 0; i < 3; i++ ) {
            ss >> gridSpacing[ i ];
            // check conversion error
            if( ! ss.good() )
               throw MeshReaderError( "VTIReader", "invalid spacing: not a number: " + spacing );
            // check negative numbers
            if( gridSpacing[ i ] < 0 )
               throw MeshReaderError( "VTIReader", "invalid spacing: negative number: " + spacing );
         }
         // check remaining characters
         std::string suffix;
         ss >> std::ws >> suffix;
         if( ! suffix.empty() )
            throw MeshReaderError( "VTIReader", "invalid spacing " + spacing + ": trailing characters: " + suffix );
      }

      // determine the grid dimension
      int dim = 0;
      for( int i = 0; i < 3; i++ )
         if( gridSpacing[ i ] > 0 )
            dim++;
         else
            break;
      spaceDimension = meshDimension = dim;

      // populate cellShape (just for completeness, not necessary for GridTypeResolver)
      if( meshDimension == 1 )
         cellShape = VTK::EntityShape::Line;
      else if( meshDimension == 2 )
         cellShape = VTK::EntityShape::Pixel;
      else if( meshDimension == 3 )
         cellShape = VTK::EntityShape::Voxel;

      // RealType is not stored in the VTI file, so just set the default
      pointsType = "double";

      // set the index type
      connectivityType = headerType;
   }
#endif

public:
   VTIReader() = default;

   VTIReader( const std::string& fileName ) : XMLVTK( fileName ) {}

   void
   detectMesh() override
   {
#ifdef HAVE_TINYXML2
      reset();
      try {
         openVTKFile();
      }
      catch( const MeshReaderError& ) {
         reset();
         throw;
      }

      // verify file type
      if( fileType == "ImageData" )
         readImageData();
      else
         throw MeshReaderError(
            "VTIReader", "the reader cannot read data of the type " + fileType + ". Use a different reader if possible." );

      // indicate success by setting the mesh type
      meshType = "Meshes::Grid";
#else
      throw_no_tinyxml();
#endif
   }
};

}  // namespace Readers
}  // namespace Meshes
}  // namespace noa::TNL
