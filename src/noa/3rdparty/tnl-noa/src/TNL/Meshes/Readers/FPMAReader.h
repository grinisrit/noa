// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <sstream>
#include <vector>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/MeshReader.h>

namespace noa::TNL {
namespace Meshes {
namespace Readers {

class FPMAReader : public MeshReader
{
public:
   FPMAReader() = default;

   FPMAReader( const std::string& fileName ) : MeshReader( fileName ) {}

   void
   detectMesh() override
   {
      reset();

      std::ifstream inputFile( fileName );
      if( ! inputFile )
         throw MeshReaderError( "FPMAReader", "failed to open the file '" + fileName + "'." );

      std::string line;
      std::istringstream iss;

      // fpma format doesn't provide types
      using PointType = double;
      pointsType = "double";
      connectivityType = offsetsType = "std::int32_t";

      // it is expected, that fpma format always stores polyhedral mesh
      spaceDimension = meshDimension = 3;
      cellShape = VTK::EntityShape::Polyhedron;

      // arrays holding the data from the file
      std::vector< double > pointsArray;
      std::vector< std::int32_t > cellConnectivityArray;
      std::vector< std::int32_t > cellOffsetsArray;
      std::vector< std::int32_t > faceConnectivityArray;
      std::vector< std::int32_t > faceOffsetsArray;

      // read number of points
      NumberOfPoints = readValue< decltype( NumberOfPoints ) >( inputFile );
      if( ! inputFile ) {
         reset();
         throw MeshReaderError( "FPMAReader", "unable to read number of points, the file may be invalid or corrupted." );
      }
      pointsArray.reserve( NumberOfPoints );
      // read points
      for( std::size_t pointIndex = 0; pointIndex < NumberOfPoints; pointIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "FPMAReader", "unable to read enough vertices, the file may be invalid or corrupted." );
         }

         // read the coordinates of a point
         for( int i = 0; i < 3; i++ ) {
            PointType aux = readValue< PointType >( inputFile );
            if( ! inputFile ) {
               reset();
               throw MeshReaderError( "FPMAReader",
                                      "unable to read " + std::to_string( i ) + "th component of the vertex number "
                                         + std::to_string( pointIndex ) + "." );
            }
            pointsArray.emplace_back( aux );
         }
      }

      // read number of faces
      NumberOfFaces = readValue< decltype( NumberOfFaces ) >( inputFile );
      if( ! inputFile ) {
         reset();
         throw MeshReaderError( "FPMAReader", "unable to read number of faces, the file may be invalid or corrupted." );
      }

      // read faces
      faceConnectivityArray.reserve( NumberOfFaces );
      faceOffsetsArray.reserve( NumberOfFaces );
      for( std::size_t faceIndex = 0; faceIndex < NumberOfFaces; faceIndex++ ) {
         // read number of points of a face
         const auto numberOfFacePoints = readValue< std::size_t >( inputFile );
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "FPMAReader", "unable to read enough faces, the file may be invalid or corrupted." );
         }

         // read points of a face
         for( std::size_t i = 0; i < numberOfFacePoints; i++ ) {
            const auto pointIndex = readValue< std::size_t >( inputFile );
            if( ! inputFile ) {
               reset();
               throw MeshReaderError( "FPMAReader",
                                      "unable to read " + std::to_string( i ) + "th component of the face number "
                                         + std::to_string( faceIndex ) + "." );
            }
            faceConnectivityArray.emplace_back( pointIndex );
         }

         faceOffsetsArray.emplace_back( faceConnectivityArray.size() );
      }

      // read number of cells
      NumberOfCells = readValue< decltype( NumberOfCells ) >( inputFile );
      if( ! inputFile ) {
         reset();
         throw MeshReaderError( "FPMAReader", "unable to read number of cells, the file may be invalid or corrupted." );
      }

      // read cells
      cellConnectivityArray.reserve( NumberOfCells );
      cellOffsetsArray.reserve( NumberOfCells );
      for( std::size_t cellIndex = 0; cellIndex < NumberOfCells; cellIndex++ ) {
         // read number of faces of a cell
         const auto numberOfCellFaces = readValue< std::size_t >( inputFile );
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "FPMAReader", "unable to read enough cells, the file may be invalid or corrupted." );
         }

         // read faces of a cell
         for( std::size_t i = 0; i < numberOfCellFaces; i++ ) {
            const auto faceIndex = readValue< std::uint32_t >( inputFile );
            if( ! iss ) {
               reset();
               throw MeshReaderError( "FPMAReader",
                                      "unable to read " + std::to_string( i ) + "th component of the cell number "
                                         + std::to_string( cellIndex ) + "." );
            }
            cellConnectivityArray.emplace_back( faceIndex );
         }

         cellOffsetsArray.emplace_back( cellConnectivityArray.size() );
      }

      // set the arrays to the base class
      this->pointsArray = std::move( pointsArray );
      this->cellConnectivityArray = std::move( cellConnectivityArray );
      this->cellOffsetsArray = std::move( cellOffsetsArray );
      this->faceConnectivityArray = std::move( faceConnectivityArray );
      this->faceOffsetsArray = std::move( faceOffsetsArray );

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
   }

private:
   template< typename T >
   T
   readValue( std::ifstream& ifs )
   {
      skipComments( ifs );
      T val;
      ifs >> val;
      return val;
   }

   static void
   skipComments( std::ifstream& ifs )
   {
      ifs >> std::ws;
      int c = ifs.peek();
      while( c == '#' && c != EOF ) {
         ifs.ignore( std::numeric_limits< std::streamsize >::max(), '\n' );  // skip to the next line
         ifs >> std::ws;
         c = ifs.peek();
      }
   }
};

}  // namespace Readers
}  // namespace Meshes
}  // namespace noa::TNL
