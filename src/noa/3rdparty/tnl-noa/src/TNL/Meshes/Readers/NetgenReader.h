// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Tomas Oberhuber
 * Vitezslav Zabka
 * Jakub Klinkovsky
 */

#pragma once

#include <fstream>
#include <sstream>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/MeshReader.h>

namespace noa::TNL {
namespace Meshes {
namespace Readers {

class NetgenReader : public MeshReader
{
public:
   NetgenReader() = default;

   NetgenReader( const std::string& fileName ) : MeshReader( fileName ) {}

   void
   detectMesh() override
   {
      reset();

      std::ifstream inputFile( fileName );
      if( ! inputFile )
         throw MeshReaderError( "NetgenReader", "failed to open the file '" + fileName + "'." );

      std::string line;
      std::istringstream iss;

      // skip whitespace
      inputFile >> std::ws;
      if( ! inputFile )
         throw MeshReaderError( "NetgenReader", "unexpected error when reading the file '" + fileName + "'." );

      // read number of points
      getline( inputFile, line );
      iss.str( line );
      iss >> NumberOfPoints;

      // real type is not stored in Netgen files
      pointsType = "double";
      // global index type is not stored in Netgen files
      connectivityType = offsetsType = "std::int32_t";
      // only std::uint8_t makes sense for entity types
      typesType = "std::uint8_t";

      // arrays holding the data from the file
      std::vector< double > pointsArray;
      std::vector< std::int32_t > connectivityArray;
      std::vector< std::int32_t > offsetsArray;
      std::vector< std::uint8_t > typesArray;

      // read points
      spaceDimension = 0;
      for( std::size_t pointIndex = 0; pointIndex < NumberOfPoints; pointIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "NetgenReader", "unable to read enough vertices, the file may be invalid or corrupted." );
         }
         getline( inputFile, line );

         // read the coordinates and compute the space dimension
         iss.clear();
         iss.str( line );
         for( int i = 0; i < 3; i++ ) {
            double aux;
            iss >> aux;
            if( ! iss ) {
               // the intermediate mesh representation uses the VTK convention - all points must have 3 coordinates
               aux = 0;
            }
            if( aux != 0.0 )
               spaceDimension = std::max( spaceDimension, i + 1 );
            pointsArray.push_back( aux );
         }
      }

      // netgen supports only triangular and tetrahedral meshes
      meshDimension = spaceDimension;
      if( meshDimension == 1 )
         cellShape = VTK::EntityShape::Line;
      else if( meshDimension == 2 )
         cellShape = VTK::EntityShape::Triangle;
      else if( meshDimension == 3 )
         cellShape = VTK::EntityShape::Tetra;
      else
         throw MeshReaderError( "NetgenReader", "unsupported mesh dimension: " + std::to_string( meshDimension ) );

      // skip whitespace
      inputFile >> std::ws;
      if( ! inputFile )
         throw MeshReaderError( "NetgenReader", "unexpected error when reading the file '" + fileName + "'." );

      // read number of cells
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> NumberOfCells;

      // read cells
      for( std::size_t cellIndex = 0; cellIndex < NumberOfCells; cellIndex++ ) {
         if( ! inputFile ) {
            reset();
            throw MeshReaderError( "NetgenReader",
                                   "unable to read enough cells, the file may be invalid or corrupted."
                                   " (cellIndex = "
                                      + std::to_string( cellIndex ) + ")" );
         }
         getline( inputFile, line );

         iss.clear();
         iss.str( line );
         // skip subdomain number
         int subdomain;
         iss >> subdomain;
         for( int v = 0; v <= meshDimension; v++ ) {
            std::size_t vid;
            iss >> vid;
            if( ! iss ) {
               reset();
               throw MeshReaderError( "NetgenReader",
                                      "unable to read enough cells, the file may be invalid or corrupted."
                                      " (cellIndex = "
                                         + std::to_string( cellIndex ) + ", subvertex = " + std::to_string( v ) + ")" );
            }
            // convert point index from 1-based to 0-based
            connectivityArray.push_back( vid - 1 );
         }
         offsetsArray.push_back( connectivityArray.size() );
      }

      // set cell types
      typesArray.resize( NumberOfCells, (std::uint8_t) cellShape );

      // set the arrays to the base class
      this->pointsArray = std::move( pointsArray );
      this->cellConnectivityArray = std::move( connectivityArray );
      this->cellOffsetsArray = std::move( offsetsArray );
      this->typesArray = std::move( typesArray );

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
   }
};

}  // namespace Readers
}  // namespace Meshes
}  // namespace noa::TNL
