// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <map>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/MeshReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Endianness.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/EntityShapeGroupChecker.h>

namespace noa::TNL {
namespace Meshes {
namespace Readers {

class VTKReader : public MeshReader
{
public:
   VTKReader() = default;

   VTKReader( const std::string& fileName ) : MeshReader( fileName ) {}

   void
   detectMesh() override
   {
      reset();

      std::ifstream inputFile( fileName );
      if( ! inputFile )
         throw MeshReaderError( "VTKReader", "failed to open the file '" + fileName + "'" );

      parseHeader( inputFile );

      if( dataset != "UNSTRUCTURED_GRID" )
         throw MeshReaderError( "VTKReader", "the dataset '" + dataset + "' is not supported" );

      // parse the file, find the starting positions of all relevant sections
      findSections( inputFile );

      std::string line;
      std::string aux;
      std::istringstream iss;

      // parse points section
      if( sectionPositions.count( "POINTS" ) == 0 )
         throw MeshReaderError( "VTKReader", "unable to find the POINTS section, the file may be invalid or corrupted" );
      inputFile.seekg( sectionPositions[ "POINTS" ] );
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      iss >> NumberOfPoints;
      iss >> pointsType;
      if( pointsType != "float" && pointsType != "double" )
         throw MeshReaderError( "VTKReader", "unsupported data type for POINTS: " + pointsType );

      // only std::uint8_t makes sense for entity types
      typesType = "std::uint8_t";

      // arrays holding the data from the VTK file
      std::vector< double > pointsArray;
      std::vector< std::int64_t > cellConnectivityArray;
      std::vector< std::int64_t > cellOffsetsArray;
      std::vector< std::uint8_t > typesArray;

      // read points
      spaceDimension = 0;
      for( std::size_t pointIndex = 0; pointIndex < NumberOfPoints; pointIndex++ ) {
         if( ! inputFile )
            throw MeshReaderError( "VTKReader", "unable to read enough vertices, the file may be invalid or corrupted" );

         // read the coordinates and compute the space dimension
         for( int i = 0; i < 3; i++ ) {
            double aux = 0;
            if( pointsType == "float" )
               aux = readValue< float >( dataFormat, inputFile );
            else
               aux = readValue< double >( dataFormat, inputFile );
            if( ! inputFile )
               throw MeshReaderError( "VTKReader",
                                      "unable to read " + std::to_string( i ) + "th component of the vertex number "
                                         + std::to_string( pointIndex ) );
            if( aux != 0.0 )
               spaceDimension = std::max( spaceDimension, i + 1 );
            pointsArray.push_back( aux );
         }
      }

      // skip to the CELL_TYPES section
      if( sectionPositions.count( "CELL_TYPES" ) == 0 )
         throw MeshReaderError( "VTKReader", "unable to find the CELL_TYPES section, the file may be invalid or corrupted" );
      inputFile.seekg( sectionPositions[ "CELL_TYPES" ] );
      getline( inputFile, line );
      iss.clear();
      iss.str( line );
      iss >> aux;
      std::size_t NumberOfEntities = 0;
      iss >> NumberOfEntities;

      // read entity types
      for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
         if( ! inputFile )
            throw MeshReaderError( "VTKReader", "unable to read enough cell types, the file may be invalid or corrupted" );
         // cell types are stored with great redundancy as int32 in the VTK file
         const std::uint8_t typeId = readValue< std::int32_t >( dataFormat, inputFile );
         typesArray.push_back( typeId );
      }

      // count entities for each dimension
      std::size_t entitiesCounts[ 4 ] = { 0, 0, 0, 0 };
      for( auto c : typesArray ) {
         const int dimension = getEntityDimension( (VTK::EntityShape) c );
         ++entitiesCounts[ dimension ];
      }

      // set meshDimension
      meshDimension = 3;
      if( entitiesCounts[ 3 ] == 0 ) {
         meshDimension--;
         if( entitiesCounts[ 2 ] == 0 ) {
            meshDimension--;
            if( entitiesCounts[ 1 ] == 0 )
               meshDimension--;
         }
      }

      if( meshDimension == 0 )
         throw MeshReaderError( "VTKReader", "Mesh dimension cannot be 0. Are there any entities at all?" );

      // filter out cell shapes
      std::vector< std::uint8_t > cellTypes;
      for( auto c : typesArray ) {
         const int dimension = getEntityDimension( (VTK::EntityShape) c );
         if( dimension == meshDimension )
            cellTypes.push_back( c );
      }

      // set number of cells
      NumberOfCells = cellTypes.size();
      if( NumberOfCells == 0 || NumberOfCells != entitiesCounts[ meshDimension ] ) {
         const std::string msg = "invalid number of cells (" + std::to_string( NumberOfCells )
                               + "). Counts of entities for each dimension (0,1,2,3) are: "
                               + std::to_string( entitiesCounts[ 0 ] ) + ", " + std::to_string( entitiesCounts[ 1 ] ) + ", "
                               + std::to_string( entitiesCounts[ 2 ] ) + ", " + std::to_string( entitiesCounts[ 3 ] );
         throw MeshReaderError( "VTKReader", msg );
      }

      // validate cell types
      using PolygonShapeGroupChecker = VTK::EntityShapeGroupChecker< VTK::EntityShape::Polygon >;
      using PolyhedronShapeGroupChecker = VTK::EntityShapeGroupChecker< VTK::EntityShape::Polyhedron >;
      cellShape = (VTK::EntityShape) cellTypes[ 0 ];

      for( auto c : cellTypes ) {
         auto entityShape = (VTK::EntityShape) c;
         if( cellShape != entityShape ) {
            // if the input mesh includes mixed shapes, use more general cellShape (polygon for 2D, polyhedron for 3D)
            if( PolygonShapeGroupChecker::bothBelong( cellShape, entityShape ) )
               cellShape = PolygonShapeGroupChecker::GeneralShape;
            else if( PolyhedronShapeGroupChecker::bothBelong( cellShape, entityShape ) )
               cellShape = PolyhedronShapeGroupChecker::GeneralShape;
            else {
               const std::string msg = "Unsupported unstructured meshes with mixed entities: there are cells with type "
                                     + VTK::getShapeName( cellShape ) + " and " + VTK::getShapeName( entityShape ) + ".";
               reset();
               throw MeshReaderError( "VTKReader", msg );
            }
         }
      }

      // find to the CELLS section
      if( sectionPositions.count( "CELLS" ) == 0 )
         throw MeshReaderError( "VTKReader", "unable to find the CELLS section, the file may be invalid or corrupted" );
      inputFile.seekg( sectionPositions[ "CELLS" ] );
      getline( inputFile, line );

      if( formatVersion == "2.0" ) {
         // read entities
         for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
            if( ! inputFile )
               throw MeshReaderError( "VTKReader",
                                      "unable to read enough cells, the file may be invalid or corrupted"
                                      " (entityIndex = "
                                         + std::to_string( entityIndex ) + ")" );

            const VTK::EntityShape entityShape = (VTK::EntityShape) typesArray[ entityIndex ];

            if( entityShape == VTK::EntityShape::Polyhedron )
               throw MeshReaderError( "VTKReader",
                                      "Reading polyhedrons from a DataFile version 2.0 is not supported. "
                                      "Convert the file to version 5.1 (e.g. using Paraview) and try again." );

            if( entityShape == cellShape || PolygonShapeGroupChecker::bothBelong( cellShape, entityShape ) ) {
               // read number of subvertices
               const std::int32_t subvertices = readValue< std::int32_t >( dataFormat, inputFile );
               for( int v = 0; v < subvertices; v++ ) {
                  // legacy VTK files do not support 64-bit integers, even in the BINARY format
                  const std::int32_t vid = readValue< std::int32_t >( dataFormat, inputFile );
                  if( ! inputFile )
                     throw MeshReaderError( "VTKReader",
                                            "unable to read enough cells, the file may be invalid or corrupted"
                                            " (entityIndex = "
                                               + std::to_string( entityIndex ) + ", subvertex = " + std::to_string( v ) + ")" );
                  cellConnectivityArray.push_back( vid );
               }
               cellOffsetsArray.push_back( cellConnectivityArray.size() );
            }
            else {
               // skip the entity
               const std::int32_t subvertices = readValue< std::int32_t >( dataFormat, inputFile );
               for( int v = 0; v < subvertices; v++ )
                  skipValue( dataFormat, inputFile, "int" );
            }
         }
      }
      else if( formatVersion == "5.1" ) {
         // parse the rest of the line: CELLS <offsets_count> <connectivity_count>
         std::int64_t offsets_count = 0;
         std::int64_t connectivity_count = 0;
         iss.clear();
         iss.str( line );
         iss >> aux >> offsets_count >> connectivity_count;
         if( offsets_count < 1 )
            throw MeshReaderError( "VTKReader", "invalid offsets count: " + std::to_string( offsets_count ) );

         // find to the OFFSETS section
         if( ! sectionPositions.count( "OFFSETS" ) )
            throw MeshReaderError( "VTKReader", "unable to find the OFFSETS section, the file may be invalid or corrupted" );
         inputFile.seekg( sectionPositions[ "OFFSETS" ] );

         // read all offsets into an auxiliary array
         std::vector< std::int64_t > allOffsetsArray;
         getline( inputFile, line );
         iss.clear();
         iss.str( line );
         std::string aux, datatype;
         iss >> aux >> datatype;
         for( std::int64_t entityIndex = 0; entityIndex < offsets_count; entityIndex++ ) {
            std::int64_t value;
            if( datatype == "vtktypeint32" )
               value = readValue< std::int32_t >( dataFormat, inputFile );
            else if( datatype == "vtktypeint64" )
               value = readValue< std::int64_t >( dataFormat, inputFile );
            else
               throw MeshReaderError( "VTKReader", "found data type which is not implemented in the reader: " + datatype );
            if( ! inputFile )
               throw MeshReaderError( "VTKReader",
                                      "unable to read enough offsets, the file may be invalid or corrupted"
                                      " (entityIndex = "
                                         + std::to_string( entityIndex ) + ")" );
            allOffsetsArray.push_back( value );
         }

         // find to the CONNECTIVITY section
         if( ! sectionPositions.count( "CONNECTIVITY" ) )
            throw MeshReaderError( "VTKReader",
                                   "unable to find the CONNECTIVITY section, the file may be invalid or corrupted" );
         inputFile.seekg( sectionPositions[ "CONNECTIVITY" ] );

         // get datatype
         getline( inputFile, line );
         iss.clear();
         iss.str( line );
         iss >> aux >> datatype;

         // arrays for polyhedral mesh
         std::vector< std::int64_t > faceConnectivityArray;
         std::vector< std::int64_t > faceOffsetsArray;
         std::int64_t faceIndex = 0;

         // read connectivity
         for( std::size_t entityIndex = 0; entityIndex < NumberOfEntities; entityIndex++ ) {
            if( ! inputFile )
               throw MeshReaderError( "VTKReader",
                                      "unable to read enough cells, the file may be invalid or corrupted"
                                      " (entityIndex = "
                                         + std::to_string( entityIndex ) + ")" );

            const VTK::EntityShape entityShape = (VTK::EntityShape) typesArray[ entityIndex ];
            const std::int64_t offsetBegin = allOffsetsArray[ entityIndex ];
            const std::int64_t offsetEnd = allOffsetsArray[ entityIndex + 1 ];

            // TODO: Polyhedrons will require to create polygon subentity seeds from given entityShapes
            //       and add their entries to faceConnectivityArray and faceOffsetsArray.
            //       CellConnectivityArray and cellOffsetsArray will contain indices addressing created polygon subentities.
            if( cellShape == VTK::EntityShape::Polyhedron && entityShape != cellShape
                && PolyhedronShapeGroupChecker::bothBelong( cellShape, entityShape ) )
               throw MeshReaderError( "VTKReader", "Converting a mixed mesh to polyhedral mesh is not implemented yet." );

            if( entityShape == cellShape && cellShape == VTK::EntityShape::Polyhedron ) {
               // read connectivity for current cell into extra array
               std::vector< std::int64_t > cell_connectivity;
               for( int v = 0; v < offsetEnd - offsetBegin; v++ ) {
                  std::int64_t value;
                  if( datatype == "vtktypeint32" )
                     value = readValue< std::int32_t >( dataFormat, inputFile );
                  else if( datatype == "vtktypeint64" )
                     value = readValue< std::int64_t >( dataFormat, inputFile );
                  else
                     throw MeshReaderError( "VTKReader",
                                            "found data type which is not implemented in the reader: " + datatype );
                  if( ! inputFile )
                     throw MeshReaderError( "VTKReader",
                                            "unable to read enough cells, the file may be invalid or corrupted"
                                            " (entityIndex = "
                                               + std::to_string( entityIndex ) + ", subvertex = " + std::to_string( v ) + ")" );
                  cell_connectivity.push_back( value );
               }
               // connectivity[offsetBegin : offsetEnd] describes the faces of
               // the cell in the following format (similar to VTU's faces array)
               //    num_faces_cell_0,
               //      num_nodes_face_0, node_ind_0, node_ind_1, ..
               //      num_nodes_face_1, node_ind_0, node_ind_1, ..
               //      ...
               std::size_t i = 1;  // skip num_faces for the cell
               while( i < cell_connectivity.size() ) {
                  const std::int64_t num_nodes = cell_connectivity.at( i++ );
                  for( std::int64_t n = 0; n < num_nodes; n++ )
                     faceConnectivityArray.push_back( cell_connectivity.at( i++ ) );
                  faceOffsetsArray.push_back( faceConnectivityArray.size() );
                  cellConnectivityArray.push_back( faceIndex++ );
               }
               cellOffsetsArray.push_back( cellConnectivityArray.size() );
            }
            else if( entityShape == cellShape || PolygonShapeGroupChecker::bothBelong( cellShape, entityShape ) ) {
               for( int v = 0; v < offsetEnd - offsetBegin; v++ ) {
                  std::int64_t vid;
                  if( datatype == "vtktypeint32" )
                     vid = readValue< std::int32_t >( dataFormat, inputFile );
                  else if( datatype == "vtktypeint64" )
                     vid = readValue< std::int64_t >( dataFormat, inputFile );
                  else
                     throw MeshReaderError( "VTKReader",
                                            "found data type which is not implemented in the reader: " + datatype );
                  if( ! inputFile )
                     throw MeshReaderError( "VTKReader",
                                            "unable to read enough cells, the file may be invalid or corrupted"
                                            " (entityIndex = "
                                               + std::to_string( entityIndex ) + ", subvertex = " + std::to_string( v ) + ")" );
                  cellConnectivityArray.push_back( vid );
               }
               cellOffsetsArray.push_back( cellConnectivityArray.size() );
            }
            else {
               // skip the entity
               for( int v = 0; v < offsetEnd - offsetBegin; v++ )
                  skipValue( dataFormat, inputFile, datatype );
            }
         }

         // set the arrays to the base class
         if( faceIndex > 0 ) {
            this->NumberOfFaces = faceIndex;
            this->faceConnectivityArray = std::move( faceConnectivityArray );
            this->faceOffsetsArray = std::move( faceOffsetsArray );
         }
      }

      // set cell types
      std::swap( cellTypes, typesArray );

      // set the arrays to the base class
      this->pointsArray = std::move( pointsArray );
      this->cellConnectivityArray = std::move( cellConnectivityArray );
      this->cellOffsetsArray = std::move( cellOffsetsArray );
      this->typesArray = std::move( typesArray );

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
   }

   VariantVector
   readPointData( std::string arrayName ) override
   {
      return readPointOrCellData( "POINT_DATA", arrayName );
   }

   VariantVector
   readCellData( std::string arrayName ) override
   {
      return readPointOrCellData( "CELL_DATA", arrayName );
   }

   void
   reset() override
   {
      resetBase();
      dataFormat = VTK::FileFormat::ascii;
      dataset = "";
      sectionPositions.clear();
   }

protected:
   // output of parseHeader
   std::string formatVersion;
   VTK::FileFormat dataFormat = VTK::FileFormat::ascii;
   std::string dataset;

   // output of findSections
   std::map< std::string, std::ios::pos_type > sectionPositions;

   // mesh properties - needed for reading POINT_DATA and CELL_DATA
   std::int32_t points_count = 0;
   std::int32_t cells_count = 0;

   void
   parseHeader( std::istream& str )
   {
      std::string line;
      std::istringstream iss;

      // check header
      getline( str, line );
      static const std::string prefix = "# vtk DataFile Version ";
      formatVersion = line.substr( prefix.length() );
      if( line.substr( 0, prefix.length() ) != prefix )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header: unsupported VTK header '" + line + "'" );
      if( formatVersion != "2.0" && formatVersion != "5.1" )
         throw MeshReaderError( "VTKReader", "unsupported VTK DataFile Version: '" + formatVersion + "'" );

      // skip title
      if( ! str )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header" );
      getline( str, line );

      // parse data type
      if( ! str )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header" );
      std::string format;
      getline( str, format );
      if( format == "ASCII" )
         dataFormat = VTK::FileFormat::ascii;
      else if( format == "BINARY" )
         dataFormat = VTK::FileFormat::binary;
      else
         throw MeshReaderError( "VTKReader", "unknown data format: '" + format + "'" );

      // parse dataset
      if( ! str )
         throw MeshReaderError( "VTKReader", "failed to parse the VTK file header" );
      getline( str, line );
      iss.clear();
      iss.str( line );
      std::string tmp;
      iss >> tmp;
      if( tmp != "DATASET" )
         throw MeshReaderError( "VTKReader", "wrong dataset specification: '" + line + "'" );
      iss >> dataset;
   }

   void
   skip_meta( std::istream& str )
   {
      // skip possible metadata
      // https://vtk.org/doc/nightly/html/IOLegacyInformationFormat.html
      std::string line;
      while( true ) {
         getline( str, line );
         if( ! str )
            throw MeshReaderError( "VTKReader", "failed to parse a METADATA section: is it terminated by a blank line?" );
         // strip whitespace
         line.erase( std::remove_if( line.begin(), line.end(), isspace ), line.end() );
         if( line.empty() )
            break;
      }
   }

   void
   findSections( std::istream& str )
   {
      while( str ) {
         // drop all whitespace (empty lines etc) before saving a position and reading a line
         str >> std::ws;
         if( str.eof() )
            break;

         // read a line which should contain the following section header
         const std::ios::pos_type currentPosition = str.tellg();
         std::string line;
         getline( str, line );
         if( ! str )
            throw MeshReaderError( "VTKReader", "failed to parse sections of the VTK file" );

         // parse the section name
         std::istringstream iss( line );
         std::string name;
         iss >> name;

         if( name == "FIELD" ) {
            sectionPositions.insert( { "FIELD", currentPosition } );
            // parse the rest of the line: FIELD FieldData <count>
            std::string aux;
            int count = 0;
            iss >> aux >> count;
            // skip the FieldData arrays
            for( int i = 0; i < count; i++ ) {
               getline( str, line );
               iss.clear();
               iss.str( line );
               // <name> <components> <tuples> <datatype>
               std::int32_t components = 0;
               std::int32_t tuples = 0;
               std::string datatype;
               iss >> aux;
               if( aux == "METADATA" ) {
                  // skip metadata and read again
                  skip_meta( str );
                  i--;
                  continue;
               }
               iss >> components >> tuples >> datatype;
               if( ! iss )
                  throw MeshReaderError( "VTKReader", "failed to extract FieldData information from line '" + line + "'" );
               // skip the points coordinates
               for( std::int32_t j = 0; j < components * tuples; j++ )
                  skipValue( dataFormat, str, datatype );
               // skip end of line (or any whitespace)
               str >> std::ws;
            }
         }
         else if( name == "POINTS" ) {
            sectionPositions.insert( { "POINTS", currentPosition } );
            // parse the rest of the line: POINTS <points_count> <datatype>
            std::string datatype;
            iss >> points_count >> datatype;
            // skip the values
            for( std::int32_t j = 0; j < 3 * points_count; j++ )
               skipValue( dataFormat, str, datatype );
            // skip end of line (or any whitespace)
            str >> std::ws;
         }
         // METADATA is a thing since version 5.1 of the file format (or something else newer than 2.0)
         else if( name == "METADATA" ) {
            sectionPositions.insert( { "METADATA", currentPosition } );
            skip_meta( str );
         }
         else if( name == "CELLS" ) {
            sectionPositions.insert( { "CELLS", currentPosition } );
            if( formatVersion == "2.0" ) {
               // index type is not stored in legacy VTK DataFile version 2.0
               // (binary files don't support int64)
               connectivityType = offsetsType = "std::int32_t";
               // parse the rest of the line: CELLS <cells_count> <values_count>
               std::int32_t values_count = 0;
               iss >> cells_count >> values_count;
               // skip the values
               for( std::int32_t j = 0; j < values_count; j++ )
                  skipValue( dataFormat, str, "int" );
            }
            else if( formatVersion == "5.1" ) {
               // parse the rest of the line: CELLS <offsets_count> <connectivity_count>
               std::int32_t offsets_count = 0;
               std::int32_t connectivity_count = 0;
               iss >> offsets_count >> connectivity_count;
               if( offsets_count < 1 )
                  throw MeshReaderError( "VTKReader", "invalid offsets count: " + std::to_string( offsets_count ) );
               cells_count = offsets_count - 1;
               // drop all whitespace (empty lines etc) before saving a position and reading a line
               str >> std::ws;
               const std::ios::pos_type offsetsPosition = str.tellg();
               // skip offsets
               str >> std::ws;
               getline( str, line );
               iss.clear();
               iss.str( line );
               std::string aux, datatype;
               iss >> aux >> datatype;
               if( aux != "OFFSETS" )
                  throw MeshReaderError( "VTKReader", "expected OFFSETS section, found '" + aux + "'" );
               sectionPositions.insert( { "OFFSETS", offsetsPosition } );
               if( datatype == "vtktypeint32" )
                  offsetsType = "std::int32_t";
               else if( datatype == "vtktypeint64" )
                  offsetsType = "std::int64_t";
               else
                  throw MeshReaderError( "VTKReader", "unsupported datatype for OFFSETS: " + datatype );
               for( std::int32_t j = 0; j < offsets_count; j++ )
                  skipValue( dataFormat, str, datatype );
               // drop all whitespace (empty lines etc) before saving a position and reading a line
               str >> std::ws;
               const std::ios::pos_type connectivityPosition = str.tellg();
               // skip connectivity
               str >> std::ws;
               getline( str, line );
               iss.clear();
               iss.str( line );
               iss >> aux >> datatype;
               if( aux != "CONNECTIVITY" )
                  throw MeshReaderError( "VTKReader", "expected CONNECTIVITY section, found '" + aux + "'" );
               sectionPositions.insert( { "CONNECTIVITY", connectivityPosition } );
               if( datatype == "vtktypeint32" )
                  connectivityType = "std::int32_t";
               else if( datatype == "vtktypeint64" )
                  connectivityType = "std::int64_t";
               else
                  throw MeshReaderError( "VTKReader", "unsupported datatype for CONNECTIVITY: " + datatype );
               for( std::int32_t j = 0; j < connectivity_count; j++ )
                  skipValue( dataFormat, str, datatype );
            }
            // skip end of line (or any whitespace)
            str >> std::ws;
         }
         else if( name == "CELL_TYPES" ) {
            sectionPositions.insert( { "CELL_TYPES", currentPosition } );
            // parse the rest of the line: CELL_TYPES <count>
            std::int32_t count = 0;
            iss >> count;
            // skip the values
            for( std::int32_t j = 0; j < count; j++ )
               // cell types are stored with great redundancy as int32 in the VTK file
               skipValue( dataFormat, str, "int" );
            // skip end of line (or any whitespace)
            str >> std::ws;
         }
         else if( name == "CELL_DATA" || name == "POINT_DATA" ) {
            if( cells_count == 0 || points_count == 0 )
               throw MeshReaderError( "VTKReader",
                                      "encountered a " + name
                                         + " section, but the mesh topology was not parsed yet "
                                           "(cells count = "
                                         + std::to_string( cells_count ) + ", points count = " + std::to_string( points_count )
                                         + ")" );

            while( str ) {
               // drop all whitespace (empty lines etc) before saving a position and reading a line
               str >> std::ws;
               if( str.eof() )
                  break;

               // read a line which should contain the following array metadata
               const std::ios::pos_type currentPosition = str.tellg();
               std::string line;
               getline( str, line );
               if( ! str )
                  throw MeshReaderError( "VTKReader", "failed to parse sections of the VTK file" );

               // parse the array type
               std::istringstream iss( line );
               std::string type;
               iss >> type;

               // handle switching between CELL_DATA and POINT_DATA
               if( ( name == "CELL_DATA" && type == "POINT_DATA" ) || ( name == "POINT_DATA" && type == "CELL_DATA" ) ) {
                  name = type;
                  continue;
               }

               const std::int32_t elements = ( name == "CELL_DATA" ) ? cells_count : points_count;

               // scalars: 1 value per cell/point
               // vectors: 3 values per cell/point
               // fields: arbitrary number of values per cell/point
               int values_per_element = 1;

               // additional metadata
               std::string array_name;
               std::string datatype;

               if( type == "SCALARS" ) {
                  // parse the rest of the line: SCALARS <array_name> <datatype>
                  iss >> array_name >> datatype;
                  sectionPositions.insert( { name + "::" + array_name, currentPosition } );
                  // skip the LOOKUP_TABLE line
                  getline( str, line );
               }
               else if( type == "VECTORS" ) {
                  values_per_element = 3;
                  // parse the rest of the line: VECTORS <array_name> <datatype>
                  iss >> array_name >> datatype;
                  sectionPositions.insert( { name + "::" + array_name, currentPosition } );
               }
               else if( type == "TENSORS" ) {
                  values_per_element = 9;
                  // parse the rest of the line: TENSORS <array_name> <datatype>
                  iss >> array_name >> datatype;
                  sectionPositions.insert( { name + "::" + array_name, currentPosition } );
               }
               else if( type == "FIELD" ) {
                  // parse the rest of the line: FIELD FieldData <count>
                  std::string aux;
                  int count = 0;
                  iss >> aux >> count;
                  // skip the FieldData arrays
                  for( int i = 0; i < count; i++ ) {
                     // drop all whitespace (empty lines etc) before saving a position and reading a line
                     str >> std::ws;
                     const std::ios::pos_type currentPosition = str.tellg();
                     getline( str, line );
                     iss.clear();
                     iss.str( line );
                     // <array_name> <components> <tuples> <datatype>
                     std::int32_t components = 0;
                     std::int32_t tuples = 0;
                     std::string datatype;
                     iss >> array_name;
                     if( array_name == "METADATA" ) {
                        // skip metadata and read again
                        skip_meta( str );
                        i--;
                        continue;
                     }
                     iss >> components >> tuples >> datatype;
                     if( ! iss )
                        throw MeshReaderError( "VTKReader",
                                               "failed to extract FieldData information from line '" + line + "'" );
                     sectionPositions.insert( { name + "::" + array_name, currentPosition } );
                     // skip the points coordinates
                     for( std::int32_t j = 0; j < components * tuples; j++ )
                        skipValue( dataFormat, str, datatype );
                     // skip end of line (or any whitespace)
                     str >> std::ws;
                  }
                  continue;
               }
               else if( type == "METADATA" ) {
                  skip_meta( str );
                  continue;
               }
               else {
                  std::cerr << "VTKReader: encountered an unsupported CELL_DATA array type: " << type
                            << ". Ignoring the rest of the file." << std::endl;
                  return;
               }

               // skip the values
               for( std::int32_t j = 0; j < elements * values_per_element; j++ )
                  skipValue( dataFormat, str, datatype );
               // skip end of line (or any whitespace)
               str >> std::ws;
            }
         }
         else
            throw MeshReaderError( "VTKReader",
                                   "parsing error: unexpected section start at byte " + std::to_string( currentPosition )
                                      + " (section name is '" + name + "')" );
      }

      // clear errors bits on the input stream
      str.clear();
   }

   VariantVector
   readPointOrCellData( std::string sectionName, const std::string& arrayName )
   {
      std::ifstream inputFile( fileName );
      if( ! inputFile )
         throw MeshReaderError( "VTKReader", "failed to open the file '" + fileName + "'" );

      std::int32_t elements = ( sectionName == "CELL_DATA" ) ? cells_count : points_count;
      int values_per_element = 1;

      sectionName += "::" + arrayName;
      if( sectionPositions.count( sectionName ) == 0 )
         throw MeshReaderError( "VTKReader", "array " + arrayName + " was not found in the CELL_DATA section" );
      inputFile.seekg( sectionPositions[ sectionName ] );

      // type: SCALARS, VECTORS, etc.
      // datatype: int, float, double
      std::string type;
      std::string datatype;

      // parse the metadata line
      std::string line;
      getline( inputFile, line );
      std::istringstream iss( line );
      iss >> type;

      // if the line starts with the array name, it must be a FIELD
      if( type == arrayName ) {
         // parse <array_name> <components> <tuples> <datatype>
         iss >> values_per_element >> elements >> datatype;
      }
      else {
         // parse the rest of the line: <type> <array_name> <datatype>
         std::string array_name;
         iss >> array_name >> datatype;
         if( type == "SCALARS" ) {
            values_per_element = 1;
            // skip the LOOKUP_TABLE line
            getline( inputFile, line );
         }
         else if( type == "VECTORS" )
            values_per_element = 3;
         else if( type == "TENSORS" )
            values_per_element = 9;
         else
            throw MeshReaderError( "VTKReader", "requested array type " + type + " is not implemented in the reader" );
      }

      if( datatype == "int" )
         return readDataArray< std::int32_t >( inputFile, elements * values_per_element );
      else if( datatype == "float" )
         return readDataArray< float >( inputFile, elements * values_per_element );
      else if( datatype == "double" )
         return readDataArray< double >( inputFile, elements * values_per_element );
      else
         throw MeshReaderError( "VTKReader", "found data type which is not implemented in the reader: " + datatype );
   }

   template< typename T >
   std::vector< T >
   readDataArray( std::istream& str, std::int32_t values )
   {
      std::vector< T > vector( values );
      for( std::int32_t i = 0; i < values; i++ )
         vector[ i ] = readValue< T >( dataFormat, str );
      return vector;
   }

   static void
   skipValue( VTK::FileFormat format, std::istream& str, const std::string& datatype )
   {
      if( datatype == "int" )  // implicit in vtk DataFile Version 2.0
         readValue< std::int32_t >( format, str );
      else if( datatype == "vtktypeint32" )  // vtk DataFile Version 5.1
         readValue< std::int32_t >( format, str );
      else if( datatype == "vtktypeint64" )  // vtk DataFile Version 5.1
         readValue< std::int64_t >( format, str );
      else if( datatype == "float" )
         readValue< float >( format, str );
      else if( datatype == "double" )
         readValue< double >( format, str );
      else
         throw MeshReaderError( "VTKReader", "found data type which is not implemented in the reader: " + datatype );
   }

   template< typename T >
   static T
   readValue( VTK::FileFormat format, std::istream& str )
   {
      T value;
      if( format == VTK::FileFormat::binary ) {
         str.read( reinterpret_cast< char* >( &value ), sizeof( T ) );
         // forceBigEndian = swapIfLittleEndian, i.e. here it forces a big-endian
         // value to the correct system endianness
         value = forceBigEndian( value );
      }
      else {
         str >> value;
      }
      return value;
   }
};

}  // namespace Readers
}  // namespace Meshes
}  // namespace noa::TNL
