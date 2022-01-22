// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Meshes/Readers/XMLVTK.h>
#include <noa/3rdparty/TNL/Meshes/EntityShapeGroupChecker.h>

namespace noaTNL {
namespace Meshes {
namespace Readers {

class VTUReader
: public XMLVTK
{
#ifdef HAVE_TINYXML2
   void readUnstructuredGrid()
   {
      using namespace noaTinyxml2;
      const XMLElement* piece = getChildSafe( datasetElement, "Piece" );
      if( piece->NextSiblingElement( "Piece" ) )
         // ambiguity - throw error, we don't know which piece to parse (or all of them?)
         throw MeshReaderError( "VTUReader", "the serial UnstructuredGrid file contains more than one <Piece> element" );
      NumberOfPoints = getAttributeInteger( piece, "NumberOfPoints" );
      NumberOfCells = getAttributeInteger( piece, "NumberOfCells" );

      // verify points
      const XMLElement* points = getChildSafe( piece, "Points" );
      const XMLElement* pointsData = verifyHasOnlyOneChild( points, "DataArray" );
      verifyDataArray( pointsData );
      const std::string pointsDataName = getAttributeString( pointsData, "Name" );
      if( pointsDataName != "Points" )
         throw MeshReaderError( "VTUReader", "the <Points> tag does not contain a <DataArray> with Name=\"Points\" attribute" );

      // verify cells
      const XMLElement* cells = getChildSafe( piece, "Cells" );
      const XMLElement* connectivity = getDataArrayByName( cells, "connectivity" );
      const XMLElement* offsets = getDataArrayByName( cells, "offsets" );
      const XMLElement* types = getDataArrayByName( cells, "types" );

      // read the points, connectivity, offsets and types into intermediate arrays
      pointsArray = readDataArray( pointsData, "Points" );
      pointsType = VTKDataTypes.at( getAttributeString( pointsData, "type" ) );
      cellConnectivityArray = readDataArray( connectivity, "connectivity" );
      connectivityType = VTKDataTypes.at( getAttributeString( connectivity, "type" ) );
      cellOffsetsArray = readDataArray( offsets, "offsets" );
      offsetsType = VTKDataTypes.at( getAttributeString( offsets, "type" ) );
      typesArray = readDataArray( types, "types" );
      typesType = VTKDataTypes.at( getAttributeString( types, "type" ) );

      // connectivity and offsets must have the same type
      if( connectivityType != offsetsType )
         throw MeshReaderError( "VTUReader", "the \"connectivity\" and \"offsets\" array do not have the same type ("
                            + connectivityType + " vs " + offsetsType + ")" );
      // cell types can be only uint8_t
      if( typesType != "std::uint8_t" )
         throw MeshReaderError( "VTUReader", "unsupported data type for the Name=\"types\" array" );

      using std::visit;
      // validate points
      visit( [this](auto&& array) {
               // check array size
               if( array.size() != 3 * NumberOfPoints )
                  throw MeshReaderError( "VTUReader", "invalid size of the Points data array (" + std::to_string(array.size())
                                                      + " vs " + std::to_string(NumberOfPoints) + ")" );
               // set spaceDimension
               spaceDimension = 1;
               std::size_t i = 0;
               for( auto c : array ) {
                  if( c != 0 ) {
                     int dim = i % 3 + 1;
                     spaceDimension = std::max( spaceDimension, dim );
                  }
                  ++i;
               }
            },
            pointsArray
         );
      // validate types
      visit( [this](auto&& array) {
               // check array size
               if( array.size() != NumberOfCells )
                  throw MeshReaderError( "VTUReader", "size of the types data array does not match the NumberOfCells attribute" );
               // check empty mesh
               if( array.size() == 0 )
                  return;
               cellShape = (VTK::EntityShape) array[0];
               meshDimension = getEntityDimension( cellShape );
               using PolygonShapeGroupChecker = VTK::EntityShapeGroupChecker< VTK::EntityShape::Polygon >;
               using PolyhedronShapeGroupChecker = VTK::EntityShapeGroupChecker< VTK::EntityShape::Polyhedron >;

               // TODO: check only entities of the same dimension (edges, faces and cells separately)
               for( auto c : array ) {
                  VTK::EntityShape entityShape = (VTK::EntityShape) c;
                  if( entityShape != cellShape ) {
                     if( PolygonShapeGroupChecker::bothBelong( cellShape, entityShape ) )
                        cellShape = PolygonShapeGroupChecker::GeneralShape;
                     else if( PolyhedronShapeGroupChecker::bothBelong( cellShape, entityShape ) )
                        cellShape = PolyhedronShapeGroupChecker::GeneralShape;
                     else {
                        const std::string msg = "Unsupported unstructured meshes with mixed entities: there are cells with type "
                                              + VTK::getShapeName(cellShape) + " and " + VTK::getShapeName(entityShape) + ".";
                        throw MeshReaderError( "VTUReader", msg );
                     }
                  }
               }
            },
            typesArray
         );
      // validate offsets
      std::size_t max_offset = 0;
      visit( [this, &max_offset](auto&& array) mutable {
               if( array.size() != NumberOfCells )
                  throw MeshReaderError( "VTUReader", "size of the offsets data array does not match the NumberOfCells attribute" );
               for( auto c : array ) {
                  if( c <= (decltype(c)) max_offset )
                     throw MeshReaderError( "VTUReader", "the offsets array is not monotonically increasing" );
                  max_offset = c;
               }
            },
            cellOffsetsArray
         );
      // validate connectivity
      visit( [this, max_offset](auto&& array) {
               if( array.size() != max_offset )
                  throw MeshReaderError( "VTUReader", "size of the connectivity data array does not match the offsets array" );
               for( auto c : array ) {
                  if( c < 0 || (std::size_t) c >= NumberOfPoints )
                     throw MeshReaderError( "VTUReader", "connectivity index " + std::to_string(c) + " is out of range" );
               }
            },
            cellConnectivityArray
         );

      if( cellShape == VTK::EntityShape::Polyhedron ) {
         // NOTE: the data format for polyhedral meshes in VTK files is not documented well (almost not at all).
         // Here are some references:
         // - https://itk.org/Wiki/VTK/Polyhedron_Support
         // - https://vtk.org/doc/nightly/html/classvtkPolyhedron.html
         // - https://github.com/nschloe/meshio/pull/916
         // - https://github.com/nschloe/meshio/blob/b358a88b7c1158d5ee2b2c873f67ba1cb0647686/src/meshio/vtu/_vtu.py#L33-L102

         const XMLElement* faces = getDataArrayByName( cells, "faces" );
         const XMLElement* faceOffsets = getDataArrayByName( cells, "faceoffsets" );
         const VariantVector vtk_facesArray = readDataArray( faces, "faces" );
         const VariantVector vtk_faceOffsetsArray = readDataArray( faceOffsets, "faceoffsets" );
         const std::string facesType = VTKDataTypes.at( getAttributeString( faces, "type" ) );
         const std::string faceOffsetsType = VTKDataTypes.at( getAttributeString( faceOffsets, "type" ) );
         if( facesType != faceOffsetsType )
            throw MeshReaderError( "VTUReader", "type of the faces array does not match the type of the faceoffsets array" );
         if( faceOffsetsType != offsetsType )
            throw MeshReaderError( "VTUReader", "type of the faceoffsets array does not match the type of the offsets array" );

         // validate face offsets
         std::size_t max_offset = 0;
         visit( [this, &max_offset](auto&& array) mutable {
                  if( array.size() != NumberOfCells )
                     throw MeshReaderError( "VTUReader", "size of the faceoffsets data array does not match the NumberOfCells attribute" );
                  for( auto c : array ) {
                     // NOTE: VTK stores -1 for cells that are not a polyhedron. We would need to populate
                     if( c < 0 )
                        continue;
                     if( c <= (decltype(c)) max_offset )
                        throw MeshReaderError( "VTUReader", "the faceoffsets array is not monotonically increasing" );
                     max_offset = c;
                  }
               },
               vtk_faceOffsetsArray
            );
         // validate faces
         visit( [this, max_offset, &vtk_faceOffsetsArray](auto&& vtk_faces) {
                  if( vtk_faces.size() != max_offset )
                     throw MeshReaderError( "VTUReader", "size of the faces data array does not match the faceoffsets array" );
                  // let's just assume that the connectivity and offsets arrays have the same type...
                  using std::get;
                  const auto& vtk_faceOffsets = get< std::decay_t<decltype(vtk_faces)> >( vtk_faceOffsetsArray );

                  // We need to translate the VTK faces and faceoffsets arrays
                  // into the format suitable for MeshReader (which uses the
                  // format from FPMA). The data format for the faces array is:
                  //    num_faces_cell_0,
                  //      num_nodes_face_0, node_ind_0, node_ind_1, ..
                  //      num_nodes_face_1, node_ind_0, node_ind_1, ..
                  //      ...
                  //    num_faces_cell_1,
                  //      ...
                  // See https://vtk.org/Wiki/VTK/Polyhedron_Support for more.
                  std::decay_t<decltype(vtk_faces)> cellOffsets, cellConnectivity, faceOffsets, faceConnectivity;
                  std::make_signed_t< std::size_t > cell_off_begin = 0;
                  std::size_t faceIndex = 0;
                  for( std::size_t cell = 0; cell < vtk_faceOffsets.size(); cell++ ) {
                     const std::make_signed_t< std::size_t > cell_off_end = vtk_faceOffsets[ cell ];
                     // TODO: VTK stores -1 for cells that are not a polyhedron. We would need to populate
                     // faceOffsetsArray and faceConnectivityArray with values based on the cell topology.
                     if( cell_off_end < 0 )
                        throw MeshReaderError( "VTUReader", "found invalid offset in the faceoffsets array: " + std::to_string(cell_off_end) );
                     if( static_cast< std::size_t >( cell_off_end ) > vtk_faces.size() )
                        throw MeshReaderError( "VTUReader", "not enough face indices for cell no " + std::to_string(cell) );
                     // faces[cell_off_begin : cell_off_end] -> face data for cell
                     const std::size_t num_faces = vtk_faces.at( cell_off_begin++ );
                     for( std::size_t f = 0; f < num_faces; f++ ) {
                        const std::size_t num_vertices = vtk_faces.at( cell_off_begin++ );
                        for( std::size_t v = 0; v < num_vertices; v++ )
                           faceConnectivity.push_back( vtk_faces.at( cell_off_begin++ ) );
                        faceOffsets.push_back( faceConnectivity.size() );
                        cellConnectivity.push_back( faceIndex++ );
                     }
                     cellOffsets.push_back( cellConnectivity.size() );

                     if( cell_off_begin != cell_off_end )
                        throw MeshReaderError( "VTUReader", "error while parsing the faces data array: did not reach the end offset for cell " + std::to_string(cell) );
                  }

                  this->NumberOfFaces = faceIndex;
                  this->cellOffsetsArray = std::move( cellOffsets );
                  this->cellConnectivityArray = std::move( cellConnectivity );
                  this->faceOffsetsArray = std::move( faceOffsets );
                  this->faceConnectivityArray = std::move( faceConnectivity );
               },
               vtk_facesArray
            );
      }
   }
#endif

public:
   VTUReader() = default;

   VTUReader( const std::string& fileName )
   : XMLVTK( fileName )
   {}

   virtual void detectMesh() override
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
      if( fileType == "UnstructuredGrid" )
         readUnstructuredGrid();
      else
         throw MeshReaderError( "VTUReader", "the reader cannot read data of the type " + fileType + ". Use a different reader if possible." );

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
#else
      throw_no_tinyxml();
#endif
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace noaTNL
