// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <limits>

#include <TNL/Containers/StaticVector.h>  // TNL::product
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Endianness.h>
#include <TNL/base64.h>
#include <stdexcept>
#ifdef HAVE_ZLIB
   #include <TNL/zlib_compression.h>
#endif

namespace TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh >
void
VTIWriter< Mesh >::writeMetadata( int cycle, double time )
{
   if( ! vtkfileOpen )
      writeHeader();
   if( imageDataOpen )
      throw std::logic_error("The <ImageData> tag is already open, but writeMetadata should be called before writeImageData.");

   if( cycle >= 0 || time >= 0 )
      metadata << "<FieldData>\n";

   if( cycle >= 0 ) {
      metadata << "<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">"
               << cycle << "</DataArray>\n";
   }
   if( time >= 0 ) {
      metadata.precision( std::numeric_limits< double >::digits10 );
      metadata << "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">"
               << time << "</DataArray>\n";
   }

   if( cycle >= 0 || time >= 0 )
      metadata << "</FieldData>\n";
}

template< typename Mesh >
void
VTIWriter< Mesh >::writeImageData( const typename Mesh::PointType& gridOrigin,
                                   const typename Mesh::CoordinatesType& begin,
                                   const typename Mesh::CoordinatesType& end,
                                   const typename Mesh::PointType& spaceSteps )
{
   if( ! vtkfileOpen )
      writeHeader();
   if( imageDataOpen )
      throw std::logic_error("The <ImageData> tag is already open.");

   std::stringstream extent, origin, spacing;

   for( int j = 0; j < Mesh::getMeshDimension(); j++ )
      extent << begin[ j ] <<  " " << end[ j ] << " ";
   // VTK knows only 3D grids
   for( int j = Mesh::getMeshDimension(); j < 3; j++ )
      extent << "0 0 ";

   for( int j = 0; j < Mesh::getMeshDimension(); j++ )
      origin << std::scientific << gridOrigin[ j ] << " ";
   // VTK knows only 3D grids
   for( int j = Mesh::getMeshDimension(); j < 3; j++ )
      origin << 0 << " ";

   for( int j = 0; j < Mesh::getMeshDimension(); j++ )
      spacing << std::scientific << spaceSteps[ j ] << " ";
   // VTK knows only 3D grids
   for( int j = Mesh::getMeshDimension(); j < 3; j++ )
      spacing << 0 << " ";

   str << "<ImageData WholeExtent=\"" << extent.str() << "\" Origin=\"" << origin.str() << "\" Spacing=\"" << spacing.str() << "\">\n";
   imageDataOpen = true;

   str << "<Piece Extent=\"" << extent.str() << "\">\n";
   pieceOpen = true;

   // write metadata if present
   if( ! metadata.str().empty() ) {
      str << metadata.str();
      metadata.str() = "";
   }

   // sets points and cells counts
   pointsCount = TNL::product( end - begin + 1 );
   cellsCount = TNL::product( end - begin );
}

template< typename Mesh >
void
VTIWriter< Mesh >::writeImageData( const Mesh& mesh )
{
   writeImageData( mesh.getOrigin(), 0, mesh.getDimensions(), mesh.getSpaceSteps() );
}

template< typename Mesh >
   template< int EntityDimension >
void
VTIWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   writeImageData( mesh );
}

template< typename Mesh >
   template< typename Array >
void
VTIWriter< Mesh >::writePointData( const Array& array,
                                   const std::string& name,
                                   const int numberOfComponents )
{
   if( ! pieceOpen )
      throw std::logic_error("The <Piece> tag has not been opened yet - call writeEntities first.");
   if( array.getSize() / numberOfComponents != typename Array::IndexType(pointsCount) )
      throw std::length_error("Mismatched array size for <PointData> section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(pointsCount) + " points in the file)");
   openPointData();
   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTIWriter< Mesh >::writeCellData( const Array& array,
                                  const std::string& name,
                                  const int numberOfComponents )
{
   if( ! pieceOpen )
      throw std::logic_error("The <Piece> tag has not been opened yet - call writeEntities first.");
   if( array.getSize() / numberOfComponents != typename Array::IndexType(cellsCount) )
      throw std::length_error("Mismatched array size for <CellData> section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(cellsCount) + " cells in the file)");
   openCellData();
   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTIWriter< Mesh >::writeDataArray( const Array& array,
                                   const std::string& name,
                                   const int numberOfComponents )
{
   // use a host buffer if direct access to the array elements is not possible
   if( std::is_same< typename Array::DeviceType, Devices::Cuda >::value )
   {
      using HostArray = typename Array::template Self< std::remove_const_t< typename Array::ValueType >, Devices::Host, typename Array::IndexType >;
      HostArray hostBuffer;
      hostBuffer = array;
      writeDataArray( hostBuffer, name, numberOfComponents );
      return;
   }

   if( numberOfComponents != 0 && numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error("Unsupported numberOfComponents parameter: " + std::to_string(numberOfComponents));

   // write DataArray header
   str << "<DataArray type=\"" << VTK::getTypeName( array[0] ) << "\"";
   str << " Name=\"" << name << "\"";
   if( numberOfComponents > 0 )
      str << " NumberOfComponents=\"" << numberOfComponents << "\"";
   str << " format=\"" << ((format == VTK::FileFormat::ascii) ? "ascii" : "binary") << "\">\n";

   switch( format )
   {
      case VTK::FileFormat::ascii:
         str.precision( std::numeric_limits< typename Array::ValueType >::digits10 );
         for( typename Array::IndexType i = 0; i < array.getSize(); i++ )
            // If Array::ValueType is uint8_t, it might be a typedef for unsigned char, which
            // would be normally printed as char rather than a number. Hence, we use the trick
            // with unary operator+, see https://stackoverflow.com/a/28414758
            str << +array[i] << " ";
         str << "\n";
         break;
      case VTK::FileFormat::zlib_compressed:
#ifdef HAVE_ZLIB
         write_compressed_block< HeaderType >( array.getData(), array.getSize(), str );
         str << "\n";
         break;
#endif
         // fall through to binary if HAVE_ZLIB is not defined
      case VTK::FileFormat::binary:
         base64::write_encoded_block< HeaderType >( array.getData(), array.getSize(), str );
         str << "\n";
         break;
   }

   // write DataArray footer
   str << "</DataArray>\n";
}

template< typename Mesh >
void
VTIWriter< Mesh >::writeHeader()
{
   str << "<?xml version=\"1.0\"?>\n";
   str << "<VTKFile type=\"ImageData\" version=\"1.0\"";
   if( isLittleEndian() )
      str << " byte_order=\"LittleEndian\"";
   else
      str << " byte_order=\"BigEndian\"";
   str << " header_type=\"" << VTK::getTypeName( HeaderType{} ) << "\"";
#ifdef HAVE_ZLIB
   if( format == VTK::FileFormat::zlib_compressed )
      str << " compressor=\"vtkZLibDataCompressor\"";
#endif
   str << ">\n";

   vtkfileOpen = true;
}

template< typename Mesh >
void
VTIWriter< Mesh >::writeFooter()
{
   closePiece();
   str << "</ImageData>\n";
   str << "</VTKFile>\n";
}

template< typename Mesh >
VTIWriter< Mesh >::~VTIWriter()
{
   if( vtkfileOpen )
      writeFooter();
}

template< typename Mesh >
void
VTIWriter< Mesh >::openCellData()
{
   if( cellDataClosed )
      throw std::logic_error("The <CellData> tag has already been closed in the current <Piece> section.");
   closePointData();
   if( ! cellDataOpen ) {
      str << "<CellData>\n";
      cellDataOpen = true;
   }
}

template< typename Mesh >
void
VTIWriter< Mesh >::closeCellData()
{
   if( cellDataOpen ) {
      str << "</CellData>\n";
      cellDataClosed = true;
      cellDataOpen = false;
   }
}

template< typename Mesh >
void
VTIWriter< Mesh >::openPointData()
{
   if( pointDataClosed )
      throw std::logic_error("The <PointData> tag has already been closed in the current <Piece> section.");
   closeCellData();
   if( ! pointDataOpen ) {
      str << "<PointData>\n";
      pointDataOpen = true;
   }
}

template< typename Mesh >
void
VTIWriter< Mesh >::closePointData()
{
   if( pointDataOpen ) {
      str << "</PointData>\n";
      pointDataClosed = true;
      pointDataOpen = false;
   }
}

template< typename Mesh >
void
VTIWriter< Mesh >::closePiece()
{
   if( pieceOpen ) {
      closeCellData();
      closePointData();
      str << "</Piece>\n";

      // reset indicators - new <Piece> can be started
      pieceOpen = false;
      cellDataOpen = cellDataClosed = false;
      pointDataOpen = pointDataClosed = false;
   }
}

} // namespace Writers
} // namespace Meshes
} // namespace TNL
