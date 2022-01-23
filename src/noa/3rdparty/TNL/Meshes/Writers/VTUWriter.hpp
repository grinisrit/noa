// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <limits>

#include <noa/3rdparty/TNL/Containers/Array.h>
#include <noa/3rdparty/TNL/Meshes/Writers/VTUWriter.h>
#include <noa/3rdparty/TNL/Meshes/Writers/detail/VTUMeshEntitiesCollector.h>
#include <noa/3rdparty/TNL/Meshes/Writers/detail/VTUPolyhedralFacesWriter.h>
#include <noa/3rdparty/TNL/Endianness.h>
#include <noa/3rdparty/TNL/base64.h>
#include <noa/3rdparty/TNL/zlib_compression.h>


namespace noa::TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh >
void
VTUWriter< Mesh >::writeMetadata( int cycle, double time )
{
   if( ! vtkfileOpen )
      writeHeader();

   if( cycle >= 0 || time >= 0 )
      str << "<FieldData>\n";

   if( cycle >= 0 ) {
      str << "<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">"
          << cycle << "</DataArray>\n";
   }
   if( time >= 0 ) {
      str.precision( std::numeric_limits< double >::digits10 );
      str << "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">"
          << time << "</DataArray>\n";
   }

   if( cycle >= 0 || time >= 0 )
      str << "</FieldData>\n";
}

template< typename Mesh >
   template< int EntityDimension >
void
VTUWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   // count points and cells before any writing
   pointsCount = mesh.template getEntitiesCount< typename Mesh::Vertex >();
   using EntityType = typename Mesh::template EntityType< EntityDimension >;
   cellsCount = mesh.template getEntitiesCount< EntityType >();

   if( ! vtkfileOpen )
      writeHeader();
   closePiece();
   str << "<Piece NumberOfPoints=\"" << pointsCount << "\" NumberOfCells=\"" << cellsCount << "\">\n";
   pieceOpen = true;

   // write points
   writePoints( mesh );

   // collect all data before writing
   using IndexType = typename Mesh::GlobalIndexType;
   std::vector< IndexType > connectivity, offsets;
   std::vector< std::uint8_t > types;
   detail::MeshEntitiesVTUCollector< Mesh, EntityDimension >::exec( mesh, connectivity, offsets, types );

   // create array views that can be passed to writeDataArray
   Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > connectivity_v( connectivity.data(), connectivity.size() );
   Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > offsets_v( offsets.data(), offsets.size() );
   Containers::ArrayView< std::uint8_t, Devices::Host, std::uint64_t > types_v( types.data(), types.size() );

   // write cells
   str << "<Cells>\n";
   writeDataArray( connectivity_v, "connectivity", 0 );
   writeDataArray( offsets_v, "offsets", 0 );
   writeDataArray( types_v, "types", 0 );
   // write faces if the mesh is polyhedral
   detail::VTUPolyhedralFacesWriter< Mesh >::exec( *this, mesh );
   str << "</Cells>\n";
}

template< typename Mesh >
   template< typename Array >
void
VTUWriter< Mesh >::writePointData( const Array& array,
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
VTUWriter< Mesh >::writeCellData( const Array& array,
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
VTUWriter< Mesh >::writeDataArray( const Array& array,
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
   using ValueType = decltype(array[0]);
   str << "<DataArray type=\"" << VTK::getTypeName( ValueType{} ) << "\"";
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
VTUWriter< Mesh >::writePoints( const Mesh& mesh )
{
   // copy all coordinates into a contiguous array
   using BufferType = Containers::Array< typename Mesh::RealType, Devices::Host, typename Mesh::GlobalIndexType >;
   BufferType buffer( 3 * pointsCount );
   typename Mesh::GlobalIndexType k = 0;
   for( std::uint64_t i = 0; i < pointsCount; i++ ) {
      const auto& vertex = mesh.template getEntity< typename Mesh::Vertex >( i );
      const auto& point = vertex.getPoint();
      for( int j = 0; j < point.getSize(); j++ )
         buffer[ k++ ] = point[ j ];
      // VTK needs zeros for unused dimensions
      for( int j = point.getSize(); j < 3; j++ )
         buffer[ k++ ] = 0;
   }

   // write the buffer
   str << "<Points>\n";
   writeDataArray( buffer, "Points", 3 );
   str << "</Points>\n";
}

template< typename Mesh >
void
VTUWriter< Mesh >::writeHeader()
{
   str << "<?xml version=\"1.0\"?>\n";
   str << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\"";
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
   str << "<UnstructuredGrid>\n";

   vtkfileOpen = true;
}

template< typename Mesh >
void
VTUWriter< Mesh >::writeFooter()
{
   closePiece();
   str << "</UnstructuredGrid>\n";
   str << "</VTKFile>\n";
}

template< typename Mesh >
VTUWriter< Mesh >::~VTUWriter()
{
   if( vtkfileOpen )
      writeFooter();
}

template< typename Mesh >
void
VTUWriter< Mesh >::openCellData()
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
VTUWriter< Mesh >::closeCellData()
{
   if( cellDataOpen ) {
      str << "</CellData>\n";
      cellDataClosed = true;
      cellDataOpen = false;
   }
}

template< typename Mesh >
void
VTUWriter< Mesh >::openPointData()
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
VTUWriter< Mesh >::closePointData()
{
   if( pointDataOpen ) {
      str << "</PointData>\n";
      pointDataClosed = true;
      pointDataOpen = false;
   }
}

template< typename Mesh >
void
VTUWriter< Mesh >::closePiece()
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
} // namespace noa::TNL
