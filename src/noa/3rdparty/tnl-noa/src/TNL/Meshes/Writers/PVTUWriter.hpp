// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <filesystem>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/PVTUWriter.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh >
void
PVTUWriter< Mesh >::writeMetadata( int cycle, double time )
{
   if( ! vtkfileOpen )
      throw std::logic_error( "writeMetadata has to be called after writeEntities in case of the PVTU format, otherwise header "
                              "attributes would be left unset." );

   if( cycle >= 0 || time >= 0 )
      str << "<FieldData>\n";

   if( cycle >= 0 ) {
      str << "<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">" << cycle << "</DataArray>\n";
   }
   if( time >= 0 ) {
      str.precision( std::numeric_limits< double >::digits10 );
      str << "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">" << time << "</DataArray>\n";
   }

   if( cycle >= 0 || time >= 0 )
      str << "</FieldData>\n";
}

template< typename Mesh >
template< int EntityDimension >
void
PVTUWriter< Mesh >::writeEntities( const DistributedMeshes::DistributedMesh< Mesh >& distributedMesh )
{
   writeEntities< EntityDimension >(
      distributedMesh.getLocalMesh(), distributedMesh.getGhostLevels(), Mesh::Config::dualGraphMinCommonVertices );
}

template< typename Mesh >
template< int EntityDimension >
void
PVTUWriter< Mesh >::writeEntities( const Mesh& mesh, const unsigned GhostLevel, const unsigned MinCommonVertices )
{
   if( ! vtkfileOpen )
      writeHeader( GhostLevel, MinCommonVertices );

   // write points
   writePoints( mesh );
}

template< typename Mesh >
template< typename ValueType >
void
PVTUWriter< Mesh >::writePPointData( const std::string& name, const int numberOfComponents )
{
   if( ! vtkfileOpen )
      throw std::logic_error( "The VTKFile has not been opened yet - call writeEntities first." );
   openPPointData();
   writePDataArray< ValueType >( name, numberOfComponents );
}

template< typename Mesh >
template< typename ValueType >
void
PVTUWriter< Mesh >::writePCellData( const std::string& name, const int numberOfComponents )
{
   if( ! vtkfileOpen )
      throw std::logic_error( "The VTKFile has not been opened yet - call writeEntities first." );
   openPCellData();
   writePDataArray< ValueType >( name, numberOfComponents );
}

template< typename Mesh >
template< typename ValueType >
void
PVTUWriter< Mesh >::writePDataArray( const std::string& name, const int numberOfComponents )
{
   if( numberOfComponents != 0 && numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error( "Unsupported numberOfComponents parameter: " + std::to_string( numberOfComponents ) );

   str << "<PDataArray type=\"" << VTK::getTypeName( ValueType{} ) << "\" ";
   str << "Name=\"" << name << "\" ";
   str << "NumberOfComponents=\"" << numberOfComponents << "\"/>\n";
}

template< typename Mesh >
std::string
PVTUWriter< Mesh >::addPiece( const std::string& mainFileName, const unsigned subdomainIndex )
{
   namespace fs = std::filesystem;

   // get the basename of the main file (filename without extension)
   const fs::path mainPath = mainFileName;
   const fs::path basename = mainPath.stem();
   if( mainPath.extension() != ".pvtu" )
      throw std::logic_error( "The mainFileName parameter must be the name of the "
                              ".pvtu file (i.e., it must have the .pvtu suffix)." );

   // close PCellData and PPointData sections
   closePCellData();
   closePPointData();

   // create subdirectory for subdomains
   const fs::path subdirectory = mainPath.parent_path() / basename;
   fs::create_directory( subdirectory );

   // write <Piece> tag
   const std::string subfile = "subdomain." + std::to_string( subdomainIndex ) + ".vtu";
   const std::string source = basename / subfile;
   str << "<Piece Source=\"" << source << "\"/>\n";

   // return subfile path
   return subdirectory / subfile;
}

template< typename Mesh >
std::string
PVTUWriter< Mesh >::addPiece( const std::string& mainFileName, const MPI_Comm communicator )
{
   std::string source;
   for( int i = 0; i < MPI::GetSize( communicator ); i++ ) {
      const std::string s = addPiece( mainFileName, i );
      if( i == MPI::GetRank( communicator ) )
         source = s;
   }
   return source;
}

template< typename Mesh >
void
PVTUWriter< Mesh >::writeHeader( const unsigned GhostLevel, const unsigned MinCommonVertices )
{
   str << "<?xml version=\"1.0\"?>\n";
   str << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\"";
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
   str << "<PUnstructuredGrid GhostLevel=\"" << GhostLevel << "\"";
   if( MinCommonVertices > 0 )
      str << " MinCommonVertices=\"" << MinCommonVertices << "\"";
   str << ">\n";

   vtkfileOpen = true;
}

template< typename Mesh >
void
PVTUWriter< Mesh >::writePoints( const Mesh& mesh )
{
   str << "<PPoints>\n";
   writePDataArray< typename Mesh::RealType >( "Points", 3 );
   str << "</PPoints>\n";
}

template< typename Mesh >
void
PVTUWriter< Mesh >::writeFooter()
{
   closePCellData();
   closePPointData();
   str << "</PUnstructuredGrid>\n";
   str << "</VTKFile>\n";
}

template< typename Mesh >
PVTUWriter< Mesh >::~PVTUWriter()
{
   if( vtkfileOpen )
      writeFooter();
}

template< typename Mesh >
void
PVTUWriter< Mesh >::openPCellData()
{
   if( pCellDataClosed )
      throw std::logic_error( "The <PCellData> tag has already been closed." );
   closePPointData();
   if( ! pCellDataOpen ) {
      str << "<PCellData>\n";
      pCellDataOpen = true;
   }
}

template< typename Mesh >
void
PVTUWriter< Mesh >::closePCellData()
{
   if( pCellDataOpen ) {
      str << "</PCellData>\n";
      pCellDataClosed = true;
      pCellDataOpen = false;
   }
}

template< typename Mesh >
void
PVTUWriter< Mesh >::openPPointData()
{
   if( pPointDataClosed )
      throw std::logic_error( "The <PPointData> tag has already been closed." );
   closePCellData();
   if( ! pPointDataOpen ) {
      str << "<PPointData>\n";
      pPointDataOpen = true;
   }
}

template< typename Mesh >
void
PVTUWriter< Mesh >::closePPointData()
{
   if( pPointDataOpen ) {
      str << "</PPointData>\n";
      pPointDataClosed = true;
      pPointDataOpen = false;
   }
}

}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL
