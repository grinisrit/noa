// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <memory>  // std::unique_ptr
#include <filesystem>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/PVTIWriter.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

template< typename Grid >
void
PVTIWriter< Grid >::writeMetadata( int cycle, double time )
{
   if( ! vtkfileOpen )
      writeHeader();
   if( pImageDataOpen )
      throw std::logic_error(
         "The <PImageData> tag is already open, but writeMetadata should be called before writeImageData." );

   if( cycle >= 0 || time >= 0 )
      metadata << "<FieldData>\n";

   if( cycle >= 0 ) {
      metadata << "<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">" << cycle
               << "</DataArray>\n";
   }
   if( time >= 0 ) {
      metadata.precision( std::numeric_limits< double >::digits10 );
      metadata << "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">" << time
               << "</DataArray>\n";
   }

   if( cycle >= 0 || time >= 0 )
      metadata << "</FieldData>\n";
}

template< typename Grid >
void
PVTIWriter< Grid >::writeImageData( const DistributedMeshes::DistributedMesh< Grid >& distributedGrid )
{
   writeImageData( distributedGrid.getGlobalGrid(),
                   distributedGrid.getGhostLevels() );  // TODO: ..., Grid::Config::dualGraphMinCommonVertices );
}

template< typename Grid >
void
PVTIWriter< Grid >::writeImageData( const Grid& globalGrid, const unsigned GhostLevel, const unsigned MinCommonVertices )
{
   if( ! vtkfileOpen )
      writeHeader();
   if( pImageDataOpen )
      throw std::logic_error( "The <PImageData> tag is already open." );

   std::stringstream extent;
   std::stringstream origin;
   std::stringstream spacing;

   auto dims = globalGrid.getDimensions();
   for( int j = 0; j < dims.getSize(); j++ )
      extent << "0 " << dims[ j ] << " ";
   // VTK knows only 3D grids
   for( int j = dims.getSize(); j < 3; j++ )
      extent << "0 0 ";

   auto o = globalGrid.getOrigin();
   for( int j = 0; j < o.getSize(); j++ )
      origin << std::scientific << o[ j ] << " ";
   // VTK knows only 3D grids
   for( int j = o.getSize(); j < 3; j++ )
      origin << 0 << " ";

   auto h = globalGrid.getSpaceSteps();
   for( int j = 0; j < h.getSize(); j++ )
      spacing << std::scientific << h[ j ] << " ";
   // VTK knows only 3D grids
   for( int j = h.getSize(); j < 3; j++ )
      spacing << 0 << " ";

   str << "<PImageData"
       << " WholeExtent=\"" << extent.str() << "\""
       << " Origin=\"" << origin.str() << "\""
       << " Spacing=\"" << spacing.str() << "\""
       << " GhostLevel=\"" << GhostLevel << "\"";
   if( MinCommonVertices > 0 )
      str << " MinCommonVertices=\"" << MinCommonVertices << "\"";
   str << ">\n";
   pImageDataOpen = true;

   // write metadata if present
   if( ! metadata.str().empty() ) {
      str << metadata.str();
      metadata.str() = "";
   }
}

template< typename Grid >
template< int EntityDimension >
void
PVTIWriter< Grid >::writeEntities( const DistributedMeshes::DistributedMesh< Grid >& distributedMesh )
{
   writeImageData( distributedMesh );
}

template< typename Grid >
template< int EntityDimension >
void
PVTIWriter< Grid >::writeEntities( const Grid& grid, const unsigned GhostLevel, const unsigned MinCommonVertices )
{
   writeImageData( grid, GhostLevel, MinCommonVertices );
}

template< typename Grid >
template< typename ValueType >
void
PVTIWriter< Grid >::writePPointData( const std::string& name, const int numberOfComponents )
{
   if( ! vtkfileOpen )
      throw std::logic_error( "The VTKFile has not been opened yet - call writeEntities first." );
   openPPointData();
   writePDataArray< ValueType >( name, numberOfComponents );
}

template< typename Grid >
template< typename ValueType >
void
PVTIWriter< Grid >::writePCellData( const std::string& name, const int numberOfComponents )
{
   if( ! vtkfileOpen )
      throw std::logic_error( "The VTKFile has not been opened yet - call writeEntities first." );
   openPCellData();
   writePDataArray< ValueType >( name, numberOfComponents );
}

template< typename Grid >
template< typename ValueType >
void
PVTIWriter< Grid >::writePDataArray( const std::string& name, const int numberOfComponents )
{
   if( numberOfComponents != 0 && numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error( "Unsupported numberOfComponents parameter: " + std::to_string( numberOfComponents ) );

   str << "<PDataArray type=\"" << VTK::getTypeName( ValueType{} ) << "\" ";
   str << "Name=\"" << name << "\" ";
   str << "NumberOfComponents=\"" << numberOfComponents << "\"/>\n";
}

template< typename Grid >
std::string
PVTIWriter< Grid >::addPiece( const std::string& mainFileName,
                              const unsigned subdomainIndex,
                              const typename Grid::CoordinatesType& globalBegin,
                              const typename Grid::CoordinatesType& globalEnd )
{
   namespace fs = std::filesystem;

   // get the basename of the main file (filename without extension)
   const fs::path mainPath = mainFileName;
   const fs::path basename = mainPath.stem();
   if( mainPath.extension() != ".pvti" )
      throw std::logic_error( "The mainFileName parameter must be the name of the "
                              ".pvti file (i.e., it must have the .pvti suffix)." );

   // close PCellData and PPointData sections
   closePCellData();
   closePPointData();

   // prepare the extent
   std::stringstream extent;
   for( int j = 0; j < Grid::getMeshDimension(); j++ )
      extent << globalBegin[ j ] << " " << globalEnd[ j ] << " ";
   // VTK knows only 3D grids
   for( int j = Grid::getMeshDimension(); j < 3; j++ )
      extent << "0 0 ";

   // create subdirectory for subdomains
   const fs::path subdirectory = mainPath.parent_path() / basename;
   fs::create_directory( subdirectory );

   // write <Piece> tag
   const std::string subfile = "subdomain." + std::to_string( subdomainIndex ) + ".vti";
   const std::string source = basename / subfile;
   str << "<Piece Extent=\"" << extent.str() << "\" Source=\"" << source << "\"/>\n";

   // return subfile path
   return subdirectory / subfile;
}

template< typename Grid >
std::string
PVTIWriter< Grid >::addPiece( const std::string& mainFileName,
                              const DistributedMeshes::DistributedMesh< Grid >& distributedMesh )
{
   const MPI::Comm& communicator = distributedMesh.getCommunicator();
   const typename Grid::CoordinatesType& globalBegin = distributedMesh.getGlobalBegin() - distributedMesh.getLowerOverlap();
   const typename Grid::CoordinatesType& globalEnd =
      globalBegin + distributedMesh.getLocalSize() + distributedMesh.getUpperOverlap();

   // exchange globalBegin and globalEnd among the ranks
   const int nproc = communicator.size();
   std::unique_ptr< typename Grid::CoordinatesType[] > beginsForScatter{ new typename Grid::CoordinatesType[ nproc ] };
   std::unique_ptr< typename Grid::CoordinatesType[] > endsForScatter{ new typename Grid::CoordinatesType[ nproc ] };
   for( int i = 0; i < nproc; i++ ) {
      beginsForScatter[ i ] = globalBegin;
      endsForScatter[ i ] = globalEnd;
   }
   std::unique_ptr< typename Grid::CoordinatesType[] > globalBegins{ new typename Grid::CoordinatesType[ nproc ] };
   std::unique_ptr< typename Grid::CoordinatesType[] > globalEnds{ new typename Grid::CoordinatesType[ nproc ] };
   // NOTE: exchanging general data types does not work with MPI
   // MPI::Alltoall( beginsForScatter.get(), 1, globalBegins.get(), 1, communicator );
   // MPI::Alltoall( endsForScatter.get(), 1, globalEnds.get(), 1, communicator );
   MPI::Alltoall( (char*) beginsForScatter.get(),
                  sizeof( typename Grid::CoordinatesType ),
                  (char*) globalBegins.get(),
                  sizeof( typename Grid::CoordinatesType ),
                  communicator );
   MPI::Alltoall( (char*) endsForScatter.get(),
                  sizeof( typename Grid::CoordinatesType ),
                  (char*) globalEnds.get(),
                  sizeof( typename Grid::CoordinatesType ),
                  communicator );

   // add pieces for all ranks, return the source for the current rank
   std::string source;
   for( int i = 0; i < communicator.size(); i++ ) {
      const std::string s = addPiece( mainFileName, i, globalBegins[ i ], globalEnds[ i ] );
      if( i == communicator.rank() )
         source = s;
   }
   return source;
}

template< typename Grid >
void
PVTIWriter< Grid >::writeHeader()
{
   str << "<?xml version=\"1.0\"?>\n";
   str << "<VTKFile type=\"PImageData\" version=\"1.0\"";
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

template< typename Grid >
void
PVTIWriter< Grid >::writeFooter()
{
   closePCellData();
   closePPointData();
   str << "</PImageData>\n";
   str << "</VTKFile>\n";
}

template< typename Grid >
PVTIWriter< Grid >::~PVTIWriter()
{
   if( vtkfileOpen )
      writeFooter();
}

template< typename Grid >
void
PVTIWriter< Grid >::openPCellData()
{
   if( pCellDataClosed )
      throw std::logic_error( "The <PCellData> tag has already been closed." );
   closePPointData();
   if( ! pCellDataOpen ) {
      str << "<PCellData>\n";
      pCellDataOpen = true;
   }
}

template< typename Grid >
void
PVTIWriter< Grid >::closePCellData()
{
   if( pCellDataOpen ) {
      str << "</PCellData>\n";
      pCellDataClosed = true;
      pCellDataOpen = false;
   }
}

template< typename Grid >
void
PVTIWriter< Grid >::openPPointData()
{
   if( pPointDataClosed )
      throw std::logic_error( "The <PPointData> tag has already been closed." );
   closePCellData();
   if( ! pPointDataOpen ) {
      str << "<PPointData>\n";
      pPointDataOpen = true;
   }
}

template< typename Grid >
void
PVTIWriter< Grid >::closePPointData()
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
