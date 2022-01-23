// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Meshes/Writers/VTIWriter.h>
#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/DistributedGrid.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

// NOTE: Grid should be the local grid type, not DistributedMesh<Grid<...>>
template< typename Grid >
class PVTIWriter
{
   static_assert( Grid::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

//   using HeaderType = std::uint64_t;
   // LOL, VTK does not support signed header types (but the GridTypeResolver maps unsigned types to signed, so we are good)
   using HeaderType = std::make_unsigned_t< typename Grid::GlobalIndexType >;
public:

   PVTIWriter() = delete;

   PVTIWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::zlib_compressed )
   : str(str.rdbuf()), format(format)
   {}

   // If desired, cycle and time of the simulation can put into the file. This follows the instructions at
   // http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
   void writeMetadata( std::int32_t cycle = -1, double time = -1 );

   void writeImageData( const DistributedMeshes::DistributedMesh< Grid >& distributedMesh );

   void writeImageData( const Grid& globalGrid,
                        const unsigned GhostLevel = 0,
                        const unsigned MinCommonVertices = 0 );

   // Only for compatibility with VTUWriter - calls writeImageData, the EntityDimension is unused
   template< int EntityDimension = Grid::getMeshDimension() >
   void writeEntities( const DistributedMeshes::DistributedMesh< Grid >& distributedMesh );

   // Only for compatibility with VTUWriter - calls writeImageData, the EntityDimension is unused
   template< int EntityDimension = Grid::getMeshDimension() >
   void writeEntities( const Grid& grid,
                       const unsigned GhostLevel = 0,
                       const unsigned MinCommonVertices = 0 );

   template< typename ValueType >
   void writePPointData( const std::string& name,
                         const int numberOfComponents = 1 );

   template< typename ValueType >
   void writePCellData( const std::string& name,
                        const int numberOfComponents = 1 );

   template< typename ValueType >
   void writePDataArray( const std::string& name,
                         const int numberOfComponents = 1 );

   // add a single piece and return its source path
   // (useful for sequential writing, e.g. from tnl-decompose-grid)
   std::string addPiece( const std::string& mainFileName,
                         const unsigned subdomainIndex,
                         const typename Grid::CoordinatesType& globalBegin,
                         const typename Grid::CoordinatesType& globalEnd );

   // add all pieces and return the source path for the current rank
   // (useful for parallel writing)
   std::string addPiece( const std::string& mainFileName,
                         const DistributedMeshes::DistributedMesh< Grid >& distributedMesh );

   ~PVTIWriter();

protected:
   void writeHeader();

   void writeFooter();

   std::ostream str;

   VTK::FileFormat format;

   // indicator if the <VTKFile> tag is open
   bool vtkfileOpen = false;

   // auxiliary holder for metadata (writeMetadata should be callable before
   // writeEntities to match the VTU format, but the <ImageData> element can be
   // opened only from writeEntities)
   std::stringstream metadata;

   // indicator if the <PImageData> tag is open
   bool pImageDataOpen = false;

   // indicators if a <PCellData> tag is open or closed
   bool pCellDataOpen = false;
   bool pCellDataClosed = false;

   // indicators if a <PPointData> tag is open or closed
   bool pPointDataOpen = false;
   bool pPointDataClosed = false;

   void openPCellData();
   void closePCellData();
   void openPPointData();
   void closePPointData();
};

} // namespace Writers
} // namespace Meshes
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Meshes/Writers/PVTIWriter.hpp>
