// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Meshes/Writers/VTUWriter.h>
#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

// NOTE: Mesh should be the local mesh type, not DistributedMesh
template< typename Mesh >
class PVTUWriter
{
   using HeaderType = std::uint64_t;
public:

   PVTUWriter() = delete;

   PVTUWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::zlib_compressed )
   : str(str.rdbuf()), format(format)
   {}

   // If desired, cycle and time of the simulation can put into the file. This follows the instructions at
   // http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
   void writeMetadata( std::int32_t cycle = -1, double time = -1 );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const DistributedMeshes::DistributedMesh< Mesh >& distributedMesh );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh,
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
   // (useful for sequential writing, e.g. from tnl-decompose-mesh)
   std::string addPiece( const std::string& mainFileName,
                         const unsigned subdomainIndex );

   // add all pieces and return the source path for the current rank
   // (useful for parallel writing)
   std::string addPiece( const std::string& mainFileName,
                         const MPI_Comm communicator );

   ~PVTUWriter();

protected:
   void writeHeader( const unsigned GhostLevel = 0,
                     const unsigned MinCommonVertices = 0 );

   void writePoints( const Mesh& mesh );

   void writeFooter();

   std::ostream str;

   VTK::FileFormat format;

   // indicator if the <VTKFile> tag is open
   bool vtkfileOpen = false;

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

#include <noa/3rdparty/TNL/Meshes/Writers/PVTUWriter.hpp>
