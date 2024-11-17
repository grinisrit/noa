// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <sstream>
#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/VTKTraits.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

/**
 * \brief Writer of data linked with meshes into [VTI format](https://kitware.github.io/vtk-examples/site/VTKFileFormats/).
 *
 * \tparam Mesh type of mesh.
 */
template< typename Mesh >
class VTIWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

   //   using HeaderType = std::uint64_t;
   // LOL, VTK does not support signed header types (but the GridTypeResolver maps unsigned types to signed, so we are good)
   using HeaderType = std::make_unsigned_t< typename Mesh::GlobalIndexType >;

public:
   /**
    * \brief Construct with no parameters is not allowed.
    */
   VTIWriter() = delete;

   /**
    * \brief Constructor of a VTIWriter.
    *
    * \param str output stream used for the export of the data.
    * \param format
    */
   VTIWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::zlib_compressed );

   /**
    * \brief If desired, cycle and time of the simulation can put into the file.
    *
    * This follows the instructions at http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
    *
    * \param cycle is the of the simulation.
    * \param time is the time of the simulation.
    */
   void
   writeMetadata( std::int32_t cycle = -1, double time = -1 );

   // low-level writing method (used also when writing a subdomain for a PVTI dataset)
   void
   writeImageData( const typename Mesh::PointType& origin,
                   const typename Mesh::CoordinatesType& begin,
                   const typename Mesh::CoordinatesType& end,
                   const typename Mesh::PointType& spaceSteps );

   void
   writeImageData( const Mesh& mesh );

   /**
    * \brief Write mesh entites to the output file.
    *
    * It is here only for compatibility with VTUWriter - it just calls writeImageData, the EntityDimension is unused.
    *
    * \param mesh is a mesh to be exported.
    */
   template< int EntityDimension = Mesh::getMeshDimension() >
   void
   writeEntities( const Mesh& mesh );

   /**
    * \brief Writes data linked with mesh vertexes.
    *
    * \tparam Array type of array holding the data.
    * \param array instance of an array holding the data.
    * \param name is a name of data which will appear in the outptu file.
    * \param numberOfComponents is number of compononets of the data for each vertex.
    */
   template< typename Array >
   void
   writePointData( const Array& array, const std::string& name, int numberOfComponents = 1 );

   /**
    * \brief Writes data linked with mesh cells.
    *
    * \tparam Array type of array holding the data.
    * \param array instance of an array holding the data.
    * \param name is a name of data which will appear in the outptu file.
    * \param numberOfComponents is number of compononets of the data for each cell.
    */
   template< typename Array >
   void
   writeCellData( const Array& array, const std::string& name, int numberOfComponents = 1 );

   template< typename Array >
   void
   writeDataArray( const Array& array, const std::string& name, int numberOfComponents = 1 );

   /**
    * \brief Destructor.
    */
   ~VTIWriter();

protected:
   void
   writeHeader();

   void
   writeFooter();

   std::ostream str;

   VTK::FileFormat format;

   // number of points written to the file
   std::uint64_t pointsCount = 0;

   // number of cells (in the VTK sense) written to the file
   std::uint64_t cellsCount = 0;

   // indicator if the <VTKFile> tag is open
   bool vtkfileOpen = false;

   // auxiliary holder for metadata (writeMetadata should be callable before
   // writeEntities to match the VTU format, but the <ImageData> element can be
   // opened only from writeEntities)
   std::stringstream metadata;

   // indicator if the <ImageData> tag is open
   bool imageDataOpen = false;

   // indicator if a <Piece> tag is open
   bool pieceOpen = false;

   // indicators if a <CellData> tag is open or closed
   bool cellDataOpen = false;
   bool cellDataClosed = false;

   // indicators if a <PointData> tag is open or closed
   bool pointDataOpen = false;
   bool pointDataClosed = false;

   void
   openCellData();
   void
   closeCellData();
   void
   openPointData();
   void
   closePointData();

   void
   closePiece();
};

}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/VTIWriter.hpp>
