// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/VTKTraits.h>

namespace noa::TNL {
namespace Meshes {
//! \brief Namespace for mesh writers.
namespace Writers {

/**
 * \brief Writer of data linked with meshes into [VTK format](https://kitware.github.io/vtk-examples/site/VTKFileFormats/).
 *
 * \tparam Mesh type of mesh.
 */
template< typename Mesh >
class VTKWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also space dimension when grids allow it
   //   static_assert( Mesh::getSpaceDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

public:
   /**
    * \brief Construct with no parameters is not allowed.
    */
   VTKWriter() = delete;

   /**
    * \brief Constructor of a VTIWriter.
    *
    * \param str output stream used for the export of the data.
    * \param format
    */
   VTKWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::binary );

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

   /**
    * \brief Write mesh entites to the output file.
    *
    * \tparam EntityDimension is a dimension of entities to be exported.
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

protected:
   void
   writePoints( const Mesh& mesh );

   void
   writeHeader();

   std::ostream str;

   VTK::FileFormat format;

   // number of cells (in the VTK sense) written to the file
   std::uint64_t cellsCount = 0;

   // number of points written to the file
   std::uint64_t pointsCount = 0;

   // indicator if the header has been written
   bool headerWritten = false;

   // number of data arrays written in each section
   int cellDataArrays = 0;
   int pointDataArrays = 0;

   // indicator of the current section
   VTK::DataType currentSection = VTK::DataType::CellData;
};

}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/VTKWriter.hpp>
