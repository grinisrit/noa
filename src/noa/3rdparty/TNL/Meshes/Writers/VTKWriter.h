// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <noa/3rdparty/TNL/Meshes/VTKTraits.h>

namespace noaTNL {
namespace Meshes {
//! \brief Namespace for mesh writers.
namespace Writers {

template< typename Mesh >
class VTKWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also space dimension when grids allow it
//   static_assert( Mesh::getSpaceDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

public:

   VTKWriter() = delete;

   VTKWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::binary )
   : str(str.rdbuf()), format(format)
   {
      if( format != VTK::FileFormat::ascii && format != VTK::FileFormat::binary )
         throw std::domain_error("The Legacy VTK file formats support only ASCII and BINARY formats.");
   }

   // If desired, cycle and time of the simulation can put into the file. This follows the instructions at
   // http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
   void writeMetadata( std::int32_t cycle = -1, double time = -1 );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh );

   template< typename Array >
   void writePointData( const Array& array,
                        const std::string& name,
                        const int numberOfComponents = 1 );

   template< typename Array >
   void writeCellData( const Array& array,
                       const std::string& name,
                       const int numberOfComponents = 1 );

   template< typename Array >
   void writeDataArray( const Array& array,
                        const std::string& name,
                        const int numberOfComponents = 1 );

protected:
   void writePoints( const Mesh& mesh );

   void writeHeader();

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

} // namespace Writers
} // namespace Meshes
} // namespace noaTNL

#include <noa/3rdparty/TNL/Meshes/Writers/VTKWriter.hpp>
