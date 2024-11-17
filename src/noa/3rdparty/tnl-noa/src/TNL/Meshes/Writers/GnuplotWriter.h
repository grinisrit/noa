// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

/**
 * \brief Writer of data linked with meshes into [Gnuplot format](http://www.gnuplot.info/).
 *
 * \tparam Mesh type of mesh.
 */
template< typename Mesh >
class GnuplotWriter
{
public:
   /**
    * \brief Construct with no parameters is not allowed.
    */
   GnuplotWriter() = delete;

   /**
    * \brief Constructor of a VTIWriter.
    *
    * \param str output stream used for the export of the data.
    */
   GnuplotWriter( std::ostream& str );

   /**
    * \brief This method is for compatinility wirh other writers.
    *
    * It just writes a header.
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
    * \param mesh instance of the mesh.
    * \param array instance of an array holding the data.
    * \param name is a name of data which will appear in the outptu file.
    * \param numberOfComponents is number of compononets of the data for each vertex.
    */
   template< typename Array >
   void
   writePointData( const Mesh& mesh, const Array& array, const std::string& name, int numberOfComponents = 1 );

   /**
    * \brief Writes data linked with mesh cells.
    *
    * \tparam Array type of array holding the data.
    * \param mesh instance of the mesh.
    * \param array instance of an array holding the data.
    * \param name is a name of data which will appear in the outptu file.
    * \param numberOfComponents is number of compononets of the data for each cell.
    */
   template< typename Array >
   void
   writeCellData( const Mesh& mesh, const Array& array, const std::string& name, int numberOfComponents = 1 );

   void
   writeHeader( const Mesh& mesh );

   template< typename Element >
   static void
   write( std::ostream& str, const Element& d );

   template< typename Real >
   static void
   write( std::ostream& str, const Containers::StaticVector< 1, Real >& d );

   template< typename Real >
   static void
   write( std::ostream& str, const Containers::StaticVector< 2, Real >& d );

   template< typename Real >
   static void
   write( std::ostream& str, const Containers::StaticVector< 3, Real >& d );

protected:
   std::ostream& str;
};

}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/GnuplotWriter.hpp>
