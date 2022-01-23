// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/TNL/Meshes/VTKTraits.h>

namespace noa::TNL {
namespace Meshes {
//! \brief Namespace for mesh writers.
namespace Writers {

namespace details {

template< typename Mesh, int EntityDimension, int SubDimension > struct MeshEntitiesFPMAWriter;

} // namespace details

template< typename Mesh >
class FPMAWriter
{
   static_assert( std::is_same< typename Mesh::Cell::EntityTopology, Topologies::Polyhedron >::value, "The FPMA format supports polyhedral meshes." );

   template< int EntityDimension, int SubDimension >
   using EntitiesWriter = details::MeshEntitiesFPMAWriter< Mesh, EntityDimension, SubDimension >;

public:
   using IndexType = typename Mesh::GlobalIndexType;

   FPMAWriter() = delete;

   FPMAWriter( std::ostream& str )
   : str(str.rdbuf())
   {
   }

   void writeEntities( const Mesh& mesh );

protected:
   void writePoints( const Mesh& mesh );

   std::ostream str;

   // number of cells written to the file
   //IndexType cellsCount = 0;

   // number of faces written to the file
   //IndexType facesCount = 0;

   // number of points written to the file
   //IndexType pointsCount = 0;
};

} // namespace Writers
} // namespace Meshes
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Meshes/Writers/FPMAWriter.hpp>
