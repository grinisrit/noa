// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>  // std::enable_if_t

#include <noa/3rdparty/TNL/Containers/ArrayView.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Polyhedron.h>

namespace noaTNL {
namespace Meshes {
namespace Writers {
namespace detail {

// specialization for meshes
template< typename Mesh >
struct VTUPolyhedralFacesWriter
{
   // specialization for all meshes except polyhedral
   template< typename W, typename M >
   static std::enable_if_t< ! std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value >
   exec( W& writer, const M& mesh )
   {}

   // specialization for polyhedral meshes
   template< typename W, typename M >
   static std::enable_if_t< std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value >
   exec( W& writer, const M& mesh )
   {
      // build the "face stream" for VTK
      using IndexType = typename Mesh::GlobalIndexType;
      std::vector< IndexType > faces, faceoffsets;
      for( IndexType c = 0; c < mesh.template getEntitiesCount< M::getMeshDimension() >(); c++ ) {
         const IndexType num_faces = mesh.template getSubentitiesCount< M::getMeshDimension(), M::getMeshDimension() - 1 >( c );
         faces.push_back( num_faces );
         for( IndexType f = 0; f < num_faces; f++ ) {
            const auto& face = mesh.template getEntity< M::getMeshDimension() - 1 >( mesh.template getSubentityIndex< M::getMeshDimension(), M::getMeshDimension() - 1 >( c, f ) );
            const IndexType num_vertices = face.template getSubentitiesCount< 0 >();
            faces.push_back( num_vertices );
            for( IndexType v = 0; v < num_vertices; v++ ) {
               const IndexType vertex = face.template getSubentityIndex< 0 >( v );
               faces.push_back( vertex );
            }
         }
         faceoffsets.push_back( faces.size() );
      }

      // create array views that can be passed to writeDataArray
      Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > faces_v( faces.data(), faces.size() );
      Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > faceoffsets_v( faceoffsets.data(), faceoffsets.size() );

      // write cells
      writer.writeDataArray( faces_v, "faces", 0 );
      writer.writeDataArray( faceoffsets_v, "faceoffsets", 0 );
   }
};

// specialization for grids
template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTUPolyhedralFacesWriter< Meshes::Grid< Dimension, MeshReal, Device, MeshIndex > >
{
   template< typename W, typename M >
   static void exec( W& writer, const M& mesh )
   {}
};

} // namespace detail
} // namespace Writers
} // namespace Meshes
} // namespace noaTNL
