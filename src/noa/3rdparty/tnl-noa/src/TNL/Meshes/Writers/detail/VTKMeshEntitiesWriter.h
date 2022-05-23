// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <ostream>

#include <noa/3rdparty/tnl-noa/src/TNL/Endianness.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/MeshEntity.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/VTKTraits.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {
namespace detail {

template< typename T >
void
writeValue( VTK::FileFormat format, std::ostream& str, T value )
{
   if( format == VTK::FileFormat::binary ) {
      value = forceBigEndian( value );
      str.write( reinterpret_cast< const char* >( &value ), sizeof( T ) );
   }
   else {
      // precision affects only floating-point types, not integers
      str.precision( std::numeric_limits< T >::digits10 );
      str << value << " ";
   }
}

// TODO: specialization for disabled entities
// Unstructured meshes, entities
template< typename Mesh, int EntityDimension, typename EntityType = typename Mesh::template EntityType< EntityDimension > >
struct VTKMeshEntitiesWriter
{
   template< typename Index >
   static void
   writeOffsets( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         offset += entity.template getSubentitiesCount< 0 >();
         writeValue< Index >( format, str, offset );
      }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         const Index verticesPerEntity = entity.template getSubentitiesCount< 0 >();
         for( Index j = 0; j < verticesPerEntity; j++ )
            writeValue< Index >( format, str, entity.template getSubentityIndex< 0 >( j ) );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// Unstructured meshes, polyhedrons
template< typename Mesh >
struct VTKMeshEntitiesWriter< Mesh, 3, MeshEntity< typename Mesh::Config, typename Mesh::DeviceType, Topologies::Polyhedron > >
{
   template< typename Index >
   static void
   writeOffsets( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      const Index entitiesCount = mesh.template getEntitiesCount< 3 >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const Index num_faces = mesh.template getSubentitiesCount< 3, 2 >( i );
         // one value (num_faces) for each cell
         offset++;
         // one value (num_vertices) for each face
         offset += num_faces;
         // list of vertex indices for each face
         for( Index f = 0; f < num_faces; f++ ) {
            const Index face = mesh.template getSubentityIndex< 3, 2 >( i, f );
            offset += mesh.template getSubentitiesCount< 2, 0 >( face );
         }
         writeValue< Index >( format, str, offset );
      }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      const Index entitiesCount = mesh.template getEntitiesCount< 3 >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const Index num_faces = mesh.template getSubentitiesCount< 3, 2 >( i );
         writeValue< Index >( format, str, num_faces );

         for( Index f = 0; f < num_faces; f++ ) {
            const Index face = mesh.template getSubentityIndex< 3, 2 >( i, f );
            const Index num_vertices = mesh.template getSubentitiesCount< 2, 0 >( face );
            writeValue< Index >( format, str, num_vertices );
            for( Index v = 0; v < num_vertices; v++ ) {
               const Index vertex = mesh.template getSubentityIndex< 2, 0 >( face, v );
               writeValue< Index >( format, str, vertex );
            }
         }

         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// Unstructured meshes, vertices
template< typename Mesh >
struct VTKMeshEntitiesWriter< Mesh, 0, MeshEntity< typename Mesh::Config, typename Mesh::DeviceType, Topologies::Vertex > >
{
   template< typename Index >
   static void
   writeOffsets( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< 0 >;

      Index offset = 0;
      writeValue< Index >( format, str, offset );

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ )
         writeValue< Index >( format, str, ++offset );

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< 0 >;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         writeValue< Index >( format, str, i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 1D grids, cells
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
         offset += 2;
         writeValue< Index >( format, str, offset );
      }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
         writeValue< Index >( format, str, i );
         writeValue< Index >( format, str, i + 1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 1D grids, vertices
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ )
         writeValue< Index >( format, str, ++offset );

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ ) {
         writeValue< Index >( format, str, i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 2D grids, cells
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
            offset += 4;
            writeValue< Index >( format, str, offset );
         }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
            writeValue< Index >( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
            writeValue< Index >( format, str, j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
            writeValue< Index >( format, str, ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
            writeValue< Index >( format, str, ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
            if( format == VTK::FileFormat::ascii )
               str << "\n";
         }
   }
};

// 2D grids, faces
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ ) {
            offset += 2;
            writeValue< Index >( format, str, offset );
         }

      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
            offset += 2;
            writeValue< Index >( format, str, offset );
         }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ ) {
            writeValue< Index >( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
            writeValue< Index >( format, str, ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
            if( format == VTK::FileFormat::ascii )
               str << "\n";
         }

      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
         for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
            writeValue< Index >( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
            writeValue< Index >( format, str, j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
            if( format == VTK::FileFormat::ascii )
               str << "\n";
         }
   }
};

// 2D grids, vertices
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
            writeValue< Index >( format, str, ++offset );

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
         for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ ) {
            writeValue< Index >( format, str, j * mesh.getDimensions().x() + i );
            if( format == VTK::FileFormat::ascii )
               str << "\n";
         }
   }
};

// 3D grids, cells
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               offset += 8;
               writeValue< Index >( format, str, offset );
            }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }
   }
};

// 3D grids, faces
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               offset += 4;
               writeValue< Index >( format, str, offset );
            }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               offset += 4;
               writeValue< Index >( format, str, offset );
            }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               offset += 4;
               writeValue< Index >( format, str, offset );
            }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }
   }
};

// 3D grids, edges
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               offset += 2;
               writeValue< Index >( format, str, offset );
            }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               offset += 2;
               writeValue< Index >( format, str, offset );
            }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               offset += 2;
               writeValue< Index >( format, str, offset );
            }

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + ( j + 1 ) * ( mesh.getDimensions().x() + 1 ) + i );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
         for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
            for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               writeValue< Index >( format,
                                    str,
                                    ( k + 1 ) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }
   }
};

// 3D grids, vertices
template< typename MeshReal, typename Device, typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   template< typename Index >
   static void
   writeOffsets( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      Index offset = 0;
      writeValue< Index >( format, str, offset );

      for( MeshIndex k = 0; k < ( mesh.getDimensions().z() + 1 ); k++ )
         for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
            for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
               writeValue< Index >( format, str, ++offset );

      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }

   template< typename Index >
   static void
   writeConnectivity( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k < ( mesh.getDimensions().z() + 1 ); k++ )
         for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
            for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ ) {
               writeValue< Index >( format,
                                    str,
                                    k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 )
                                       + j * ( mesh.getDimensions().x() + 1 ) + i );
               if( format == VTK::FileFormat::ascii )
                  str << "\n";
            }
   }
};

// TODO: specialization for disabled entities
template< typename Mesh, int EntityDimension >
struct VTKMeshEntityTypesWriter
{
   static void
   exec( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const int type = (int) VTK::TopologyToEntityShape< typename EntityType::EntityTopology >::shape;
         writeValue( format, str, type );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

template< int Dimension, typename MeshReal, typename Device, typename MeshIndex, int EntityDimension >
struct VTKMeshEntityTypesWriter< Grid< Dimension, MeshReal, Device, MeshIndex >, EntityDimension >
{
   using MeshType = Grid< Dimension, MeshReal, Device, MeshIndex >;

   static void
   exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename MeshType::template EntityType< EntityDimension >;

      const MeshIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( MeshIndex i = 0; i < entitiesCount; i++ ) {
         const int type = (int) VTK::GridEntityShape< EntityType >::shape;
         writeValue( format, str, type );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

}  // namespace detail
}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL
