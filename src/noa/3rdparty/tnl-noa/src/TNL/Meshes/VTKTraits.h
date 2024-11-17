// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>
#include <cstdint>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Edge.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Triangle.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Quadrangle.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Tetrahedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Hexahedron.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polygon.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Wedge.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Pyramid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Topologies/Polyhedron.h>

namespace noa::TNL {
namespace Meshes {
namespace VTK {

// VTK file formats
enum class FileFormat : std::uint8_t
{
   ascii,
   binary,
   zlib_compressed
};

// VTK data types
enum class DataType : std::uint8_t
{
   CellData,
   PointData
};

// VTK entity shapes
enum class EntityShape : std::uint8_t
{
   Vertex = 1,
   PolyVertex = 2,
   Line = 3,
   PolyLine = 4,
   Triangle = 5,
   TriangleStrip = 6,
   Polygon = 7,
   Pixel = 8,
   Quad = 9,
   Tetra = 10,
   Voxel = 11,
   Hexahedron = 12,
   Wedge = 13,
   Pyramid = 14,
   PentagonalPrism = 15,
   HexagonalPrism = 16,
   Polyhedron = 42
};

inline std::string
getShapeName( EntityShape shape )
{
   switch( shape ) {
      case EntityShape::Vertex:
         return "Vertex";
      case EntityShape::PolyVertex:
         return "PolyVertex";
      case EntityShape::Line:
         return "Line";
      case EntityShape::PolyLine:
         return "PolyLine";
      case EntityShape::Triangle:
         return "Triangle";
      case EntityShape::TriangleStrip:
         return "TriangleStrip";
      case EntityShape::Polygon:
         return "Polygon";
      case EntityShape::Pixel:
         return "Pixel";
      case EntityShape::Quad:
         return "Quad";
      case EntityShape::Tetra:
         return "Tetra";
      case EntityShape::Voxel:
         return "Voxel";
      case EntityShape::Hexahedron:
         return "Hexahedron";
      case EntityShape::Wedge:
         return "Wedge";
      case EntityShape::Pyramid:
         return "Pyramid";
      case EntityShape::PentagonalPrism:
         return "PentagonalPrism";
      case EntityShape::HexagonalPrism:
         return "HexagonalPrism";
      case EntityShape::Polyhedron:
         return "Polyhedron";
   }
   return "<unknown entity shape>";
}

inline int
getEntityDimension( EntityShape shape )
{
   switch( shape ) {
      case EntityShape::Vertex:
         return 0;
      case EntityShape::PolyVertex:
         return 0;
      case EntityShape::Line:
         return 1;
      case EntityShape::PolyLine:
         return 1;
      case EntityShape::Triangle:
         return 2;
      case EntityShape::TriangleStrip:
         return 2;
      case EntityShape::Polygon:
         return 2;
      case EntityShape::Pixel:
         return 2;
      case EntityShape::Quad:
         return 2;
      case EntityShape::Tetra:
         return 3;
      case EntityShape::Voxel:
         return 3;
      case EntityShape::Hexahedron:
         return 3;
      case EntityShape::Wedge:
         return 3;
      case EntityShape::Pyramid:
         return 3;
      case EntityShape::PentagonalPrism:
         return 3;
      case EntityShape::HexagonalPrism:
         return 3;
      case EntityShape::Polyhedron:
         return 3;
   }
   // this can actually happen when an invalid uint8_t value is converted to EntityShape
   throw std::runtime_error( "VTK::getEntityDimension: invalid entity shape value " + std::to_string( int( shape ) ) );
}

// static mapping of TNL entity topologies to EntityShape
template< typename Topology >
struct TopologyToEntityShape
{};
template<>
struct TopologyToEntityShape< Topologies::Vertex >
{
   static constexpr EntityShape shape = EntityShape::Vertex;
};
template<>
struct TopologyToEntityShape< Topologies::Edge >
{
   static constexpr EntityShape shape = EntityShape::Line;
};
template<>
struct TopologyToEntityShape< Topologies::Triangle >
{
   static constexpr EntityShape shape = EntityShape::Triangle;
};
template<>
struct TopologyToEntityShape< Topologies::Polygon >
{
   static constexpr EntityShape shape = EntityShape::Polygon;
};
template<>
struct TopologyToEntityShape< Topologies::Quadrangle >
{
   static constexpr EntityShape shape = EntityShape::Quad;
};
template<>
struct TopologyToEntityShape< Topologies::Tetrahedron >
{
   static constexpr EntityShape shape = EntityShape::Tetra;
};
template<>
struct TopologyToEntityShape< Topologies::Hexahedron >
{
   static constexpr EntityShape shape = EntityShape::Hexahedron;
};
template<>
struct TopologyToEntityShape< Topologies::Wedge >
{
   static constexpr EntityShape shape = EntityShape::Wedge;
};
template<>
struct TopologyToEntityShape< Topologies::Pyramid >
{
   static constexpr EntityShape shape = EntityShape::Pyramid;
};
template<>
struct TopologyToEntityShape< Topologies::Polyhedron >
{
   static constexpr EntityShape shape = EntityShape::Polyhedron;
};

// mapping used in VTKWriter
template< typename GridEntity >
struct GridEntityShape
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr EntityShape shape = ( dim == 0 ) ? EntityShape::Vertex
                                      : ( dim == 1 ) ? EntityShape::Line
                                      : ( dim == 2 ) ? EntityShape::Pixel
                                                     : EntityShape::Voxel;
};

// type names used in the VTK library (for the XML formats)
// NOTE: C++ has fixed-width integer types (std::int8_t etc.) but also minimum-width types (int, long int, long long int, etc.)
// which makes it impossible to cover all possibilities with function overloading, because some types may be aliases for others
template< typename T >
std::enable_if_t< std::is_integral_v< T > && std::is_signed_v< T >, std::string >
getTypeName( T )
{
   static_assert( sizeof( T ) == 1 || sizeof( T ) == 2 || sizeof( T ) == 4 || sizeof( T ) == 8 );
   if constexpr( sizeof( T ) == 1 )
      return "Int8";
   if constexpr( sizeof( T ) == 2 )
      return "Int16";
   if constexpr( sizeof( T ) == 4 )
      return "Int32";
   if constexpr( sizeof( T ) == 8 )
      return "Int64";
}

template< typename T >
std::enable_if_t< std::is_integral_v< T > && std::is_unsigned_v< T >, std::string >
getTypeName( T )
{
   static_assert( sizeof( T ) == 1 || sizeof( T ) == 2 || sizeof( T ) == 4 || sizeof( T ) == 8 );
   if constexpr( sizeof( T ) == 1 )
      return "UInt8";
   if constexpr( sizeof( T ) == 2 )
      return "UInt16";
   if constexpr( sizeof( T ) == 4 )
      return "UInt32";
   if constexpr( sizeof( T ) == 8 )
      return "UInt64";
}

inline std::string
getTypeName( float )
{
   return "Float32";
}

inline std::string
getTypeName( double )
{
   return "Float64";
}

/**
 * Ghost points and ghost cells
 *
 * The following bit fields are consistent with the corresponding VTK enums [1], which in turn
 * are consistent with VisIt ghost zones specification [2].
 *
 * - [1]
 * https://github.com/Kitware/VTK/blob/060f626b8df0b8144ec8f10c41f936b712c0330b/Common/DataModel/vtkDataSetAttributes.h#L118-L138
 * - [2] http://www.visitusers.org/index.php?title=Representing_ghost_data
 */
enum class CellGhostTypes : std::uint8_t
{
   DUPLICATECELL = 1,         // the cell is present on multiple processors
   HIGHCONNECTIVITYCELL = 2,  // the cell has more neighbors than in a regular mesh
   LOWCONNECTIVITYCELL = 4,   // the cell has less neighbors than in a regular mesh
   REFINEDCELL = 8,           // other cells are present that refines it
   EXTERIORCELL = 16,         // the cell is on the exterior of the data set
   HIDDENCELL = 32            // the cell is needed to maintain connectivity, but the data values should be ignored
};

enum class PointGhostTypes : std::uint8_t
{
   DUPLICATEPOINT = 1,  // the cell is present on multiple processors
   HIDDENPOINT = 2      // the point is needed to maintain connectivity, but the data values should be ignored
};

/**
 * A DataArray with this name is used in the parallel VTK files to indicate ghost regions.
 * Each value must be assigned according to the bit fields PointGhostTypes or CellGhostType.
 *
 * For details, see https://blog.kitware.com/ghost-and-blanking-visibility-changes/
 */
inline const char*
ghostArrayName()
{
   return "vtkGhostType";
}

}  // namespace VTK
}  // namespace Meshes
}  // namespace noa::TNL
