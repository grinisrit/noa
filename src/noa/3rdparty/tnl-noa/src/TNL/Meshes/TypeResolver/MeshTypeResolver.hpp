// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/MeshTypeResolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/VTKTraits.h>

namespace noa::TNL {
namespace Meshes {

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
bool
MeshTypeResolver< ConfigTag, Device >::run( Reader& reader, Functor&& functor )
{
   return detail< Reader, Functor >::resolveCellTopology( reader, std::forward< Functor >( functor ) );
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveCellTopology( Reader& reader, Functor&& functor )
{
   switch( reader.getCellShape() ) {
      case VTK::EntityShape::Line:
         return resolveSpaceDimension< Topologies::Edge >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Triangle:
         return resolveSpaceDimension< Topologies::Triangle >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Quad:
         return resolveSpaceDimension< Topologies::Quadrangle >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Tetra:
         return resolveSpaceDimension< Topologies::Tetrahedron >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Hexahedron:
         return resolveSpaceDimension< Topologies::Hexahedron >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Polygon:
         return resolveSpaceDimension< Topologies::Polygon >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Wedge:
         return resolveSpaceDimension< Topologies::Wedge >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Pyramid:
         return resolveSpaceDimension< Topologies::Pyramid >( reader, std::forward< Functor >( functor ) );
      case VTK::EntityShape::Polyhedron:
         return resolveSpaceDimension< Topologies::Polyhedron >( reader, std::forward< Functor >( functor ) );
      default:
         std::cerr << "unsupported cell topology: " << VTK::getShapeName( reader.getCellShape() ) << std::endl;
         return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveSpaceDimension( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshCellTopologyTag< ConfigTag, CellTopology >::enabled ) {
      switch( reader.getSpaceDimension() ) {
         case 1:
            return resolveReal< CellTopology, 1 >( reader, std::forward< Functor >( functor ) );
         case 2:
            return resolveReal< CellTopology, 2 >( reader, std::forward< Functor >( functor ) );
         case 3:
            return resolveReal< CellTopology, 3 >( reader, std::forward< Functor >( functor ) );
         default:
            std::cerr << "unsupported space dimension: " << reader.getSpaceDimension() << std::endl;
            return false;
      }
   }
   else {
      std::cerr << "The cell topology " << getType< CellTopology >() << " is disabled in the build configuration." << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveReal( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshSpaceDimensionTag< ConfigTag, CellTopology, SpaceDimension >::enabled ) {
      if( reader.getRealType() == "float" )
         return resolveGlobalIndex< CellTopology, SpaceDimension, float >( reader, std::forward< Functor >( functor ) );
      if( reader.getRealType() == "double" )
         return resolveGlobalIndex< CellTopology, SpaceDimension, double >( reader, std::forward< Functor >( functor ) );
      if( reader.getRealType() == "long double" )
         return resolveGlobalIndex< CellTopology, SpaceDimension, long double >( reader, std::forward< Functor >( functor ) );
      std::cerr << "Unsupported real type: " << reader.getRealType() << std::endl;
      return false;
   }
   else {
      std::cerr << "The combination of space dimension (" << SpaceDimension << ") and mesh dimension ("
                << CellTopology::dimension << ") is either invalid or disabled in the build configuration." << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension, typename Real >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveGlobalIndex( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshRealTag< ConfigTag, Real >::enabled ) {
      if( reader.getGlobalIndexType() == "std::int16_t" || reader.getGlobalIndexType() == "std::uint16_t" )
         return resolveLocalIndex< CellTopology, SpaceDimension, Real, std::int16_t >( reader,
                                                                                       std::forward< Functor >( functor ) );
      if( reader.getGlobalIndexType() == "std::int32_t" || reader.getGlobalIndexType() == "std::uint32_t" )
         return resolveLocalIndex< CellTopology, SpaceDimension, Real, std::int32_t >( reader,
                                                                                       std::forward< Functor >( functor ) );
      if( reader.getGlobalIndexType() == "std::int64_t" || reader.getGlobalIndexType() == "std::uint64_t" )
         return resolveLocalIndex< CellTopology, SpaceDimension, Real, std::int64_t >( reader,
                                                                                       std::forward< Functor >( functor ) );
      std::cerr << "Unsupported global index type: " << reader.getGlobalIndexType() << std::endl;
      return false;
   }
   else {
      std::cerr << "The mesh real type " << getType< Real >() << " is disabled in the build configuration." << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveLocalIndex( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshGlobalIndexTag< ConfigTag, GlobalIndex >::enabled ) {
      if( reader.getLocalIndexType() == "std::int16_t" || reader.getLocalIndexType() == "std::uint16_t" )
         return resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, std::int16_t >(
            reader, std::forward< Functor >( functor ) );
      if( reader.getLocalIndexType() == "std::int32_t" || reader.getLocalIndexType() == "std::uint32_t" )
         return resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, std::int32_t >(
            reader, std::forward< Functor >( functor ) );
      if( reader.getLocalIndexType() == "std::int64_t" || reader.getLocalIndexType() == "std::uint64_t" )
         return resolveMeshType< CellTopology, SpaceDimension, Real, GlobalIndex, std::int64_t >(
            reader, std::forward< Functor >( functor ) );
      std::cerr << "Unsupported local index type: " << reader.getLocalIndexType() << std::endl;
      return false;
   }
   else {
      std::cerr << "The mesh global index type " << getType< GlobalIndex >() << " is disabled in the build configuration."
                << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex, typename LocalIndex >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveMeshType( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshLocalIndexTag< ConfigTag, LocalIndex >::enabled ) {
      using MeshConfig = typename BuildConfigTags::MeshConfigTemplateTag<
         ConfigTag >::template MeshConfig< CellTopology, SpaceDimension, Real, GlobalIndex, LocalIndex >;
      return resolveTerminate< MeshConfig >( reader, std::forward< Functor >( functor ) );
   }
   else {
      std::cerr << "The mesh local index type " << getType< LocalIndex >() << " is disabled in the build configuration."
                << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename MeshConfig >
bool
MeshTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveTerminate( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::MeshDeviceTag< ConfigTag, Device >::enabled
                 && BuildConfigTags::MeshTag< ConfigTag,
                                              Device,
                                              typename MeshConfig::CellTopology,
                                              MeshConfig::spaceDimension,
                                              typename MeshConfig::RealType,
                                              typename MeshConfig::GlobalIndexType,
                                              typename MeshConfig::LocalIndexType >::enabled )
   {
      using MeshType = Meshes::Mesh< MeshConfig, Device >;
      return std::forward< Functor >( functor )( reader, MeshType{} );
   }
   else {
      std::cerr << "The mesh config type " << getType< MeshConfig >() << " is disabled in the build configuration for device "
                << getType< Device >() << "." << std::endl;
      return false;
   }
}

}  // namespace Meshes
}  // namespace noa::TNL
