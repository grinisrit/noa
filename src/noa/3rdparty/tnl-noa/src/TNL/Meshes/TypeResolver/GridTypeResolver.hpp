// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/GridTypeResolver.h>

namespace noa::TNL {
namespace Meshes {

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
bool
GridTypeResolver< ConfigTag, Device >::run( Reader& reader, Functor&& functor )
{
   return detail< Reader, Functor >::resolveGridDimension( reader, std::forward< Functor >( functor ) );
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveGridDimension( Reader& reader, Functor&& functor )
{
   if( reader.getMeshDimension() == 1 )
      return resolveReal< 1 >( reader, std::forward< Functor >( functor ) );
   if( reader.getMeshDimension() == 2 )
      return resolveReal< 2 >( reader, std::forward< Functor >( functor ) );
   if( reader.getMeshDimension() == 3 )
      return resolveReal< 3 >( reader, std::forward< Functor >( functor ) );
   std::cerr << "Unsupported mesh dimension: " << reader.getMeshDimension() << std::endl;
   return false;
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< int MeshDimension >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveReal( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridDimensionTag< ConfigTag, MeshDimension >::enabled ) {
      if( reader.getRealType() == "float" )
         return resolveIndex< MeshDimension, float >( reader, std::forward< Functor >( functor ) );
      if( reader.getRealType() == "double" )
         return resolveIndex< MeshDimension, double >( reader, std::forward< Functor >( functor ) );
      if( reader.getRealType() == "long double" )
         return resolveIndex< MeshDimension, long double >( reader, std::forward< Functor >( functor ) );
      std::cerr << "Unsupported real type: " << reader.getRealType() << std::endl;
      return false;
   }
   else {
      std::cerr << "The grid dimension " << MeshDimension << " is disabled in the build configuration." << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< int MeshDimension, typename Real >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveIndex( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridRealTag< ConfigTag, Real >::enabled ) {
      if( reader.getGlobalIndexType() == "std::int16_t" || reader.getGlobalIndexType() == "std::uint16_t" )
         return resolveGridType< MeshDimension, Real, std::int16_t >( reader, std::forward< Functor >( functor ) );
      if( reader.getGlobalIndexType() == "std::int32_t" || reader.getGlobalIndexType() == "std::uint32_t" )
         return resolveGridType< MeshDimension, Real, std::int32_t >( reader, std::forward< Functor >( functor ) );
      if( reader.getGlobalIndexType() == "std::int64_t" || reader.getGlobalIndexType() == "std::uint64_t" )
         return resolveGridType< MeshDimension, Real, std::int64_t >( reader, std::forward< Functor >( functor ) );
      std::cerr << "Unsupported index type: " << reader.getRealType() << std::endl;
      return false;
   }
   else {
      std::cerr << "The grid real type " << getType< Real >() << " is disabled in the build configuration." << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< int MeshDimension, typename Real, typename Index >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveGridType( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridIndexTag< ConfigTag, Index >::enabled ) {
      using GridType = Meshes::Grid< MeshDimension, Real, Device, Index >;
      return resolveTerminate< GridType >( reader, std::forward< Functor >( functor ) );
   }
   else {
      std::cerr << "The grid index type " << getType< Index >() << " is disabled in the build configuration." << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device >
template< typename Reader, typename Functor >
template< typename GridType >
bool
GridTypeResolver< ConfigTag, Device >::detail< Reader, Functor >::resolveTerminate( Reader& reader, Functor&& functor )
{
   if constexpr( BuildConfigTags::GridTag< ConfigTag, GridType >::enabled ) {
      return std::forward< Functor >( functor )( reader, GridType{} );
   }
   else {
      std::cerr << "The mesh type " << TNL::getType< GridType >() << " is disabled in the build configuration." << std::endl;
      return false;
   }
}

}  // namespace Meshes
}  // namespace noa::TNL
