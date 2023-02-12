// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Writers/GnuplotWriter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Traits.h>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh >
GnuplotWriter< Mesh >::GnuplotWriter( std::ostream& str ) : str( str )
{}

template< typename Mesh >
template< int EntityDimension >
void
GnuplotWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   this->writeHeader( mesh );
}

template< typename Mesh >
template< typename Array >
void
GnuplotWriter< Mesh >::writePointData( const Mesh& mesh, const Array& array, const std::string& name, int numberOfComponents )
{
   using RealType = typename Mesh::RealType;
   using IndexType = typename Array::IndexType;
   IndexType pointsCount = mesh.template getEntitiesCount< 0 >();
   if( array.getSize() / numberOfComponents != typename Array::IndexType( pointsCount ) )
      throw std::length_error( "Mismatched array size for POINT_DATA section: " + std::to_string( array.getSize() )
                               + " (there are " + std::to_string( pointsCount ) + " points in the file)" );

   str << "# " << name << std::endl;

   RealType last_x = std::numeric_limits< RealType >::lowest();
   for( IndexType idx = 0; idx < pointsCount; idx++ ) {
      auto entity = mesh.template getEntity< 0 >( idx );
      auto center = entity.getCenter();
      if( isGrid< Mesh >::value && center.x() < last_x )
         str << std::endl;
      last_x = entity.getCenter().x();
      this->write( str, center );
      str << array.getElement( idx ) << std::endl;
   }
}

template< typename Mesh >
template< typename Array >
void
GnuplotWriter< Mesh >::writeCellData( const Mesh& mesh, const Array& array, const std::string& name, int numberOfComponents )
{
   using RealType = typename Mesh::RealType;
   using IndexType = typename Array::IndexType;
   IndexType cellsCount = mesh.template getEntitiesCount< Mesh::getMeshDimension() >();
   if( array.getSize() / numberOfComponents != typename Array::IndexType( cellsCount ) )
      throw std::length_error( "Mismatched array size for CELL_DATA section: " + std::to_string( array.getSize() )
                               + " (there are " + std::to_string( cellsCount ) + " cells in the file)" );

   str << "# " << name << std::endl;

   RealType last_x = std::numeric_limits< RealType >::lowest();
   for( IndexType idx = 0; idx < cellsCount; idx++ ) {
      auto entity = mesh.template getEntity< Mesh::getMeshDimension() >( idx );
      auto center = entity.getCenter();
      if( isGrid< Mesh >::value && center.x() < last_x )
         str << std::endl;
      last_x = center.x();
      this->write( str, center );
      str << array.getElement( idx ) << std::endl;
   }
}

template< typename Mesh >
void
GnuplotWriter< Mesh >::writeHeader( const Mesh& mesh )
{
   str << "# File generater by TNL" << std::endl;
}

template< typename Mesh >
template< typename Element >
void
GnuplotWriter< Mesh >::write( std::ostream& str, const Element& d )
{
   str << d;
}

template< typename Mesh >
template< typename Real >
void
GnuplotWriter< Mesh >::write( std::ostream& str, const Containers::StaticVector< 1, Real >& d )
{
   str << d.x() << " ";
}

template< typename Mesh >
template< typename Real >
void
GnuplotWriter< Mesh >::write( std::ostream& str, const Containers::StaticVector< 2, Real >& d )
{
   str << d.x() << " " << d.y() << " ";
}

template< typename Mesh >
template< typename Real >
void
GnuplotWriter< Mesh >::write( std::ostream& str, const Containers::StaticVector< 3, Real >& d )
{
   str << d.x() << " " << d.y() << " " << d.z() << " ";
}

}  // namespace Writers
}  // namespace Meshes
}  // namespace noa::TNL
