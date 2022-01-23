// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <noa/3rdparty/TNL/Meshes/MeshEntity.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
MeshEntity< MeshConfig, Device, EntityTopology >::
MeshEntity( const MeshType& mesh, const GlobalIndexType index )
: meshPointer( &mesh ),
  index( index )
{
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
bool
MeshEntity< MeshConfig, Device, EntityTopology >::
operator==( const MeshEntity& entity ) const
{
   return getIndex() == entity.getIndex();
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
bool
MeshEntity< MeshConfig, Device, EntityTopology >::
operator!=( const MeshEntity& entity ) const
{
   return ! ( *this == entity );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
constexpr int
MeshEntity< MeshConfig, Device, EntityTopology >::
getEntityDimension()
{
   return EntityTopology::dimension;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
const Mesh< MeshConfig, Device >&
MeshEntity< MeshConfig, Device, EntityTopology >::
getMesh() const
{
   return *meshPointer;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
typename Mesh< MeshConfig, Device >::GlobalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getIndex() const
{
   return index;
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::PointType
MeshEntity< MeshConfig, Device, EntityTopology >::
getPoint() const
{
   static_assert( getEntityDimension() == 0, "getPoint() can be used only on vertices" );
   return meshPointer->getPoint( getIndex() );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Subdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSubentitiesCount() const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSubentitiesCount< getEntityDimension(), Subdimension >( this->getIndex() );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Subdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSubentityIndex( const LocalIndexType localIndex ) const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSubentityIndex< getEntityDimension(), Subdimension >( this->getIndex(), localIndex );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Superdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::LocalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSuperentitiesCount() const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSuperentitiesCount< getEntityDimension(), Superdimension >( this->getIndex() );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
   template< int Superdimension >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::GlobalIndexType
MeshEntity< MeshConfig, Device, EntityTopology >::
getSuperentityIndex( const LocalIndexType localIndex ) const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getSuperentityIndex< getEntityDimension(), Superdimension >( this->getIndex(), localIndex );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
__cuda_callable__
typename MeshEntity< MeshConfig, Device, EntityTopology >::TagType
MeshEntity< MeshConfig, Device, EntityTopology >::
getTag() const
{
   TNL_ASSERT_TRUE( meshPointer, "meshPointer was not set" );
   return meshPointer->template getEntityTag< getEntityDimension() >( this->getIndex() );
}

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, Device, EntityTopology >& entity )
{
   return str << getType< decltype(entity) >() << "( <meshPointer>, " << entity.getIndex() << " )";
}

} // namespace Meshes
} // namespace noa::TNL
