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

#include <noa/3rdparty/TNL/File.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/layers/SubentityStorageLayer.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/layers/SuperentityStorageLayer.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/layers/DualGraphLayer.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename DimensionTag,
          bool EntityStorage = (DimensionTag::value <= MeshConfig::meshDimension) >
class StorageLayer;


template< typename MeshConfig, typename Device >
class StorageLayerFamily
   : public StorageLayer< MeshConfig, Device, DimensionTag< 0 > >,
     public DualGraphLayer< MeshConfig, Device >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using BaseType       = StorageLayer< MeshConfig, Device, DimensionTag< 0 > >;
   template< int Dimension >
   using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;

   template< int Dimension, int Subdimension >
   using SubentityTraits = typename MeshTraitsType::template SubentityTraits< typename EntityTraits< Dimension >::EntityTopology, Subdimension >;

   template< int Dimension, int Superdimension >
   using SuperentityTraits = typename MeshTraitsType::template SuperentityTraits< typename EntityTraits< Dimension >::EntityTopology, Superdimension >;

public:
   StorageLayerFamily() = default;

   explicit StorageLayerFamily( const StorageLayerFamily& other ) = default;

   StorageLayerFamily( StorageLayerFamily&& other ) = default;

   template< typename Device_ >
   StorageLayerFamily( const StorageLayerFamily< MeshConfig, Device_ >& other )
   {
      operator=( other );
   }

   StorageLayerFamily& operator=( const StorageLayerFamily& layer ) = default;

   StorageLayerFamily& operator=( StorageLayerFamily&& layer ) = default;

   template< typename Device_ >
   StorageLayerFamily& operator=( const StorageLayerFamily< MeshConfig, Device_ >& layer )
   {
      BaseType::operator=( layer );
      DualGraphLayer< MeshConfig, Device >::operator=( layer );
      return *this;
   }

   bool operator==( const StorageLayerFamily& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               DualGraphLayer< MeshConfig, Device >::operator==( layer ) );
   }

   template< int Dimension, int Subdimension >
   void
   setSubentitiesCounts( const typename MeshTraitsType::NeighborCountsArray& counts )
   {
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      static_assert( SubentityTraits< Dimension, Subdimension >::storageEnabled,
                     "You try to set subentitiesCounts for a combination of Dimension and Subdimension which is disabled in the mesh configuration." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
                                                    Device,
                                                    typename EntityTraits< Dimension >::EntityTopology >;
      BaseType::template setSubentitiesCounts< Subdimension >( counts );
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::LocalIndexType
   getSubentitiesCount( const GlobalIndexType entityIndex ) const
   {
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      static_assert( SubentityTraits< Dimension, Subdimension >::storageEnabled,
                     "You try to get subentities count for subentities which are disabled in the mesh configuration." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
                                                    Device,
                                                    typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSubentitiesCount< Subdimension >( entityIndex );
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::template SubentityMatrixType< Dimension >&
   getSubentitiesMatrix()
   {
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      static_assert( SubentityTraits< Dimension, Subdimension >::storageEnabled,
                     "You try to get subentities matrix which is disabled in the mesh configuration." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
                                                    Device,
                                                    typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSubentitiesMatrix< Subdimension >();
   }

   template< int Dimension, int Subdimension >
   __cuda_callable__
   const typename MeshTraitsType::template SubentityMatrixType< Dimension >&
   getSubentitiesMatrix() const
   {
      static_assert( Dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      static_assert( SubentityTraits< Dimension, Subdimension >::storageEnabled,
                     "You try to get subentities matrix which is disabled in the mesh configuration." );
      using BaseType = SubentityStorageLayerFamily< MeshConfig,
                                                    Device,
                                                    typename EntityTraits< Dimension >::EntityTopology >;
      return BaseType::template getSubentitiesMatrix< Subdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::NeighborCountsArray&
   getSuperentitiesCountsArray()
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities counts array which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     DimensionTag< Dimension > >;
      return BaseType::template getSuperentitiesCountsArray< Superdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   const typename MeshTraitsType::NeighborCountsArray&
   getSuperentitiesCountsArray() const
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities counts array which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     DimensionTag< Dimension > >;
      return BaseType::template getSuperentitiesCountsArray< Superdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   typename MeshTraitsType::SuperentityMatrixType&
   getSuperentitiesMatrix()
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities matrix which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     DimensionTag< Dimension > >;
      return BaseType::template getSuperentitiesMatrix< Superdimension >();
   }

   template< int Dimension, int Superdimension >
   __cuda_callable__
   const typename MeshTraitsType::SuperentityMatrixType&
   getSuperentitiesMatrix() const
   {
      static_assert( Dimension < Superdimension, "Invalid combination of Dimension and Superdimension." );
      static_assert( SuperentityTraits< Dimension, Superdimension >::storageEnabled,
                     "You try to get superentities matrix which is disabled in the mesh configuration." );
      using BaseType = SuperentityStorageLayerFamily< MeshConfig,
                                                     Device,
                                                     DimensionTag< Dimension > >;
      return BaseType::template getSuperentitiesMatrix< Superdimension >();
   }
};


template< typename MeshConfig,
          typename Device,
          typename DimensionTag >
class StorageLayer< MeshConfig,
                    Device,
                    DimensionTag,
                    true >
   : public SubentityStorageLayerFamily< MeshConfig,
                                         Device,
                                         typename MeshTraits< MeshConfig, Device >::template EntityTraits< DimensionTag::value >::EntityTopology >,
     public SuperentityStorageLayerFamily< MeshConfig,
                                           Device,
                                           DimensionTag >,
     public StorageLayer< MeshConfig, Device, typename DimensionTag::Increment >
{
public:
   using BaseType = StorageLayer< MeshConfig, Device, typename DimensionTag::Increment >;
   using MeshTraitsType   = MeshTraits< MeshConfig, Device >;
   using GlobalIndexType  = typename MeshTraitsType::GlobalIndexType;
   using EntityTraitsType = typename MeshTraitsType::template EntityTraits< DimensionTag::value >;
   using EntityType       = typename EntityTraitsType::EntityType;
   using EntityTopology   = typename EntityTraitsType::EntityTopology;
   using SubentityStorageBaseType = SubentityStorageLayerFamily< MeshConfig, Device, EntityTopology >;
   using SuperentityStorageBaseType = SuperentityStorageLayerFamily< MeshConfig, Device, DimensionTag >;

   StorageLayer() = default;

   explicit StorageLayer( const StorageLayer& other ) = default;

   template< typename Device_ >
   StorageLayer( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      operator=( other );
   }

   StorageLayer& operator=( const StorageLayer& other ) = default;

   StorageLayer& operator=( StorageLayer&& other ) = default;

   template< typename Device_ >
   StorageLayer& operator=( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      entitiesCount = other.getEntitiesCount( DimensionTag() );
      SubentityStorageBaseType::operator=( other );
      SuperentityStorageBaseType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }

   void print( std::ostream& str ) const
   {
      str << "Number of entities with dimension " << DimensionTag::value << ": " << entitiesCount << std::endl;
      SubentityStorageBaseType::print( str );
      SuperentityStorageBaseType::print( str );
      str << std::endl;
      BaseType::print( str );
   }

   bool operator==( const StorageLayer& meshLayer ) const
   {
      return ( entitiesCount == meshLayer.entitiesCount &&
               SubentityStorageBaseType::operator==( meshLayer ) &&
               SuperentityStorageBaseType::operator==( meshLayer ) &&
               BaseType::operator==( meshLayer ) );
   }


   using BaseType::getEntitiesCount;
   __cuda_callable__
   GlobalIndexType getEntitiesCount( DimensionTag ) const
   {
      return this->entitiesCount;
   }

protected:
   using BaseType::setEntitiesCount;
   void setEntitiesCount( DimensionTag, const GlobalIndexType& entitiesCount )
   {
      this->entitiesCount = entitiesCount;
   }

   GlobalIndexType entitiesCount = 0;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename DimensionTag_, bool Storage_ >
   friend class StorageLayer;
};

template< typename MeshConfig,
          typename Device >
class StorageLayer< MeshConfig, Device, DimensionTag< MeshConfig::meshDimension + 1 >, false >
{
protected:
   using DimensionTag     = Meshes::DimensionTag< MeshConfig::meshDimension >;
   using GlobalIndexType  = typename MeshConfig::GlobalIndexType;

   StorageLayer() = default;

   explicit StorageLayer( const StorageLayer& other ) = default;

   StorageLayer( StorageLayer&& other ) = default;

   template< typename Device_ >
   StorageLayer( const StorageLayer< MeshConfig, Device_, DimensionTag >& other ) {}

   StorageLayer& operator=( const StorageLayer& other ) = default;

   StorageLayer& operator=( StorageLayer&& other ) = default;

   template< typename Device_ >
   StorageLayer& operator=( const StorageLayer< MeshConfig, Device_, DimensionTag >& other )
   {
      return *this;
   }


   void setEntitiesCount() {}
   void getEntitiesCount() const {}

   void print( std::ostream& str ) const {}

   bool operator==( const StorageLayer& meshLayer ) const
   {
      return true;
   }
};

} // namespace Meshes
} // namespace noa::TNL
