// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ConfigValidator.h"
#include "Initializer.h"
#include "Layer.h"
#include "Traits.h"

namespace noaTNL {
namespace Meshes {
namespace EntityTags {

template< typename MeshConfig, typename Device, typename Dimension = DimensionTag< 0 > >
class LayerInheritor
   : public Layer< MeshConfig, Device, Dimension >,
     public LayerInheritor< MeshConfig, Device, typename Dimension::Increment >
{
   using LayerType = Layer< MeshConfig, Device, Dimension >;
   using BaseType = LayerInheritor< MeshConfig, Device, typename Dimension::Increment >;
protected:
   using LayerType::setEntitiesCount;
   using LayerType::getEntityTagsView;
   using LayerType::getEntityTag;
   using LayerType::addEntityTag;
   using LayerType::removeEntityTag;
   using LayerType::isBoundaryEntity;
   using LayerType::isGhostEntity;
   using LayerType::updateEntityTagsLayer;
   using LayerType::getBoundaryIndices;
   using LayerType::getInteriorIndices;
   using LayerType::getGhostEntitiesCount;
   using LayerType::getGhostEntitiesOffset;

   using BaseType::setEntitiesCount;
   using BaseType::getEntityTagsView;
   using BaseType::getEntityTag;
   using BaseType::addEntityTag;
   using BaseType::removeEntityTag;
   using BaseType::isBoundaryEntity;
   using BaseType::isGhostEntity;
   using BaseType::updateEntityTagsLayer;
   using BaseType::getBoundaryIndices;
   using BaseType::getInteriorIndices;
   using BaseType::getGhostEntitiesCount;
   using BaseType::getGhostEntitiesOffset;


   LayerInheritor() = default;

   explicit LayerInheritor( const LayerInheritor& other ) = default;

   LayerInheritor( LayerInheritor&& other ) = default;

   template< typename Device_ >
   LayerInheritor( const LayerInheritor< MeshConfig, Device_, Dimension >& other )
   {
      operator=( other );
   }

   LayerInheritor& operator=( const LayerInheritor& other ) = default;

   LayerInheritor& operator=( LayerInheritor&& other ) = default;

   template< typename Device_ >
   LayerInheritor& operator=( const LayerInheritor< MeshConfig, Device_, Dimension >& other )
   {
      LayerType::operator=( other );
      BaseType::operator=( other );
      return *this;
   }


   void print( std::ostream& str ) const
   {
      LayerType::print( str );
      BaseType::print( str );
   }

   bool operator==( const LayerInheritor& layer ) const
   {
      return LayerType::operator==( layer ) &&
             BaseType::operator==( layer );
   }
};

template< typename MeshConfig, typename Device >
class LayerInheritor< MeshConfig, Device, DimensionTag< MeshConfig::meshDimension + 1 > >
{
protected:
   void setEntitiesCount();
   void getEntityTagsView();
   void getEntityTag() const;
   void addEntityTag();
   void removeEntityTag();
   void isBoundaryEntity() const;
   void isGhostEntity() const;
   void updateEntityTagsLayer();
   void getBoundaryIndices() const;
   void getInteriorIndices() const;
   void getGhostEntitiesCount() const;
   void getGhostEntitiesOffset() const;

   LayerInheritor() = default;
   explicit LayerInheritor( const LayerInheritor& other ) = default;
   LayerInheritor( LayerInheritor&& other ) = default;
   template< typename Device_ >
   LayerInheritor( const LayerInheritor< MeshConfig, Device_, DimensionTag< MeshConfig::meshDimension + 1 > >& other ) {}
   LayerInheritor& operator=( const LayerInheritor& other ) = default;
   LayerInheritor& operator=( LayerInheritor&& other ) = default;
   template< typename Device_ >
   LayerInheritor& operator=( const LayerInheritor< MeshConfig, Device_, DimensionTag< MeshConfig::meshDimension + 1 > >& other ) { return *this; }

   void print( std::ostream& str ) const {}

   bool operator==( const LayerInheritor& layer ) const
   {
      return true;
   }
};

// Note that MeshType is an incomplete type and therefore cannot be used to access
// MeshType::Config etc. at the time of declaration of this class template.
template< typename MeshConfig, typename Device, typename MeshType >
class LayerFamily
   : public ConfigValidator< MeshConfig >,
     public Initializer< MeshConfig, Device, MeshType >,
     public LayerInheritor< MeshConfig, Device >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using EntityTagsArrayType = typename MeshTraitsType::EntityTagsArrayType;
   using TagType = typename MeshTraitsType::EntityTagType;
   using BaseType = LayerInheritor< MeshConfig, Device, DimensionTag< 0 > >;
   template< int Dimension >
   using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;
   template< int Dimension >
   using WeakTrait = WeakStorageTrait< MeshConfig, Device, DimensionTag< Dimension > >;

   friend Initializer< MeshConfig, Device, MeshType >;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

   template< int Dimension >
   typename EntityTagsArrayType::ViewType
   getEntityTagsView()
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getEntityTagsView( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   typename EntityTagsArrayType::ConstViewType
   getEntityTagsView() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getEntityTagsView( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   TagType getEntityTag( const GlobalIndexType& entityIndex ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getEntityTag( DimensionTag< Dimension >(), entityIndex );
   }

   template< int Dimension >
   __cuda_callable__
   void addEntityTag( const GlobalIndexType& entityIndex, TagType tag )
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      BaseType::addEntityTag( DimensionTag< Dimension >(), entityIndex, tag );
   }

   template< int Dimension >
   __cuda_callable__
   void removeEntityTag( const GlobalIndexType& entityIndex, TagType tag )
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      BaseType::removeEntityTag( DimensionTag< Dimension >(), entityIndex, tag );
   }

   template< int Dimension >
   __cuda_callable__
   bool isBoundaryEntity( const GlobalIndexType& entityIndex ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::isBoundaryEntity( DimensionTag< Dimension >(), entityIndex );
   }

   template< int Dimension >
   __cuda_callable__
   bool isGhostEntity( const GlobalIndexType& entityIndex ) const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::isGhostEntity( DimensionTag< Dimension >(), entityIndex );
   }

   template< int Dimension >
   auto getBoundaryIndices() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getBoundaryIndices( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   auto getInteriorIndices() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getInteriorIndices( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getGhostEntitiesCount() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getGhostEntitiesCount( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   __cuda_callable__
   GlobalIndexType getGhostEntitiesOffset() const
   {
      static_assert( WeakTrait< Dimension >::entityTagsEnabled, "You try to access entity tags which are not configured for storage." );
      return BaseType::getGhostEntitiesOffset( DimensionTag< Dimension >() );
   }

   template< int Dimension >
   void updateEntityTagsLayer()
   {
      BaseType::updateEntityTagsLayer( DimensionTag< Dimension >() );
   }

protected:
   template< int Dimension >
   void entityTagsSetEntitiesCount( const GlobalIndexType& entitiesCount )
   {
      BaseType::setEntitiesCount( DimensionTag< Dimension >(), entitiesCount );
   }
};

} // namespace EntityTags
} // namespace Meshes
} // namespace noaTNL
