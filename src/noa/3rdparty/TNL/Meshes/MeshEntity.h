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

#include <TNL/Meshes/Mesh.h>

namespace TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology_ >
class MeshEntity
{
   static_assert( std::is_same< EntityTopology_, typename Mesh< MeshConfig, Device >::template EntityTraits< EntityTopology_::dimension >::EntityTopology >::value,
                  "Specified entity topology is not compatible with the MeshConfig." );

   public:
      using MeshType        = Mesh< MeshConfig, Device >;
      using DeviceType      = Device;
      using EntityTopology  = EntityTopology_;
      using GlobalIndexType = typename MeshType::GlobalIndexType;
      using LocalIndexType  = typename MeshType::LocalIndexType;
      using PointType       = typename MeshType::PointType;
      using TagType         = typename MeshType::MeshTraitsType::EntityTagType;

      template< int Subdimension >
      using SubentityTraits = typename MeshType::MeshTraitsType::template SubentityTraits< EntityTopology, Subdimension >;

      template< int Superdimension >
      using SuperentityTraits = typename MeshType::MeshTraitsType::template SuperentityTraits< EntityTopology, Superdimension >;

      // constructors
      MeshEntity() = delete;

      __cuda_callable__
      MeshEntity( const MeshType& mesh, const GlobalIndexType index );

      __cuda_callable__
      MeshEntity( const MeshEntity& entity ) = default;

      __cuda_callable__
      MeshEntity& operator=( const MeshEntity& entity ) = default;

      __cuda_callable__
      bool operator==( const MeshEntity& entity ) const;

      __cuda_callable__
      bool operator!=( const MeshEntity& entity ) const;

      /**
       * \brief Returns the dimension of this mesh entity.
       */
      static constexpr int getEntityDimension();

      /**
       * \brief Returns a reference to the mesh that owns this mesh entity.
       */
      __cuda_callable__
      const MeshType& getMesh() const;

      /**
       * \brief Returns the index of this mesh entity.
       */
      __cuda_callable__
      GlobalIndexType getIndex() const;

      /**
       * \brief Returns the spatial coordinates of this vertex.
       *
       * Can be used only when \ref getEntityDimension returns 0.
       */
      __cuda_callable__
      PointType getPoint() const;

      /**
       * \brief Returns the count of subentities of this entity.
       */
      template< int Subdimension >
      __cuda_callable__
      LocalIndexType getSubentitiesCount() const;

      /**
       * \brief Returns the global index of the subentity specified by its local index.
       */
      template< int Subdimension >
      __cuda_callable__
      GlobalIndexType getSubentityIndex( const LocalIndexType localIndex ) const;

      /**
       * \brief Returns the count of superentities of this entity.
       */
      template< int Superdimension >
      __cuda_callable__
      LocalIndexType getSuperentitiesCount() const;

      /**
       * \brief Returns the global index of the superentity specified by its local index.
       */
      template< int Superdimension >
      __cuda_callable__
      GlobalIndexType getSuperentityIndex( const LocalIndexType localIndex ) const;

      /**
       * \brief Returns the tag associated with this entity.
       */
      __cuda_callable__
      TagType getTag() const;

   protected:
      const MeshType* meshPointer = nullptr;
      GlobalIndexType index = 0;
};

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const MeshEntity< MeshConfig, Device, EntityTopology >& entity );

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/MeshEntity.hpp>
