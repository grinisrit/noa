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

#include <ostream>
#include <noa/3rdparty/TNL/Logger.h>
#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/ConfigValidator.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/layers/StorageLayer.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/layers/EntityTags/LayerFamily.h>

#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace noa::TNL {
/**
 * \brief Namespace for numerical meshes and related objects.
 */
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology_ >
class MeshEntity;

template< typename MeshConfig > class Initializer;
template< typename Mesh > class EntityStorageRebinder;
template< typename Mesh, int Dimension > struct IndexPermutationApplier;


template< typename MeshConfig, typename Device, typename MeshType >
class MeshInitializableBase
{
   public:
      using MeshTraitsType = MeshTraits< MeshConfig, Device >;

      // The points and cellSeeds arrays will be reset when not needed to save memory.
      void init( typename MeshTraitsType::PointArrayType& points,
                 typename MeshTraitsType::FaceSeedMatrixType& faceSeeds,
                 typename MeshTraitsType::CellSeedMatrixType& cellSeeds );
};

// The mesh cannot be initialized on CUDA GPU, so this specialization is empty.
template< typename MeshConfig, typename MeshType >
class MeshInitializableBase< MeshConfig, Devices::Cuda, MeshType >
{
};


template< typename MeshConfig,
          typename Device = Devices::Host >
class Mesh
   : public ConfigValidator< MeshConfig >,
     public MeshInitializableBase< MeshConfig, Device, Mesh< MeshConfig, Device > >,
     public StorageLayerFamily< MeshConfig, Device >,
     public EntityTags::LayerFamily< MeshConfig, Device, Mesh< MeshConfig, Device > >
{
      using StorageBaseType = StorageLayerFamily< MeshConfig, Device >;
      using EntityTagsLayerFamily = EntityTags::LayerFamily< MeshConfig, Device, Mesh >;

   public:
      using Config          = MeshConfig;
      using MeshTraitsType  = MeshTraits< MeshConfig, Device >;
      using DeviceType      = typename MeshTraitsType::DeviceType;
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using PointType       = typename MeshTraitsType::PointType;
      using RealType        = typename PointType::RealType;
      using GlobalIndexArray = Containers::Array< GlobalIndexType, DeviceType, GlobalIndexType >;

      template< int Dimension >
      using EntityTraits = typename MeshTraitsType::template EntityTraits< Dimension >;

      template< int Dimension >
      using EntityType = typename EntityTraits< Dimension >::EntityType;

      // constructors
      Mesh() = default;

      Mesh( const Mesh& mesh ) = default;

      Mesh( Mesh&& mesh ) = default;

      template< typename Device_ >
      Mesh( const Mesh< MeshConfig, Device_ >& mesh );

      Mesh& operator=( const Mesh& mesh ) = default;

      Mesh& operator=( Mesh&& mesh ) = default;

      template< typename Device_ >
      Mesh& operator=( const Mesh< MeshConfig, Device_ >& mesh );


      static constexpr int getMeshDimension();

      // types of common entities
      using Cell = EntityType< getMeshDimension() >;
      using Face = EntityType< getMeshDimension() - 1 >;
      using Vertex = EntityType< 0 >;

      /**
       * \brief Returns the count of mesh entities of the given dimension.
       */
      template< int Dimension >
      __cuda_callable__
      GlobalIndexType getEntitiesCount() const;

      /**
       * \brief Returns the mesh entity of the given dimension and index.
       *
       * Note that objects representing mesh entities are not stored in the mesh,
       * but created on demand. Since the \ref MeshEntity contains just a pointer
       * to the mesh and the supplied entity index, the creation should be fast.
       */
      template< int Dimension >
      __cuda_callable__
      EntityType< Dimension > getEntity( const GlobalIndexType entityIndex ) const;

      template< int Dimension >
      void setEntitiesCount( const typename MeshTraitsType::GlobalIndexType& entitiesCount );

      // duplicated for compatibility with grids
      template< typename EntityType >
      __cuda_callable__
      GlobalIndexType getEntitiesCount() const;

      template< typename EntityType >
      __cuda_callable__
      EntityType getEntity( const GlobalIndexType entityIndex ) const;

      /**
       * \brief Returns the spatial coordinates of the vertex with given index.
       */
      const typename MeshTraitsType::PointArrayType& getPoints() const;

      typename MeshTraitsType::PointArrayType& getPoints();

      __cuda_callable__
      const PointType& getPoint( const GlobalIndexType vertexIndex ) const;

      /**
       * \brief Returns the spatial coordinates of the vertex with given index.
       */
      __cuda_callable__
      PointType& getPoint( const GlobalIndexType vertexIndex );

      /**
       * \brief Returns the count of subentities of the entity with given index.
       */
      template< int EntityDimension, int SubentityDimension >
      __cuda_callable__
      constexpr LocalIndexType getSubentitiesCount( const GlobalIndexType entityIndex ) const;

      /**
       * \brief Returns the global index of the subentity specified by its local index.
       */
      template< int EntityDimension, int SubentityDimension >
      __cuda_callable__
      GlobalIndexType getSubentityIndex( const GlobalIndexType entityIndex, const LocalIndexType subentityIndex ) const;

      /**
       * \brief Returns the count of superentities of the entity with given index.
       */
      template< int EntityDimension, int SuperentityDimension >
      __cuda_callable__
      LocalIndexType getSuperentitiesCount( const GlobalIndexType entityIndex ) const;

      /**
       * \brief Returns the global index of the superentity specified by its local index.
       */
      template< int EntityDimension, int SuperentityDimension >
      __cuda_callable__
      GlobalIndexType getSuperentityIndex( const GlobalIndexType entityIndex, const LocalIndexType superentityIndex ) const;

      /**
       * \brief Returns the count of neighbor cells of the cell with given index,
       * based on the information stored in the dual graph.
       */
      __cuda_callable__
      LocalIndexType getCellNeighborsCount( const GlobalIndexType cellIndex ) const;

      /**
       * \brief Returns the global index of the cell's specific neighbor cell wigh given local index,
       * based on the information stored in the dual graph.
       */
      __cuda_callable__
      GlobalIndexType getCellNeighborIndex( const GlobalIndexType cellIndex, const LocalIndexType neighborIndex ) const;


      /**
       * \brief Execute function \e f in parallel for all mesh entities with dimension \e EntityDimension.
       *
       * The function \e f is executed as `f(i)`, where `GlobalIndexType i` is the global index of the
       * mesh entity to be processed. The mesh itself is not passed to the function `f`, it is the user's
       * responsibility to ensure proper access to the mesh if needed, e.g. by the means of lambda capture
       * and/or using a \ref noa::TNL::Pointers::SharedPointer "SharedPointer".
       */
      template< int EntityDimension, typename Device2 = DeviceType, typename Func >
      void forAll( Func f ) const;

      /**
       * \brief Execute function \e f in parallel for all boundary mesh entities with dimension \e EntityDimension.
       *
       * The function \e f is executed as `f(i)`, where `GlobalIndexType i` is the global index of the
       * mesh entity to be processed. The mesh itself is not passed to the function `f`, it is the user's
       * responsibility to ensure proper access to the mesh if needed, e.g. by the means of lambda capture
       * and/or using a \ref noa::TNL::Pointers::SharedPointer "SharedPointer".
       */
      template< int EntityDimension, typename Device2 = DeviceType, typename Func >
      void forBoundary( Func f ) const;

      /**
       * \brief Execute function \e f in parallel for all interior mesh entities with dimension \e EntityDimension.
       *
       * The function \e f is executed as `f(i)`, where `GlobalIndexType i` is the global index of the
       * mesh entity to be processed. The mesh itself is not passed to the function `f`, it is the user's
       * responsibility to ensure proper access to the mesh if needed, e.g. by the means of lambda capture
       * and/or using a \ref noa::TNL::Pointers::SharedPointer "SharedPointer".
       */
      template< int EntityDimension, typename Device2 = DeviceType, typename Func >
      void forInterior( Func f ) const;

      /**
       * \brief Execute function \e f in parallel for all local mesh entities with dimension \e EntityDimension.
       *
       * The function \e f is executed as `f(i)`, where `GlobalIndexType i` is the global index of the
       * mesh entity to be processed. The mesh itself is not passed to the function `f`, it is the user's
       * responsibility to ensure proper access to the mesh if needed, e.g. by the means of lambda capture
       * and/or using a \ref noa::TNL::Pointers::SharedPointer "SharedPointer".
       */
      template< int EntityDimension, typename Device2 = DeviceType, typename Func >
      void forLocal( Func f ) const;

      /**
       * \brief Execute function \e f in parallel for all ghost mesh entities with dimension \e EntityDimension.
       *
       * The function \e f is executed as `f(i)`, where `GlobalIndexType i` is the global index of the
       * mesh entity to be processed. The mesh itself is not passed to the function `f`, it is the user's
       * responsibility to ensure proper access to the mesh if needed, e.g. by the means of lambda capture
       * and/or using a \ref noa::TNL::Pointers::SharedPointer "SharedPointer".
       */
      template< int EntityDimension, typename Device2 = DeviceType, typename Func >
      void forGhost( Func f ) const;


      /**
       * \brief Reorders the entities of the given dimension.
       *
       * The permutations follow the definition used in the Metis library: Let M
       * be the original mesh and M' the permuted mesh. Then entity with index i
       * in M' is the entity with index perm[i] in M and entity with index j in
       * M is the entity with index iperm[j] in M'.
       */
      template< int Dimension >
      void reorderEntities( const GlobalIndexArray& perm,
                            const GlobalIndexArray& iperm );


      void print( std::ostream& str ) const;

      bool operator==( const Mesh& mesh ) const;

      bool operator!=( const Mesh& mesh ) const;

      void writeProlog( Logger& logger ) const;

   protected:
      typename MeshTraitsType::PointArrayType points;

      friend Initializer< MeshConfig >;

      template< typename Mesh, int Dimension >
      friend struct IndexPermutationApplier;

      template< int EntityDimension, int SubentityDimension >
      void setSubentitiesCounts( const typename MeshTraitsType::NeighborCountsArray& counts );
};

template< typename MeshConfig, typename Device >
std::ostream& operator<<( std::ostream& str, const Mesh< MeshConfig, Device >& mesh );

} // namespace Meshes
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Meshes/MeshEntity.h>

#include <noa/3rdparty/TNL/Meshes/Mesh.hpp>
