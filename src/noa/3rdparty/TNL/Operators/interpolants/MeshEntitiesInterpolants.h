// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noaTNL {
namespace Operators {   

template< typename Mesh,
          int InEntityDimension,
          int OutEntityDimenions >
class MeshEntitiesInterpolants
{
};

/***
 * 1D grid mesh entity interpolation: 1 -> 0
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 1, Real, Device, Index >, 1, 0 >
   : public Functions::Domain< 1, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 1, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimension() == 1,
            "Mesh function must be defined on cells." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities();
 
         return 0.5 * ( u[ neighborEntities.template getEntityIndex< -1 >() ] +
                        u[ neighborEntities.template getEntityIndex<  1 >() ] );
      }
};

/***
 * 1D grid mesh entity interpolation: 0 -> 1
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 1, Real, Device, Index >, 0, 1 >
   : public Functions::Domain< 1, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 1, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntitiesDimension() == 0,
            "Mesh function must be defined on vertices (or faces in case on 1D grid)." );
 
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 0 >& neighborEntities = entity.template getNeighborEntities< 0 >();
 
         return 0.5 * ( u[ neighborEntities.template getEntityIndex< -1 >() ] +
                        u[ neighborEntities.template getEntityIndex<  1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 2 -> 1
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 2, Real, Device, Index >, 2, 1 >
   : public Functions::Domain< 2, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > MeshType;
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimension() == 2,
            "Mesh function must be defined on cells." );
 
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();
 
         if( entity.getOrientation().x() == 1.0 )
            return 0.5 * ( u[ neighborEntities.template getEntityIndex< -1, 0 >() ] +
                           u[ neighborEntities.template getEntityIndex<  1, 0 >() ] );
         else
            return 0.5 * ( u[ neighborEntities.template getEntityIndex< 0, -1 >() ] +
                           u[ neighborEntities.template getEntityIndex< 0,  1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 2 -> 0
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 2, Real, Device, Index >, 2, 0 >
   : public Functions::Domain< 2, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimension() == 2,
            "Mesh function must be defined on cells." );
 
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities();
 
         return 0.25 * ( u[ neighborEntities.template getEntityIndex< -1,  1 >() ] +
                         u[ neighborEntities.template getEntityIndex<  1,  1 >() ] +
                         u[ neighborEntities.template getEntityIndex< -1, -1 >() ] +
                         u[ neighborEntities.template getEntityIndex<  1, -1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 1 -> 2
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 2, Real, Device, Index >, 1, 2 >
   : public Functions::Domain< 2, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntitiesDimension() == 1,
            "Mesh function must be defined on faces." );
 
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.template getNeighborEntities< 1 >();
 
         return 0.25 * ( u[ neighborEntities.template getEntityIndex< -1,  0 >() ] +
                         u[ neighborEntities.template getEntityIndex<  1,  0 >() ] +
                         u[ neighborEntities.template getEntityIndex<  0,  1 >() ] +
                         u[ neighborEntities.template getEntityIndex<  0, -1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 0 -> 2
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 2, Real, Device, Index >, 0, 2 >
   : public Functions::Domain< 2, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 2, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimension() == 1,
            "Mesh function must be defined on vertices." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 0 >& neighborEntities = entity.getNeighborEntities();
 
         return 0.25 * ( u[ neighborEntities.template getEntityIndex< -1,  1 >() ] +
                         u[ neighborEntities.template getEntityIndex<  1,  1 >() ] +
                         u[ neighborEntities.template getEntityIndex< -1, -1 >() ] +
                         u[ neighborEntities.template getEntityIndex<  1, -1 >() ] );
      }
};

/***
 * 3D grid mesh entity interpolation: 3 -> 2
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 3, Real, Device, Index >, 3, 2 >
   : public Functions::Domain< 3, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimension() == 3,
            "Mesh function must be defined on cells." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities();
 
         if( entity.getOrientation().x() == 1.0 )
            return 0.5 * ( u[ neighborEntities.template getEntityIndex< -1,  0,  0 >() ] +
                           u[ neighborEntities.template getEntityIndex<  1,  0,  0 >() ] );
         if( entity.getOrientation().y() == 1.0 )
            return 0.5 * ( u[ neighborEntities.template getEntityIndex<  0, -1,  0 >() ] +
                           u[ neighborEntities.template getEntityIndex<  0,  1,  0 >() ] );
         else
            return 0.5 * ( u[ neighborEntities.template getEntityIndex<  0,  0, -1 >() ] +
                           u[ neighborEntities.template getEntityIndex<  0,  0,  1 >() ] );
      }
};

/***
 * 3D grid mesh entity interpolation: 2 -> 3
 */
template< typename Real,
          typename Device,
          typename Index >
class MeshEntitiesInterpolants< Meshes::Grid< 3, Real, Device, Index >, 2, 3 >
   : public Functions::Domain< 3, Functions::MeshInteriorDomain >
{
   public:
 
      typedef Meshes::Grid< 3, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntitiesDimension() == 2,
            "Mesh function must be defined on faces." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.template getNeighborEntities< 2 >();
 
         return 1.0 / 6.0 * ( u[ neighborEntities.template getEntityIndex< -1,  0,  0 >() ] +
                              u[ neighborEntities.template getEntityIndex<  1,  0,  0 >() ] +
                              u[ neighborEntities.template getEntityIndex<  0, -1,  0 >() ] +
                              u[ neighborEntities.template getEntityIndex<  0,  1,  0 >() ] +
                              u[ neighborEntities.template getEntityIndex<  0,  0, -1 >() ] +
                              u[ neighborEntities.template getEntityIndex<  0,  0,  1 >() ] );
      }
};

} // namespace Operators
} // namespace noaTNL

