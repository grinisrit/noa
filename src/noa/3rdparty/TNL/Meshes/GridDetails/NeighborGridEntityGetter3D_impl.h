// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
#include <TNL/Algorithms/staticFor.h>

namespace TNL {
namespace Meshes {

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       3         |              3            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config >,
   3,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 3;
      static constexpr int NeighborEntityDimension = 3;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX,
                                                         entity.getCoordinates().y() + stepY,
                                                         entity.getCoordinates().z() + stepZ ) );
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + ( stepZ * entity.getMesh().getDimensions().y() + stepY ) * entity.getMesh().getDimensions().x() + stepX;
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};

};


/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       3         |              3            |       Cross       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config >,
   3,
   GridEntityStencilStorageTag< GridEntityCrossStencil > >
{
   public:

      static constexpr int EntityDimension = 3;
      static constexpr int NeighborEntityDimension = 3;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;
      typedef GridEntityStencilStorageTag< GridEntityCrossStencil > StencilStorage;

      static constexpr int stencilSize = Config::getStencilSize();

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX,
                                                         entity.getCoordinates().y() + stepY,
                                                         entity.getCoordinates().z() + stepZ ) );
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
#ifndef HAVE_CUDA // TODO: fix this to work with CUDA
         if( ( stepX != 0 && stepY != 0 ) ||
             ( stepX != 0 && stepZ != 0 ) ||
             ( stepY != 0 && stepZ != 0 ) ||
             ( stepX < -stencilSize || stepX > stencilSize ||
               stepY < -stencilSize || stepY > stencilSize ||
               stepZ < -stencilSize || stepZ > stencilSize ) )
            return this->entity.getIndex() + ( stepZ * entity.getMesh().getDimensions().y() + stepY ) * entity.getMesh().getDimensions().x() + stepX;
         if( stepY == 0 && stepZ == 0 )
            return stencilX[ stepX + stencilSize ];
         if( stepZ == 0 )
            return stencilY[ stepY + stencilSize ];
         return stencilZ[ stepZ + stencilSize ];
#else
         return this->entity.getIndex() + ( stepZ * entity.getMesh().getDimensions().y() + stepY ) * entity.getMesh().getDimensions().x() + stepX;
#endif

      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex )
      {
#ifndef HAVE_CUDA // TODO: fix this to work with CUDA
         auto stencilXRefresher = [&] ( auto index ) {
            stencilX[ index + stencilSize ] = entityIndex + index;
         };
         auto stencilYRefresher = [&] ( auto index ) {
            stencilY[ index + stencilSize ] =
               entityIndex + index * entity.getMesh().getDimensions().x();
         };
         auto stencilZRefresher = [&] ( auto index ) {
            stencilZ[ index + stencilSize ] =
               entityIndex + index * entity.getMesh().cellZNeighborsStep;
         };
         Algorithms::staticFor< IndexType, -stencilSize, 0 >( stencilZRefresher );
         Algorithms::staticFor< IndexType, 1, stencilSize + 1 >( stencilZRefresher );
         Algorithms::staticFor< IndexType, -stencilSize, 0 >( stencilYRefresher );
         Algorithms::staticFor< IndexType, 1, stencilSize + 1 >( stencilYRefresher );
         Algorithms::staticFor< IndexType, -stencilSize, stencilSize + 1 >( stencilXRefresher );
#endif
      };

   protected:

      const GridEntityType& entity;

      IndexType stencilX[ 2 * stencilSize + 1 ];
      IndexType stencilY[ 2 * stencilSize + 1 ];
      IndexType stencilZ[ 2 * stencilSize + 1 ];

      //NeighborGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       3         |              2            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config >,
   2,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 3;
      static constexpr int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;
      typedef typename GridEntityType::EntityBasisType EntityBasisType;

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         static_assert( ! stepX + ! stepY + ! stepZ == 2, "Only one of the steps can be non-zero." );
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) )
                       < entity.getMesh().getDimensions() +
                        CoordinatesType( ( stepX > 0 ), ( stepY > 0 ), ( stepZ > 0 ) ),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                         entity.getCoordinates().y() + stepY + ( stepY < 0 ),
                                                         entity.getCoordinates().z() + stepZ + ( stepZ < 0 ) ),
                                        EntityOrientationType( stepX ? (stepX > 0 ? 1 : -1) : 0,
                                                               stepY ? (stepY > 0 ? 1 : -1) : 0,
                                                               stepZ ? (stepZ > 0 ? 1 : -1) : 0 ),
                                        EntityBasisType( ! stepX, !stepY, !stepZ ) );
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( this->entity.getMesh(), getEntity< stepX, stepY, stepZ >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};

/****      TODO: Finish it, knonw it is only a copy of specialization for none stored stencil
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       3         |              2            |       Cross       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config >,
   2,
   GridEntityStencilStorageTag< GridEntityCrossStencil > >
{
   public:

      static constexpr int EntityDimension = 3;
      static constexpr int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;
      typedef typename GridEntityType::EntityBasisType EntityBasisType;

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         static_assert( ! stepX + ! stepY + ! stepZ == 2, "Only one of the steps can be non-zero." );
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) )
                       < entity.getMesh().getDimensions() +
                        CoordinatesType( ( stepX > 0 ), ( stepY > 0 ), ( stepZ > 0 ) ),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                         entity.getCoordinates().y() + stepY + ( stepY < 0 ),
                                                         entity.getCoordinates().z() + stepZ + ( stepZ < 0 ) ),
                                        EntityOrientationType( stepX ? (stepX > 0 ? 1 : -1) : 0,
                                                               stepY ? (stepY > 0 ? 1 : -1) : 0,
                                                               stepZ ? (stepZ > 0 ? 1 : -1) : 0 ),
                                        EntityBasisType( ! stepX, !stepY, !stepZ ) );
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( this->entity.getMesh(), getEntity< stepX, stepY, stepZ >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};


/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       3         |              1            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config >,
   1,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 3;
      static constexpr int NeighborEntityDimension = 1;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;
      typedef typename GridEntityType::EntityOrientationType EntityOrientationType;
      typedef typename GridEntityType::EntityBasisType EntityBasisType;

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         static_assert( ! stepX + ! stepY + ! stepZ == 1, "Exactly two of the steps must be non-zero." );
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) )
                       < entity.getMesh().getDimensions() +
                        CoordinatesType( ( stepX > 0 ), ( stepY > 0 ), ( stepZ > 0 ) ),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                         entity.getCoordinates().y() + stepY + ( stepY < 0 ),
                                                         entity.getCoordinates().z() + stepZ + ( stepZ < 0 ) ),
                                        EntityOrientationType( !!stepX, !!stepY, !!stepZ ),
                                        EntityBasisType( !stepX, !stepY, !stepZ ));
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( this->entity.getMesh(), getEntity< stepX, stepY, stepZ >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};


/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       3         |            0              |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 3, Config >,
   0,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 3;
      static constexpr int NeighborEntityDimension = 0;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY,int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT( stepX != 0 && stepY != 0 && stepZ != 0,
                    std::cerr << " stepX = " << stepX
                         << " stepY = " << stepY
                         << " stepZ = " << stepZ );
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ),
                                        stepZ + ( stepZ < 0 ) )
                       < entity.getMesh().getDimensions() +
                            CoordinatesType( ( stepX > 0 ), ( stepY > 0 ), ( stepZ > 0 ) ),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 )  ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ), stepZ + ( stepZ < 0 ) )
                   << " entity.getMesh().getDimensions() + CoordinatesType( sign( stepX ), sign( stepY ), sign( stepZ ) ) = "
                   << entity.getMesh().getDimensions()  + CoordinatesType( sign( stepX ), sign( stepY ), sign( stepZ ) )
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                         entity.getCoordinates().y() + stepY + ( stepY < 0 ),
                                                         entity.getCoordinates().z() + stepZ + ( stepZ < 0 ) ) );
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( entity.getMesh(), getEntity< stepX, stepY, stepZ >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              3            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 2, Config >,
   3,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 2;
      static constexpr int NeighborEntityDimension = 3;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         /*TNL_ASSERT( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
                    ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ) &&
                    ( ( !! stepZ ) == ( !! entity.getOrientation().z() ) ),
                    std::cerr << "( stepX, stepY, stepZ ) cannot be perpendicular to entity coordinates: stepX = " << stepX
                         << " stepY = " << stepY << " stepZ = " << stepZ
                         << " entity.getOrientation() = " << entity.getOrientation() );*/
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions() + TNL::abs(entity.getOrientation()), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX - ( stepX > 0 ) * ( entity.getOrientation().x() != 0.0 ),
                                        stepY - ( stepY > 0 ) * ( entity.getOrientation().y() != 0.0 ),
                                        stepZ - ( stepZ > 0 ) * ( entity.getOrientation().z() != 0.0 ) ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX - ( stepX > 0 ) * ( entity.getOrientation().x() != 0.0 ),
                                        stepY - ( stepY > 0 ) * ( entity.getOrientation().y() != 0.0 ),
                                        stepZ - ( stepZ > 0 ) * ( entity.getOrientation().z() != 0.0 ) ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ) * ( entity.getOrientation().x() != 0.0 ), stepY + ( stepY < 0 ) * ( entity.getOrientation().y() != 0.0 ), stepZ + ( stepZ < 0 ) * ( entity.getOrientation().z() != 0.0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType(
                        stepX + ( stepX < 0 ) * ( entity.getOrientation().x() != 0.0 ),
                        stepY + ( stepY < 0 ) * ( entity.getOrientation().y() != 0.0 ),
                        stepZ + ( stepZ < 0 ) * ( entity.getOrientation().z() != 0.0 ) )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX - ( stepX > 0 ) * ( entity.getOrientation().x() != 0.0 ),
                                                         entity.getCoordinates().y() + stepY - ( stepY > 0 ) * ( entity.getOrientation().y() != 0.0 ),
                                                         entity.getCoordinates().z() + stepZ - ( stepZ > 0 ) * ( entity.getOrientation().z() != 0.0 ) ) );
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( entity.getMesh(), getEntity< stepX, stepY, stepZ >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       0         |              0            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 3, Real, Device, Index >, 0, Config >,
   0,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 0;
      static constexpr int NeighborEntityDimension = 0;
      typedef Meshes::Grid< 3, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                        CoordinatesType( entity.getCoordinates().x() + stepX,
                                                         entity.getCoordinates().y() + stepY,
                                                         entity.getCoordinates().z() + stepZ ) );
      }

      template< int stepX, int stepY, int stepZ >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) >= CoordinatesType( 0, 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY, stepZ ) <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX, stepY, stepZ )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + stepZ * ( entity.getMesh().getDimensions().y() + 1 + stepY ) * ( entity.getMesh().getDimensions().x() + 1 ) + stepX;
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};

};

} // namespace Meshes
} // namespace TNL
