// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid1D.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid2D.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid3D.h>
#include <noa/3rdparty/TNL/Algorithms/staticFor.h>

namespace noa::TNL {
namespace Meshes {

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              2            | No specialization |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config >,
   2,
   StencilStorage >
{
   public:

      static constexpr int EntityDimension = 2;
      static constexpr int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
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

      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                         CoordinatesType( entity.getCoordinates().x() + stepX,
                                                          entity.getCoordinates().y() + stepY ) );
      }

      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              2            |       Cross       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config >,
   2,
   GridEntityStencilStorageTag< GridEntityCrossStencil > >
{
   public:

      static constexpr int EntityDimension = 2;
      static constexpr int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
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

      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
            return NeighborGridEntityType( this->entity.getMesh(),
                                            CoordinatesType( entity.getCoordinates().x() + stepX,
                                                             entity.getCoordinates().y() + stepY ) );
      }

      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
#ifndef HAVE_CUDA // TODO: fix this to work with CUDA
         if( ( stepX != 0 && stepY != 0 ) ||
             ( stepX < -stencilSize || stepX > stencilSize ||
               stepY < -stencilSize || stepY > stencilSize ) )
            return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
         if( stepY == 0 )
            return stencilX[ stepX + stencilSize ];
         return stencilY[ stepY + stencilSize ];
#else
         return this->entity.getIndex() + stepY * entity.getMesh().getDimensions().x() + stepX;
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
         Algorithms::staticFor< IndexType, -stencilSize, 0 >( stencilYRefresher );
         Algorithms::staticFor< IndexType, 1, stencilSize + 1 >( stencilYRefresher );
         Algorithms::staticFor< IndexType, -stencilSize, stencilSize + 1 >( stencilXRefresher );
#endif
      };

   protected:

      const GridEntityType& entity;

      IndexType stencilX[ 2 * stencilSize + 1 ];
      IndexType stencilY[ 2 * stencilSize + 1 ];

      //NeighborGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |              1            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config >,
   1,
   StencilStorage >
{
   public:

      static constexpr int EntityDimension = 2;
      static constexpr int NeighborEntityDimension = 1;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
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

      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         static_assert( ! stepX + ! stepY == 1, "Only one of the steps can be non-zero." );
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ),
                                        stepY + ( stepY < 0 ) )
                       < entity.getMesh().getDimensions() + CoordinatesType( ( stepX > 0 ), ( stepY > 0 ) ),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                         CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                          entity.getCoordinates().y() + stepY + ( stepY < 0 ) ),
                                         EntityOrientationType( stepX ? (stepX > 0 ? 1 : -1) : 0,
                                                                stepY ? (stepY > 0 ? 1 : -1) : 0 ),
                                         EntityBasisType( ! stepX, ! stepY ) );
      }

      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( this->entity.getMesh(), this->template getEntity< stepX, stepY >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       2         |            0              |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config >,
   0,
   StencilStorage >
{
   public:

      static constexpr int EntityDimension = 2;
      static constexpr int NeighborEntityDimension = 0;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
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

      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT( stepX != 0 && stepY != 0,
                    std::cerr << " stepX = " << stepX << " stepY = " << stepY );
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                       < entity.getMesh().getDimensions() + CoordinatesType( ( stepX > 0 ), ( stepY > 0 ) ),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " entity.getMesh().getDimensions() + CoordinatesType( sign( stepX ), sign( stepY ) ) = "
                   << entity.getMesh().getDimensions()  + CoordinatesType( sign( stepX ), sign( stepY ) )
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                         CoordinatesType( entity.getCoordinates().x() + stepX + ( stepX < 0 ),
                                                          entity.getCoordinates().y() + stepY + ( stepY < 0 ) ) );
      }

      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( this->entity.getMesh(), this->template getEntity< stepX, stepY >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       1         |              2            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 1, Config >,
   2,
   StencilStorage >
{
   public:

      static constexpr int EntityDimension = 1;
      static constexpr int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
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

      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         /*TNL_ASSERT( ( ( !! stepX ) == ( !! entity.getOrientation().x() ) ) &&
                    ( ( !! stepY ) == ( !! entity.getOrientation().y() ) ),
                    std::cerr << "( stepX, stepY ) cannot be perpendicular to entity coordinates: stepX = " << stepX << " stepY = " << stepY
                         << " entity.getOrientation() = " << entity.getOrientation() );*/
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions() + noa::TNL::abs(entity.getOrientation()), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() +
                       CoordinatesType( stepX - ( stepX > 0 ) * ( entity.getOrientation().x() != 0.0 ),
                                        stepY - ( stepY > 0 ) * ( entity.getOrientation().y() != 0.0 ) ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() +
                       CoordinatesType( stepX - ( stepX > 0 ) * ( entity.getOrientation().x() != 0.0 ),
                                        stepY - ( stepY > 0 ) * ( entity.getOrientation().y() != 0.0 ) ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 )  * ( entity.getOrientation().x() != 0.0 ), stepY + ( stepY < 0 ) * ( entity.getOrientation().y() != 0.0 ) ) = "
                   << entity.getCoordinates()  + CoordinatesType( stepX + ( stepX < 0 ), stepY + ( stepY < 0 ) )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                     CoordinatesType( entity.getCoordinates().x() + stepX - ( stepX > 0 ) * ( entity.getOrientation().x() != 0.0 ),
                                      entity.getCoordinates().y() + stepY - ( stepY > 0 ) * ( entity.getOrientation().y() != 0.0 ) ) );
      }

      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         return GridEntityGetterType::getEntityIndex( this->entity.getMesh(), this->template getEntity< stepX, stepY >() );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stencil Storage  |
 * +-----------------+---------------------------+-------------------+
 * |       0         |              0            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 0, Config >,
   0,
   StencilStorage >
{
   public:

      static constexpr int EntityDimension = 0;
      static constexpr int NeighborEntityDimension = 0;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
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

      template< int stepX, int stepY >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(),
                                         CoordinatesType( entity.getCoordinates().x() + stepX,
                                                          entity.getCoordinates().y() + stepY ) );
      }

      template< int stepX, int stepY >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0, 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( stepX, stepY ) >= CoordinatesType( 0, 0 ) &&
                    entity.getCoordinates() + CoordinatesType( stepX, stepY ) <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates()  + CoordinatesType( stepX, stepY ) = " << entity.getCoordinates()  + CoordinatesType( stepX, stepY )
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + stepY * ( entity.getMesh().getDimensions().x() + 1 ) + stepX;
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;

      //NeighborGridEntityGetter(){};
};

} // namespace Meshes
} // namespace noa::TNL
