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

namespace noaTNL {
namespace Meshes {

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       1         |              1            |       ----        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
   1,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 1;
      static constexpr int NeighborEntityDimension = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
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

      template< int step >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }

      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + step;
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
};


/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       1         |              1            |  Cross/Full       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
   1,
   StencilStorage >
{
   public:

      static constexpr int EntityDimension = 1;
      static constexpr int NeighborEntityDimension = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetterType;

      static constexpr int stencilSize = Config::getStencilSize();

      __cuda_callable__ inline
      NeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}

      template< int step >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntityType( this->entity.getMesh(), CoordinatesType( entity.getCoordinates().x() + step ) );
      }

      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates() + CoordinatesType( step ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates() + CoordinatesType( step ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
#ifndef HAVE_CUDA  // TODO: fix it -- does not work with nvcc
         if( step < -stencilSize || step > stencilSize )
            return this->entity.getIndex() + step;
         return stencil[ stencilSize + step ];
#else
         return this->entity.getIndex() + step;
#endif

      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex )
      {
#ifndef HAVE_CUDA  // TODO: fix it -- does not work with nvcc
         Algorithms::staticFor< IndexType, -stencilSize, stencilSize + 1 >(
            [&] ( auto index ) {
               stencil[ index + stencilSize ] = entityIndex + index;
            }
         );
#endif
      };

   protected:

      const GridEntityType& entity;

      IndexType stencil[ 2 * stencilSize + 1 ];
};

/****
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       1         |              0            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 1, Config >,
   0,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 1;
      static constexpr int NeighborEntityDimension = 0;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
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

      template< int step >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step + ( step < 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step + ( step < 0 ) <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step + ( step < 0 ) ) );
      }

      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LT( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step + ( step < 0 ) >= CoordinatesType( 0 ).x() &&
                    entity.getCoordinates().x() + step + ( step < 0 ) <= entity.getMesh().getDimensions().x(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + step + ( step < 0 );
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
 * |       0         |              1            |       None        |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   1,
   StencilStorage > //GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 0;
      static constexpr int NeighborEntityDimension = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
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

      void test() const { std::cerr << "***" << std::endl; };

      template< int step >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }

      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= 0 &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions().x(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + step - ( step > 0 );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
};

/****   TODO: Implement this, now it is only a copy of specialization for none stencil storage
 * +-----------------+---------------------------+-------------------+
 * | EntityDimenions | NeighborEntityDimension |  Stored Stencil   |
 * +-----------------+---------------------------+-------------------+
 * |       0         |              1            |       Cross       |
 * +-----------------+---------------------------+-------------------+
 */
template< typename Real,
          typename Device,
          typename Index,
          typename Config >
class NeighborGridEntityGetter<
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   1,
   GridEntityStencilStorageTag< GridEntityCrossStencil > >
{
   public:

      static constexpr int EntityDimension = 0;
      static constexpr int NeighborEntityDimension = 1;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
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


      template< int step >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step - ( step > 0 ) ) );
      }

      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step - ( step > 0 ) >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step - ( step > 0 ) < entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return this->entity.getIndex() + step - ( step > 0 );
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};

   protected:

      const GridEntityType& entity;
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
   GridEntity< Meshes::Grid< 1, Real, Device, Index >, 0, Config >,
   0,
   GridEntityStencilStorageTag< GridEntityNoStencil > >
{
   public:

      static constexpr int EntityDimension = 0;
      static constexpr int NeighborEntityDimension = 0;
      typedef Meshes::Grid< 1, Real, Device, Index > GridType;
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

      template< int step >
      __cuda_callable__ inline
      NeighborGridEntityType getEntity() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );
         return NeighborGridEntity( CoordinatesType( entity.getCoordinates().x() + step ) );
      }

      template< int step >
      __cuda_callable__ inline
      IndexType getEntityIndex() const
      {
         TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "wrong coordinates" );
         TNL_ASSERT_LE( entity.getCoordinates(), entity.getMesh().getDimensions(), "wrong coordinates" );
         TNL_ASSERT( entity.getCoordinates().x() + step >= CoordinatesType( 0 ) &&
                    entity.getCoordinates().x() + step <= entity.getMesh().getDimensions(),
              std::cerr << "entity.getCoordinates() = " << entity.getCoordinates()
                   << " entity.getMesh().getDimensions() = " << entity.getMesh().getDimensions()
                   << " EntityDimension = " << EntityDimension );

         return this->entity.getIndex() + step;
      }

      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};


   protected:

      const GridEntityType& entity;
};

} // namespace Meshes
} // namespace noaTNL
