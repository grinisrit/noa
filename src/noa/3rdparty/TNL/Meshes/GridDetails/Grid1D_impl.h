// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <iomanip>
#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridEntityGetter_impl.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/NeighborGridEntityGetter1D_impl.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid1D.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>

namespace noaTNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
Grid< 1, Real, Device, Index >::Grid()
: numberOfCells( 0 ),
  numberOfVertices( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
Grid< 1, Real, Device, Index >::Grid( const Index xSize )
: numberOfCells( 0 ),
  numberOfVertices( 0 )
{
   this->setDimensions( xSize );
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index >::computeSpaceSteps()
{
   if( this->getDimensions().x() != 0 )
   {
      this->spaceSteps.x() = this->proportions.x() / ( Real )  this->getDimensions().x();
      this->computeSpaceStepPowers();
   }
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index > ::computeSpaceStepPowers()
{
      const RealType& hx = this->spaceSteps.x();
      this->spaceStepsProducts[ 0 ] = 1.0 / ( hx * hx );
      this->spaceStepsProducts[ 1 ] = 1.0 / hx;
      this->spaceStepsProducts[ 2 ] = 1.0;
      this->spaceStepsProducts[ 3 ] = hx;
      this->spaceStepsProducts[ 4 ] = hx * hx;

}


template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index > ::computeProportions()
{
    this->proportions.x()=this->dimensions.x()*this->spaceSteps.x();
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index > :: setOrigin( const PointType& origin)
{
   this->origin = origin;
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setDimensions( const Index xSize )
{
   TNL_ASSERT_GE( xSize, 0, "Grid size must be non-negative." );
   this->dimensions.x() = xSize;
   this->numberOfCells = xSize;
   this->numberOfVertices = xSize + 1;
   computeSpaceSteps();

   // only default behaviour, DistributedGrid must use the setters explicitly after setDimensions
   localBegin = 0;
   interiorBegin = 1;
   localEnd = dimensions;
   interiorEnd = dimensions - 1;
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setDimensions( const CoordinatesType& dimensions )
{
   this->setDimensions( dimensions. x() );
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::CoordinatesType&
   Grid< 1, Real, Device, Index >::getDimensions() const
{
   return this->dimensions;
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setLocalBegin( const CoordinatesType& begin )
{
   localBegin = begin;
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__
const typename Grid< 1, Real, Device, Index >::CoordinatesType&
   Grid< 1, Real, Device, Index >::getLocalBegin() const
{
   return localBegin;
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setLocalEnd( const CoordinatesType& end )
{
   localEnd = end;
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__
const typename Grid< 1, Real, Device, Index >::CoordinatesType&
   Grid< 1, Real, Device, Index >::
   getLocalEnd() const
{
   return localEnd;
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setInteriorBegin( const CoordinatesType& begin )
{
   interiorBegin = begin;
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__
const typename Grid< 1, Real, Device, Index >::CoordinatesType&
   Grid< 1, Real, Device, Index >::getInteriorBegin() const
{
   return interiorBegin;
}

template< typename Real,
          typename Device,
          typename Index  >
void Grid< 1, Real, Device, Index >::setInteriorEnd( const CoordinatesType& end )
{
   interiorEnd = end;
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__
const typename Grid< 1, Real, Device, Index >::CoordinatesType&
   Grid< 1, Real, Device, Index >::getInteriorEnd() const
{
   return interiorEnd;
}

template< typename Real,
          typename Device,
          typename Index >
void Grid< 1, Real, Device, Index >::setDomain( const PointType& origin,
                                                const PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;
   computeSpaceSteps();
}

template< typename Real,
          typename Device,
          typename Index  >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::PointType&
  Grid< 1, Real, Device, Index >::getOrigin() const
{
   return this->origin;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::PointType&
   Grid< 1, Real, Device, Index >::getProportions() const
{
   return this->proportions;
}


template< typename Real,
          typename Device,
          typename Index >
   template< int EntityDimension >
__cuda_callable__  inline
Index
Grid< 1, Real, Device, Index >::
getEntitiesCount() const
{
   static_assert( EntityDimension <= 1 &&
                  EntityDimension >= 0, "Wrong grid entity dimensions." );

   switch( EntityDimension )
   {
      case 1:
         return this->numberOfCells;
      case 0:
         return this->numberOfVertices;
   }
   return -1;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__  inline
Index
Grid< 1, Real, Device, Index >::
getEntitiesCount() const
{
   return getEntitiesCount< Entity::getEntityDimension() >();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
 __cuda_callable__ inline
Entity
Grid< 1, Real, Device, Index >::
getEntity( const IndexType& entityIndex ) const
{
   static_assert( Entity::getEntityDimension() <= 1 &&
                  Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntity( *this, entityIndex );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Entity >
__cuda_callable__ inline
Index
Grid< 1, Real, Device, Index >::
getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::getEntityDimension() <= 1 &&
                  Entity::getEntityDimension() >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity >::getEntityIndex( *this, entity );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
const typename Grid< 1, Real, Device, Index >::PointType&
Grid< 1, Real, Device, Index >::
getSpaceSteps() const
{
   return this->spaceSteps;
}

template< typename Real,
          typename Device,
          typename Index >
inline void
Grid< 1, Real, Device, Index >::
setSpaceSteps(const typename Grid< 1, Real, Device, Index >::PointType& steps)
{
    this->spaceSteps=steps;
    computeSpaceStepPowers();
    computeProportions();
}

template< typename Real,
          typename Device,
          typename Index >
   template< int xPow >
__cuda_callable__ inline
const Real&
Grid< 1, Real, Device, Index >::
getSpaceStepsProducts() const
{
   static_assert( xPow >= -2 && xPow <= 2, "unsupported value of xPow" );
   return this->spaceStepsProducts[ xPow + 2 ];
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
const Real&
Grid< 1, Real, Device, Index >::
getCellMeasure() const
{
   return this->template getSpaceStepsProducts< 1 >();
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__ inline
Real Grid< 1, Real, Device, Index >::
getSmallestSpaceStep() const
{
   return this->spaceSteps.x();
}

template< typename Real,
          typename Device,
          typename Index >
void
Grid< 1, Real, Device, Index >::
writeProlog( Logger& logger ) const
{
   logger.writeParameter( "Dimension:", getMeshDimension() );
   logger.writeParameter( "Domain origin:", this->origin );
   logger.writeParameter( "Domain proportions:", this->proportions );
   logger.writeParameter( "Domain dimensions:", this->dimensions );
   logger.writeParameter( "Space steps:", this->getSpaceSteps() );
   logger.writeParameter( "Number of cells:", getEntitiesCount< Cell >() );
   logger.writeParameter( "Number of vertices:", getEntitiesCount< Vertex >() );
}

} // namespace Meshes
} // namespace noaTNL
