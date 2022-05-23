// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Logger.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridEntity.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridEntityConfig.h>

namespace noa::TNL {
namespace Meshes {

template< typename Real, typename Device, typename Index >
class Grid< 2, Real, Device, Index >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using GlobalIndexType = Index;
   using PointType = Containers::StaticVector< 2, Real >;
   using CoordinatesType = Containers::StaticVector< 2, Index >;

   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   using IndexType = Index;

   static constexpr int
   getMeshDimension()
   {
      return 2;
   };

   template< int EntityDimension, typename Config = GridEntityCrossStencilStorage< 1 > >
   using EntityType = GridEntity< Grid, EntityDimension, Config >;

   using Cell = EntityType< getMeshDimension(), GridEntityCrossStencilStorage< 1 > >;
   using Face = EntityType< getMeshDimension() - 1 >;
   using Vertex = EntityType< 0 >;

   /**
    * \brief See Grid1D::Grid().
    */
   Grid() = default;

   Grid( Index xSize, Index ySize );

   // empty destructor is needed only to avoid crappy nvcc warnings
   ~Grid() = default;

   /**
    * \brief Sets the size of dimensions.
    * \param xSize Size of dimension x.
    * \param ySize Size of dimension y.
    */
   void
   setDimensions( Index xSize, Index ySize );

   /**
    * \brief See Grid1D::setDimensions( const CoordinatesType& dimensions ).
    */
   void
   setDimensions( const CoordinatesType& dimensions );

   /**
    * \brief See Grid1D::getDimensions().
    */
   __cuda_callable__
   const CoordinatesType&
   getDimensions() const;

   void
   setLocalBegin( const CoordinatesType& begin );

   __cuda_callable__
   const CoordinatesType&
   getLocalBegin() const;

   void
   setLocalEnd( const CoordinatesType& end );

   __cuda_callable__
   const CoordinatesType&
   getLocalEnd() const;

   void
   setInteriorBegin( const CoordinatesType& begin );

   __cuda_callable__
   const CoordinatesType&
   getInteriorBegin() const;

   void
   setInteriorEnd( const CoordinatesType& end );

   __cuda_callable__
   const CoordinatesType&
   getInteriorEnd() const;

   /**
    * \brief See Grid1D::setDomain().
    */
   void
   setDomain( const PointType& origin, const PointType& proportions );

   /**
    * \brief See Grid1D::setOrigin()
    */
   void
   setOrigin( const PointType& origin );

   /**
    * \brief See Grid1D::getOrigin().
    */
   __cuda_callable__
   inline const PointType&
   getOrigin() const;

   /**
    * \brief See Grid1D::getProportions().
    */
   __cuda_callable__
   inline const PointType&
   getProportions() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam EntityDimension Integer specifying dimension of the entity.
    */
   template< int EntityDimension >
   __cuda_callable__
   IndexType
   getEntitiesCount() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   inline IndexType
   getEntitiesCount() const;

   /**
    * \brief See Grid1D::getEntity().
    */
   template< typename Entity >
   __cuda_callable__
   inline Entity
   getEntity( const IndexType& entityIndex ) const;

   /**
    * \brief See Grid1D::getEntityIndex().
    */
   template< typename Entity >
   __cuda_callable__
   inline Index
   getEntityIndex( const Entity& entity ) const;

   /**
    * \brief See Grid1D::getSpaceSteps().
    */
   __cuda_callable__
   inline const PointType&
   getSpaceSteps() const;

   /**
    * \brief See Grid1D::setSpaceSteps().
    */
   inline void
   setSpaceSteps( const PointType& steps );

   /**
    * \brief Returns product of space steps to the xPow.
    * \tparam xPow Exponent for dimension x.
    * \tparam yPow Exponent for dimension y.
    */
   template< int xPow, int yPow >
   __cuda_callable__
   const RealType&
   getSpaceStepsProducts() const;

   /**
    * \brief Returns the number of x-normal faces.
    */
   __cuda_callable__
   IndexType
   getNumberOfNxFaces() const;

   /**
    * \breif Returns the measure (area) of a cell in this grid.
    */
   __cuda_callable__
   inline const RealType&
   getCellMeasure() const;

   /**
    * \brief See Grid1D::getSmallestSpaceStep().
    */
   __cuda_callable__
   inline RealType
   getSmallestSpaceStep() const;

   void
   writeProlog( Logger& logger ) const;

protected:
   void
   computeProportions();

   __cuda_callable__
   void
   computeSpaceStepPowers();

   __cuda_callable__
   void
   computeSpaceSteps();

   CoordinatesType dimensions = { 0, 0 };
   CoordinatesType localBegin = { 0, 0 };
   CoordinatesType localEnd = { 0, 0 };
   CoordinatesType interiorBegin = { 0, 0 };
   CoordinatesType interiorEnd = { 0, 0 };

   IndexType numberOfCells = 0;
   IndexType numberOfNxFaces = 0;
   IndexType numberOfNyFaces = 0;
   IndexType numberOfFaces = 0;
   IndexType numberOfVertices = 0;

   PointType origin = { 0, 0 };
   PointType proportions = { 0, 0 };
   PointType spaceSteps = { 0, 0 };

   RealType spaceStepsProducts[ 5 ][ 5 ];

   template< typename, typename, int >
   friend class GridEntityGetter;
};

}  // namespace Meshes
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Grid2D_impl.h>
