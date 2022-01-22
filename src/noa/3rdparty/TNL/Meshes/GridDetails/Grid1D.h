// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Logger.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <noa/3rdparty/TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <noa/3rdparty/TNL/Meshes/GridEntity.h>
#include <noa/3rdparty/TNL/Meshes/GridEntityConfig.h>

namespace noaTNL {
namespace Meshes {

template< typename Real,
          typename Device,
          typename Index >
class Grid< 1, Real, Device, Index >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using GlobalIndexType = Index;
   using PointType = Containers::StaticVector< 1, Real >;
   using CoordinatesType = Containers::StaticVector< 1, Index >;

   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   typedef Index IndexType;

   /**
    * \brief Returns number of this mesh grid dimensions.
    */
   static constexpr int getMeshDimension() { return 1; };

   template< int EntityDimension,
             typename Config = GridEntityCrossStencilStorage< 1 > >
   using EntityType = GridEntity< Grid, EntityDimension, Config >;

   typedef EntityType< getMeshDimension(), GridEntityCrossStencilStorage< 1 > > Cell;
   typedef EntityType< 0 > Face;
   typedef EntityType< 0 > Vertex;

   /**
    * \brief Basic constructor.
    */
   Grid();

   Grid( const Index xSize );

   // empty destructor is needed only to avoid crappy nvcc warnings
   ~Grid() {}

   /**
    * \brief Sets the size of dimensions.
    * \param xSize Size of dimesion x.
    */
   void setDimensions( const Index xSize );

   /**
    * \brief Sets the number of dimensions.
    * \param dimensions Number of dimensions.
    */
   void setDimensions( const CoordinatesType& dimensions );

   __cuda_callable__
   const CoordinatesType& getDimensions() const;

   void setLocalBegin( const CoordinatesType& begin );

   __cuda_callable__
   const CoordinatesType& getLocalBegin() const;

   void setLocalEnd( const CoordinatesType& end );

   __cuda_callable__
   const CoordinatesType& getLocalEnd() const;

   void setInteriorBegin( const CoordinatesType& begin );

   __cuda_callable__
   const CoordinatesType& getInteriorBegin() const;

   void setInteriorEnd( const CoordinatesType& end );

   __cuda_callable__
   const CoordinatesType& getInteriorEnd() const;

   /**
    * \brief Sets the origin.
    * \param origin Starting point of this grid.
    */
   void setOrigin( const PointType& origin);

   /**
    * \brief Sets the origin and proportions of this grid.
    * \param origin Point where this grid starts.
    * \param proportions Total length of this grid.
    */
   void setDomain( const PointType& origin,
                   const PointType& proportions );

   /**
    * \brief Returns the origin.
    * \param origin Starting point of this grid.
    */
   __cuda_callable__
   inline const PointType& getOrigin() const;

   /**
    * \brief Gets length of one entity of this grid.
    */
   __cuda_callable__
   inline const PointType& getProportions() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam EntityDimension Integer specifying dimension of the entity.
    */
   template< int EntityDimension >
   __cuda_callable__
   IndexType getEntitiesCount() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   IndexType getEntitiesCount() const;

   /**
    * \brief Gets entity type using entity index.
    * \param entityIndex Index of entity.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   inline Entity getEntity( const IndexType& entityIndex ) const;

    /**
    * \brief Gets entity index using entity type.
    * \param entity Type of entity.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   inline Index getEntityIndex( const Entity& entity ) const;

   /**
    * \brief Returns the length of one step.
    */
   __cuda_callable__
   inline const PointType& getSpaceSteps() const;

   /**
    * \brief Sets the length of steps.
    * \param steps Length of one step.
    */
   inline void setSpaceSteps(const PointType& steps);

   /**
    * \brief Returns product of space steps to the xPow.
    * \tparam xPow Exponent.
    */
   template< int xPow >
   __cuda_callable__
   const RealType& getSpaceStepsProducts() const;

   /**
    * \breif Returns the measure (length) of a cell in this grid.
    */
   __cuda_callable__
   inline const RealType& getCellMeasure() const;

   /**
    * \brief Returns the smallest length of step out of all coordinates (axes).
    */
   __cuda_callable__
   inline RealType getSmallestSpaceStep() const;

   void writeProlog( Logger& logger ) const;

protected:

   void computeProportions();

   void computeSpaceStepPowers();

   void computeSpaceSteps();

   CoordinatesType dimensions, localBegin, localEnd, interiorBegin, interiorEnd;

   IndexType numberOfCells, numberOfVertices;

   PointType origin, proportions;

   PointType spaceSteps;

   RealType spaceStepsProducts[ 5 ];
};

} // namespace Meshes
} // namespace noaTNL

#include <noa/3rdparty/TNL/Meshes/GridDetails/Grid1D_impl.h>
