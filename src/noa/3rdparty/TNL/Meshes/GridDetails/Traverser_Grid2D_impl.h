// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/GridDetails/GridTraverser.h>

namespace noa::TNL {
namespace Meshes {

/****
 * Grid 2D, cells
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );

   if( gridPointer->getLocalBegin() < gridPointer->getInteriorBegin() && gridPointer->getInteriorEnd() < gridPointer->getLocalEnd() )
   {
      // 4 boundaries (left, right, down, up)
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1 >(
         gridPointer,
         gridPointer->getInteriorBegin() - 1,
         gridPointer->getInteriorEnd() + 1,
         userData,
         asynchronousMode,
         0 );
   }
   else
   {
      const CoordinatesType begin = gridPointer->getLocalBegin();
      const CoordinatesType end = gridPointer->getLocalEnd();
      const CoordinatesType skipBegin = gridPointer->getInteriorBegin();
      const CoordinatesType skipEnd = gridPointer->getInteriorEnd();

      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         begin,
         CoordinatesType( end.x(), skipBegin.y() ),
         userData,
         asynchronousMode,
         0 );
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         CoordinatesType( begin.x(), skipEnd.y() ),
         end,
         userData,
         asynchronousMode,
         0 );
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         CoordinatesType( begin.x(), skipBegin.y() ),
         CoordinatesType( skipBegin.x(), skipEnd.y() ),
         userData,
         asynchronousMode,
         0 );
      GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
         gridPointer,
         CoordinatesType( skipEnd.x(), skipBegin.y() ),
         CoordinatesType( end.x(), skipEnd.y() ),
         userData,
         asynchronousMode,
         0 );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimensions." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      gridPointer->getInteriorBegin(),
      gridPointer->getInteriorEnd(),
      userData,
      asynchronousMode,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      gridPointer->getLocalBegin(),
      gridPointer->getLocalEnd(),
      userData,
      asynchronousMode,
      0 );
}

/****
 * Grid 2D, faces
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary faces
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 0, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() + CoordinatesType( 1, 0 ),
      userData,
      asynchronousMode,
      1,
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 0, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() + CoordinatesType( 0, 1 ),
      userData,
      asynchronousMode,
      0,
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior faces
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 1, 0 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode,
      1,
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 1 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode,
      0,
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 1 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All faces
    */
   static_assert( GridEntity::getEntityDimension() == 1, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() + CoordinatesType( 1, 0 ),
      userData,
      asynchronousMode,
      1,
      CoordinatesType( 1, 0 ),
      CoordinatesType( 0, 1 ) );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false, 1, 1, CoordinatesType, CoordinatesType >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() + CoordinatesType( 0, 1 ),
      userData,
      asynchronousMode,
      0,
      CoordinatesType( 0, 1 ),
      CoordinatesType( 1, 0 ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1 >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() + CoordinatesType( 1, 1 ),
      userData,
      asynchronousMode,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 1 ),
      gridPointer->getDimensions(),
      userData,
      asynchronousMode,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename EntitiesProcessor,
             typename UserData >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 0 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All vertices
    */
   static_assert( GridEntity::getEntityDimension() == 0, "The entity has wrong dimension." );

   GridTraverser< GridType >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() + CoordinatesType( 1, 1 ),
      userData,
      asynchronousMode,
      0 );
}

} // namespace Meshes
} // namespace noa::TNL
