#pragma once

#include "GridTraverser.h"

namespace TNL {

/****
 * Grid 2D, cells
 */
template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processBoundaryEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Boundary cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );

   GridTraverser< GridType,GridEntity >::template processEntities< GridEntity, EntitiesProcessor, UserData, true, 1, 1 >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      userData,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processInteriorEntities( const GridPointer& gridPointer,
                         UserData& userData ) const
{
   /****
    * Interior cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );

   GridTraverser< GridType, GridEntity >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 1, 1 ),
      gridPointer->getDimensions() - CoordinatesType( 2, 2 ),
      userData,
      0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
   template< typename UserData,
             typename EntitiesProcessor >
void
Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >::
processAllEntities( const GridPointer& gridPointer,
                    UserData& userData ) const
{
   /****
    * All cells
    */
   static_assert( GridEntity::getEntityDimension() == 2, "The entity has wrong dimension." );
 
   GridTraverser< GridType, GridEntity >::template processEntities< GridEntity, EntitiesProcessor, UserData, false >(
      gridPointer,
      CoordinatesType( 0, 0 ),
      gridPointer->getDimensions() - CoordinatesType( 1, 1 ),
      userData,
      0 );
}

} // namespace TNL
