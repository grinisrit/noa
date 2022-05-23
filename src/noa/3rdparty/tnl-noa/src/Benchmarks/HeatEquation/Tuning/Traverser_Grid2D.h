#pragma once

//#include <TNL/Meshes/Traverser.h>
#include <TNL/Pointers/SharedPointer.h>

namespace TNL {
   
template< typename Mesh,
          typename MeshEntity,
          int EntitiesDimension = MeshEntity::getEntityDimension() >
class Traverser
{
   public:
      using MeshType = Mesh;
      using MeshPointer = Pointers::SharedPointer<  MeshType >;
      using DeviceType = typename MeshType::DeviceType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const MeshPointer& meshPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const MeshPointer& meshPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const MeshPointer& meshPointer,
                               UserData& userData ) const;
}; 

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class Traverser< Meshes::Grid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
      using GridType = Meshes::Grid< 2, Real, Device, Index >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using CoordinatesType = typename GridType::CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridPointer& gridPointer,
                                    UserData& userData ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridPointer& gridPointer,
                               UserData& userData ) const;
 
};


} // namespace TNL

#include "Traverser_Grid2D_impl.h"
