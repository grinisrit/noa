#pragma once

#include <core/tnlCuda.h>
#include <mesh/MeshDimensionTag.h>
#include "tnlTestNeighborGridEntityGetter.h"

template< typename GridEntity,
          int NeighborEntityDimension >
class tnlTestNeighborGridEntityLayer 
: public tnlTestNeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1 >
{   
   public:
      
      typedef tnlTestNeighborGridEntityLayer< GridEntity, NeighborEntityDimension - 1 > BaseType;
      typedef tnlTestNeighborGridEntityGetter< GridEntity, NeighborEntityDimension > NeighborEntityGetterType;
      
      using BaseType::getNeighborEntities;
      
      __cuda_callable__
      tnlTestNeighborGridEntityLayer( const GridEntity& entity )
      : neighborEntities( entity ),
        BaseType( entity ) 
      {}
            
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
         neighborEntities.refresh( grid, entityIndex );
      };
      
   protected:
      
      NeighborEntityGetterType neighborEntities;
};

template< typename GridEntity >
class tnlTestNeighborGridEntityLayer< GridEntity, 0 >
{
   public:
      
      typedef tnlTestNeighborGridEntityGetter< GridEntity, 0 > NeighborEntityGetterType;     
      
      __cuda_callable__
      tnlTestNeighborGridEntityLayer( const GridEntity& entity )
      : neighborEntities( entity )
      {}
      
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         neighborEntities.refresh( grid, entityIndex );
      };
      
   protected:
      
      NeighborEntityGetterType neighborEntities;
   
};

template< typename GridEntity >
class tnlTestNeighborGridEntitiesStorage
: public tnlTestNeighborGridEntityLayer< GridEntity, GridEntity::meshDimension >
{
   typedef tnlTestNeighborGridEntityLayer< GridEntity, GridEntity::meshDimension > BaseType;
   
   public:
      
      using BaseType::getNeighborEntities;
      
      __cuda_callable__
      tnlTestNeighborGridEntitiesStorage( const GridEntity& entity )
      : BaseType( entity )
      {}

      

      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid, 
                    const typename GridEntity::GridType::IndexType& entityIndex )
      {
         BaseType::refresh( grid, entityIndex );
      };
      
};


