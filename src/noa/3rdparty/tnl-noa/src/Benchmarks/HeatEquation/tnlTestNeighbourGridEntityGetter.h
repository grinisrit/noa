#pragma once 

#include <core/tnlTNL_ASSERT.h>


template< typename GridEntity,
          int NeighborEntityDimension,
          typename EntityStencilTag = 
            GridEntityStencilStorageTag< GridEntity::ConfigType::template neighborEntityStorage< GridEntity >( NeighborEntityDimension ) > >
class tnlTestNeighborGridEntityGetter
{
   public:

      // TODO: not all specializations are implemented yet
      
      __cuda_callable__
      tnlTestNeighborGridEntityGetter( const GridEntity& entity )
      {
         //tnlTNL_ASSERT( false, );
      };
      
      __cuda_callable__
      void refresh( const typename GridEntity::GridType& grid,
                    const typename GridEntity::IndexType& entityIndex )
      {
         //tnlTNL_ASSERT( false, );
      };

};

template< typename Real,
          typename Device,
          typename Index,
          typename Config,
          typename StencilStorage >
class tnlTestNeighborGridEntityGetter< 
   GridEntity< Meshes::Grid< 2, Real, Device, Index >, 2, Config >,
   2,
   StencilStorage >
{
   public:
      
      static const int EntityDimension = 2;
      static const int NeighborEntityDimension = 2;
      typedef Meshes::Grid< 2, Real, Device, Index > GridType;
      typedef GridEntity< GridType, EntityDimension, Config > GridEntityType;
      typedef GridEntity< GridType, NeighborEntityDimension, Config > NeighborGridEntityType;
      typedef Real RealType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef GridEntityGetter< GridType, NeighborGridEntityType > GridEntityGetter;

      __cuda_callable__ inline
      tnlTestNeighborGridEntityGetter( const GridEntityType& entity )
      : entity( entity )
      {}
            
      __cuda_callable__
      void refresh( const GridType& grid, const IndexType& entityIndex ){};
      
   protected:

      const GridEntityType& entity;
      
      //tnlTestNeighborGridEntityGetter(){};      
};


