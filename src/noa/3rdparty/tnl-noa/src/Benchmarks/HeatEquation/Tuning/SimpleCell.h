#pragma once

#include <TNL/Meshes/GridDetails/NeighborGridEntitiesStorage.h>
#include <TNL/Meshes/GridEntityConfig.h>

#define SIMPLE_CELL_HAVE_NEIGHBOR_ENTITIES_STORAGE

template< typename Grid, typename Config = Meshes::GridEntityNoStencilStorage >
class SimpleCell
{
   public:
      using GridType = Grid;
      using MeshType = GridType;
      using RealType = typename GridType::RealType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename GridType::CoordinatesType;
      using PointType = typename GridType::PointType;
      using NeighborGridEntitiesStorageType = Meshes::NeighborGridEntitiesStorage< SimpleCell, Config >;
      using ConfigType = Config;

      constexpr static int getMeshDimension() { return GridType::getMeshDimension(); };
 
      constexpr static int getEntityDimension() { return getMeshDimension(); };
       
      __cuda_callable__ inline
      SimpleCell( const GridType& grid )
      :grid( grid )
#ifdef SIMPLE_CELL_HAVE_NEIGHBOR_ENTITIES_STORAGE         
      , neighborEntitiesStorage( *this )
#endif      
      {};
 
      __cuda_callable__ inline
      SimpleCell( const GridType& grid,
                  const CoordinatesType& coordinates,
                  const CoordinatesType& orientation = CoordinatesType( ( IndexType ) 0 ),
                  const CoordinatesType& basis = CoordinatesType( ( IndexType ) 1 ) )
      : grid( grid ),
        coordinates( coordinates )
#ifdef SIMPLE_CELL_HAVE_NEIGHBOR_ENTITIES_STORAGE               
      , neighborEntitiesStorage( *this )
#endif      
      {};
 
      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const { return this->coordinates; };
 
      __cuda_callable__ inline
      CoordinatesType& getCoordinates() { return this->coordinates; };
 
      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates ) { this->coordinates = coordinates; };

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      void refresh() 
      { 
         this->entityIndex = this->grid.getEntityIndex( *this );
         this->neighborEntitiesStorage.refresh( this->grid, this->entityIndex );
      };

      __cuda_callable__ inline
      IndexType getIndex() const { return this->entityIndex; };
 
      /*__cuda_callable__ inline
      const EntityOrientationType getOrientation() const;
 
      __cuda_callable__ inline
      void setOrientation( const EntityOrientationType& orientation ){};
 
      __cuda_callable__ inline
      const EntityBasisType getBasis() const;
 
      __cuda_callable__ inline
      void setBasis( const EntityBasisType& basis ){};
 
      template< int NeighborEntityDimension = Dimension >
      __cuda_callable__ inline
      const NeighborEntities< NeighborEntityDimension >&
      getNeighborEntities() const;
      */
      __cuda_callable__ inline
      bool isBoundaryEntity() const
      {
         return false;
         /*return( this->getCoordinates().x() == 0 ||
                 this->getCoordinates().y() == 0 ||
                 this->getCoordinates().x() == this->getMesh().getDimensions().x() - 1 ||
                 this->getCoordinates().y() == this->getMesh().getDimensions().y() - 1 );*/
      };
      
 
      __cuda_callable__ inline
      PointType getCenter() const
      {
         return PointType(
            grid.getOrigin().x() + ( coordinates.x() + 0.5 ) * grid.getSpaceSteps().x(),
            grid.getOrigin().y() + ( coordinates.y() + 0.5 ) * grid.getSpaceSteps().y() );
      };
 
      /*__cuda_callable__ inline
      const RealType& getMeasure() const;
 
      __cuda_callable__ inline
      const PointType& getEntityProportions() const;*/
 
      __cuda_callable__ inline
      const GridType& getMesh() const { return this->grid; };

   protected:
 
      const GridType& grid;
 
      IndexType entityIndex;
 
      CoordinatesType coordinates;
       
#ifdef SIMPLE_CELL_HAVE_NEIGHBOR_ENTITIES_STORAGE               
      NeighborGridEntitiesStorageType neighborEntitiesStorage;
#endif
      
      
      // TODO: Test of boundary entity will likely be more
      // complicated with MPI. It might be more efficient to resolve it
      // before.
      //bool isBoundaryEnity;
};
