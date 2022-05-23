#pragma once

template< typename Grid,          
          int EntityDimension,
          typename Config >
class tnlTestGridEntity
{
};

/****
 * Specializations for cells
 */
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename Config >
class tnlTestGridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >
{
   public:
      
      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::PointType PointType;
      typedef Config ConfigType;
      
      static const int meshDimension = GridType::meshDimension;
      
      static const int entityDimension = meshDimension;

      constexpr static int getDimension() { return entityDimension; };
      
      constexpr static int getDimension() { return meshDimension; };
      
      
      typedef TNL::Containers::StaticVector< meshDimension, IndexType > EntityOrientationType;
      typedef TNL::Containers::StaticVector< meshDimension, IndexType > EntityBasisType;
      
      /*template< int NeighborEntityDimension = entityDimension >
      using NeighborEntities = 
         tnlTestNeighborGridEntityGetter<
            tnlTestGridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           entityDimension,
                           Config >,
            NeighborEntityDimension >;*/


      __cuda_callable__ inline
      tnlTestGridEntity( const GridType& grid )
      : grid( grid ),
        entityIndex( -1 )/*,
        neighborEntitiesStorage( *this )*/
      {
         this->coordinates = CoordinatesType( ( Index ) 0 );
         this->orientation = EntityOrientationType( ( Index ) 0 );
         this->basis = EntityBasisType( ( Index ) 1 );
      }
      
      
      __cuda_callable__ inline
      tnlTestGridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( 0 ),
                     const EntityBasisType& basis = EntityBasisType( 1 ) )
      : grid( grid ),
        entityIndex( -1 ),
        coordinates( coordinates )/*,
        neighborEntitiesStorage( *this )*/
      {
         this->orientation = EntityOrientationType( ( Index ) 0 );
         this->basis = EntityBasisType( ( Index ) 1 );
      }
      
      

   protected:
      
      const GridType& grid;
      
      IndexType entityIndex;      
      
      CoordinatesType coordinates;
      
      EntityOrientationType orientation;
      
      EntityBasisType basis;
      
      //NeighborGridEntitiesStorageType neighborEntitiesStorage;
      
};

