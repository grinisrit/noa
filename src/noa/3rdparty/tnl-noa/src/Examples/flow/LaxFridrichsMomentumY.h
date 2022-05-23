#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include "LaxFridrichsMomentumBase.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsMomentumY
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumY< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsMomentumBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:

      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsMomentumBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& rho_v,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 1, "Wrong preimage function" ); 
         //const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities(); 

         return 0.0;
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumY< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsMomentumBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsMomentumBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& rho_v,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 2, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 


         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1 >(); 
         const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0 >(); 
         const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2 >(); 

         const IndexType& center = entity.getIndex(); 
         const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north = neighborEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south = neighborEntities.template getEntityIndex<  0, -1 >();
         const IndexType& southEast = neighborEntities.template getEntityIndex<  1, -1 >();
         const IndexType& southWest = neighborEntities.template getEntityIndex<  -1, -1 >();
         const IndexType& northEast = neighborEntities.template getEntityIndex<  1, 1 >();
         const IndexType& northWest = neighborEntities.template getEntityIndex<  -1, 1 >();
         
         const RealType& pressure_north = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south = this->pressure.template getData< DeviceType >()[ south ];         
         const RealType& velocity_x_east = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_center = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_southEast = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_x_southWest = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_x_northEast = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_x_northWest = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ northWest ];         

         const RealType& velocity_y_north = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_y_east = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_y_west = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_center = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_southEast = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_y_southWest = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_y_northEast = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_y_northWest = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ northWest ];         
         
         
         return 1.0 / ( 4.0 * this->tau ) * this->artificialViscosity * ( rho_v[ west ] + rho_v[ east ] + rho_v[ south ] + rho_v[ north ] - 4.0 * rho_v[ center ] ) 
                - 0.5 * ( ( ( rho_v[ east ] * velocity_x_east )
                           - ( rho_v[ west ] * velocity_x_west ) )* hxInverse
                        + ( ( rho_v[ north ] * velocity_y_north + pressure_north )
                          - ( rho_v[ south ] * velocity_y_south + pressure_south ) )* hyInverse )
// 2D T_22_y
                + ( 4.0 / 3.0 * ( velocity_y_north - 2 * velocity_y_center + velocity_y_south
                                ) * hySquareInverse
                  - 2.0 / 3.0 * ( velocity_x_northEast - velocity_x_southEast - velocity_x_northWest + velocity_x_southWest
                                ) * hxInverse * hyInverse / 4
                  ) * this->dynamicalViscosity
// T_12_x
                + ( ( velocity_x_northEast - velocity_x_southEast - velocity_x_northWest + velocity_x_southWest
                    ) * hxInverse * hyInverse / 4
                  + ( velocity_y_west - 2 * velocity_y_center + velocity_y_east
                    ) * hxSquareInverse
                  ) * this->dynamicalViscosity;
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumY< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsMomentumBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsMomentumBase< MeshType, Real, Index > BaseType;
      
      using typename BaseType::RealType;
      using typename BaseType::IndexType;
      using typename BaseType::DeviceType;
      using typename BaseType::CoordinatesType;
      using typename BaseType::MeshFunctionType;
      using typename BaseType::MeshFunctionPointer;
      using typename BaseType::VelocityFieldType;
      using typename BaseType::VelocityFieldPointer;
      using BaseType::Dimensions;      

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& rho_v,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 3, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts< 0,  0, -1 >(); 
         const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0,  0 >(); 
         const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2,  0 >(); 
         const RealType& hzSquareInverse = entity.getMesh().template getSpaceStepsProducts< 0,  0, -2 >(); 

         const IndexType& center = entity.getIndex(); 
         const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south = neighborEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up    = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down  = neighborEntities.template getEntityIndex<  0,  0, -1 >();
         const IndexType& northWest = neighborEntities.template getEntityIndex<  -1,  1,  0 >(); 
         const IndexType& northEast = neighborEntities.template getEntityIndex<  1,  1,  0 >(); 
         const IndexType& southWest = neighborEntities.template getEntityIndex<  -1, -1,  0 >();
         const IndexType& southEast = neighborEntities.template getEntityIndex<  1, -1,  0 >();
         const IndexType& upWest    = neighborEntities.template getEntityIndex<  -1,  0,  1 >();
         const IndexType& upEast    = neighborEntities.template getEntityIndex<  1,  0,  1 >();
         const IndexType& upSouth    = neighborEntities.template getEntityIndex<  0,  -1,  1 >();
         const IndexType& upNorth    = neighborEntities.template getEntityIndex<  0,  1,  1 >();
         const IndexType& downWest  = neighborEntities.template getEntityIndex<  -1,  0, -1 >();
         const IndexType& downEast  = neighborEntities.template getEntityIndex<  1,  0, -1 >();
         const IndexType& downSouth  = neighborEntities.template getEntityIndex<  0,  -1, -1 >();
         const IndexType& downNorth  = neighborEntities.template getEntityIndex<  0,  1, -1 >();
         
         const RealType& pressure_west  = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east  = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& pressure_north = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south = this->pressure.template getData< DeviceType >()[ south ];
         const RealType& pressure_up    = this->pressure.template getData< DeviceType >()[ up ];
         const RealType& pressure_down  = this->pressure.template getData< DeviceType >()[ down ];
         
         const RealType& velocity_x_east  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_center  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_northWest  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ northWest ];
         const RealType& velocity_x_northEast  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_x_southWest  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_x_southEast  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_x_upWest  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ upWest ];
         const RealType& velocity_x_downWest  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ downWest ];
         const RealType& velocity_x_upEast  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ upEast ];
         const RealType& velocity_x_downEast  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ downEast ];

         const RealType& velocity_y_north = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_y_center = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_northWest = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ northWest ];
         const RealType& velocity_y_northEast = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_y_southWest = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_y_southEast = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_y_upNorth = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ upNorth ];
         const RealType& velocity_y_upSouth = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ upSouth ];
         const RealType& velocity_y_downNorth = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ downNorth ];
         const RealType& velocity_y_downSouth = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ downSouth ];

         const RealType& velocity_z_up    = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ down ]; 
         const RealType& velocity_z_center  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ center ]; 
         const RealType& velocity_z_upWest  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ upWest ]; 
         const RealType& velocity_z_upEast  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ upEast ]; 
         const RealType& velocity_z_upNorth  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ upNorth ]; 
         const RealType& velocity_z_upSouth  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ upSouth ]; 
         const RealType& velocity_z_downWest  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ downWest ]; 
         const RealType& velocity_z_downEast  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ downEast ]; 
         const RealType& velocity_z_downNorth  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ downNorth ]; 
         const RealType& velocity_z_downSouth  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ downSouth ];         

          return 1.0 / ( 6.0 * this->tau ) * this->artificialViscosity * 
                   ( rho_v[ west ] + rho_v[ east ] + rho_v[ south ] + rho_v[ north ] + rho_v[ up ] + rho_v[ down ] - 6.0 * rho_v[ center ] ) 
                - 0.5 * ( ( ( rho_v[ east ] * velocity_x_east )
                          - ( rho_v[ west ] * velocity_x_west ) ) * hxInverse
                        + ( ( rho_v[ north ] * velocity_y_north + pressure_north )
                          - ( rho_v[ south ] * velocity_y_south + pressure_south ) ) * hyInverse
                        + ( ( rho_v[ up ] * velocity_z_up )
                          - ( rho_v[ down ] * velocity_z_down ) ) * hzInverse )
// T_12_y
                + ( ( velocity_x_northEast - velocity_x_southEast - velocity_x_northWest + velocity_x_southWest
                    ) * hxInverse * hyInverse / 4
                  + (  velocity_y_north - 2 * velocity_y_center + velocity_y_south
                    ) * hySquareInverse
                  ) * this->dynamicalViscosity
// 3D T_22_y
                + ( 4.0 / 3.0 * ( velocity_y_north - 2 * velocity_y_center + velocity_y_south
                                ) * hySquareInverse
                  - 2.0 / 3.0 * ( velocity_x_northEast - velocity_x_southEast - velocity_x_northWest + velocity_x_southWest
                                ) * hxInverse * hyInverse / 4
                  - 2.0 / 3.0 * ( velocity_z_upNorth - velocity_z_downNorth - velocity_z_upSouth + velocity_z_downSouth
                                ) * hyInverse * hzInverse / 4
                  ) * this->dynamicalViscosity
// T_32_y
                + ( ( velocity_z_upNorth - velocity_z_downNorth - velocity_z_upSouth + velocity_z_downSouth
                     ) * hyInverse * hzInverse / 4
                  + ( velocity_y_north - 2 * velocity_y_center + velocity_y_south
                    ) * hySquareInverse
                  ) * this->dynamicalViscosity;
      }

      /*template< typename MeshEntity >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const MeshEntity& entity ) const;

      template< typename MeshEntity, typename Vector, typename MatrixRow >
      __cuda_callable__
      void updateLinearSystem( const RealType& time,
                               const RealType& tau,
                               const MeshType& mesh,
                               const IndexType& index,
                               const MeshEntity& entity,
                               const MeshFunctionType& u,
                               Vector& b,
                               MatrixRow& matrixRow ) const;*/
};


} // namespace TNL

