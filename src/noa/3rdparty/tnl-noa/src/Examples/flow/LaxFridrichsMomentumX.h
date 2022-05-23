#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include "LaxFridrichsMomentumBase.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsMomentumX
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsMomentumX< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
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
      Real operator()( const MeshFunction& rho_u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 1, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities(); 

         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1 >(); 
         const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2 >();
 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east = neighborEntities.template getEntityIndex< 1 >(); 
         const IndexType& west = neighborEntities.template getEntityIndex< -1 >();

         const RealType& pressure_west = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east = this->pressure.template getData< DeviceType >()[ east ];

         const RealType& velocity_x_east = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_center = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ center ];
         
         return 1.0 / ( 2.0 * this->tau ) * this->artificialViscosity * ( rho_u[ west ]  + rho_u[ east ]  - 2.0 * rho_u[ center ] ) 
                - 0.5 * ( ( rho_u[ east ] * velocity_x_east + pressure_east ) 
                         -( rho_u[ west ] * velocity_x_west + pressure_west ) ) * hxInverse
// 1D T_11_x
                - 4.0 / 3.0 *( velocity_x_east - 2 * velocity_x_center + velocity_x_west
                             ) * hxSquareInverse
                * this->dynamicalViscosity;
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
class LaxFridrichsMomentumX< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
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
      Real operator()( const MeshFunction& rho_u,
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
         
         const RealType& pressure_west = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east = this->pressure.template getData< DeviceType >()[ east ];

         const RealType& velocity_x_east = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_x_north = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_x_south = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_x_center = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_x_southEast = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_x_southWest = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_x_northEast = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_x_northWest = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ northWest ]; 

         const RealType& velocity_y_north = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_y_center = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ center ];
         const RealType& velocity_y_southEast = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ southEast ];
         const RealType& velocity_y_southWest = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ southWest ];
         const RealType& velocity_y_northEast = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ northEast ];
         const RealType& velocity_y_northWest = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ northWest ];         
         
         return 1.0 / ( 4.0 * this->tau ) * this->artificialViscosity * ( rho_u[ west ] + rho_u[ east ] + rho_u[ south ] + rho_u[ north ] - 4.0 * rho_u[ center ] ) 
                - 0.5 * ( ( ( rho_u[ east ] * velocity_x_east + pressure_east )
                          - ( rho_u[ west ] * velocity_x_west + pressure_west ) ) * hxInverse
                        + ( ( rho_u[ north ] * velocity_y_north )
                          - ( rho_u[ south ] * velocity_y_south ) ) * hyInverse )
// 2D T_11_x
                + ( 4.0 / 3.0 * ( velocity_x_east - 2 * velocity_x_center + velocity_x_west 
                                ) * hxSquareInverse
                  - 2.0 / 3.0 * ( velocity_y_northEast - velocity_y_southEast - velocity_y_northWest + velocity_y_southWest 
                                ) * hxInverse * hyInverse / 4
                  ) * this->dynamicalViscosity 
// T_21_y
                + ( ( velocity_y_northEast - velocity_y_southEast - velocity_y_northWest + velocity_y_southWest
                    ) * hxInverse * hyInverse / 4
                  + ( velocity_x_north - 2 * velocity_x_center + velocity_x_south
                    ) * hxInverse * hyInverse
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
class LaxFridrichsMomentumX< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index >
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
      Real operator()( const MeshFunction& rho_u,
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
                   ( rho_u[ west ] + rho_u[ east ] + rho_u[ south ] + rho_u[ north ] + rho_u[ up ] + rho_u[ down ] - 6.0 * rho_u[ center ] ) 
                - 0.5 * ( ( ( rho_u[ east ] * velocity_x_east + pressure_east )
                          - ( rho_u[ west ] * velocity_x_west + pressure_west ) )* hxInverse
                        + ( ( rho_u[ north ] * velocity_y_north )
                          - ( rho_u[ south ] * velocity_y_south ) )* hyInverse
                        + ( ( rho_u[ up ] * velocity_z_up )
                          - ( rho_u[ down ] * velocity_z_down ) )* hzInverse )
// 3D T_11_x
                + ( 4.0 / 3.0 * ( velocity_x_east - 2 * velocity_x_center + velocity_x_west
                                ) * hxSquareInverse
                  - 2.0 / 3.0 * ( velocity_y_northEast - velocity_y_southEast - velocity_y_northWest + velocity_y_southWest
                                ) * hxInverse * hyInverse / 4
                  - 2.0 / 3.0 * ( velocity_z_upEast - velocity_z_downEast - velocity_z_upWest + velocity_z_downWest
                                ) * hxInverse * hzInverse / 4
                  ) * this->dynamicalViscosity
// T_21_x
                + ( ( velocity_y_northEast - velocity_y_southEast - velocity_y_northWest + velocity_y_southWest
                    ) * hxInverse * hyInverse / 4
                  + ( velocity_x_east - 2 * velocity_x_center + velocity_x_west
                    ) * hxSquareInverse
                  ) * this->dynamicalViscosity
// T_31_x
                + ( ( velocity_z_upEast - velocity_z_downEast - velocity_z_upWest + velocity_z_downWest
                    ) * hxInverse * hzInverse / 4
                  + ( velocity_x_east - 2 * velocity_x_center + velocity_x_west
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


} // namespace TNL

