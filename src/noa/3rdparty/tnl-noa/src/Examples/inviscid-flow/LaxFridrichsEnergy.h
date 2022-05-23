#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsEnergyBase
{
   public:
      
      typedef Real RealType;
      typedef Index IndexType;
      typedef Mesh MeshType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer<  MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer<  VelocityFieldType > VelocityFieldPointer;
      
      LaxFridrichsEnergyBase()
       : artificialViscosity( 1.0 ){};

      void setTau(const Real& tau)
      {
          this->tau = tau;
      };
      
      void setVelocity( const VelocityFieldPointer& velocity )
      {
          this->velocity = velocity;
      };
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
          this->pressure = pressure;
      };
      
      void setArtificialViscosity( const RealType& artificialViscosity )
      {
         this->artificialViscosity = artificialViscosity;
      }      

      protected:
         
         RealType tau;
         
         VelocityFieldPointer velocity;
         
         MeshFunctionPointer pressure;
         
         RealType artificialViscosity;
};
   
template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichsEnergy
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class LaxFridrichsEnergy< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsEnergyBase< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:

      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsEnergyBase< MeshType, Real, Index > BaseType;
      
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
      Real operator()( const MeshFunction& e,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 1, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities(); 

         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1 >(); 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east = neighborEntities.template getEntityIndex< 1 >(); 
         const IndexType& west = neighborEntities.template getEntityIndex< -1 >();
         const RealType& pressure_west = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& velocity_x_east = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         return 1.0 / ( 2.0 * this->tau ) * this->artificialViscosity * ( e[ west ] - 2.0 * e[ center ]  + e[ east ] ) 
                - 0.5 * ( ( e[ east ] + pressure_east ) * velocity_x_east  
                        - ( e[ west ] + pressure_west ) * velocity_x_west ) * hxInverse;
  
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
class LaxFridrichsEnergy< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsEnergyBase< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsEnergyBase< MeshType, Real, Index > BaseType;
      
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
      Real operator()( const MeshFunction& e,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 2, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 
 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1 >(); 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0 >(); 
         const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0 >(); 
         const IndexType& north = neighborEntities.template getEntityIndex<  0,  1 >(); 
         const IndexType& south = neighborEntities.template getEntityIndex<  0, -1 >();
         const RealType& pressure_west = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& pressure_north = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south = this->pressure.template getData< DeviceType >()[ south ];
         const RealType& velocity_x_east = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_north = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];         
         
         return 1.0 / ( 4.0 * this->tau ) * this->artificialViscosity * ( e[ west ] + e[ east ] + e[ south ] + e[ north ] - 4.0 * e[ center ] ) 
                - 0.5 * ( ( ( ( e[ east ] + pressure_east ) * velocity_x_east )
                          -( ( e[ west ] + pressure_west ) * velocity_x_west ) ) * hxInverse
                        + ( ( ( e[ north ] + pressure_north ) * velocity_y_north )
                          -( ( e[ south ] + pressure_south ) * velocity_y_south ) ) * hyInverse );
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
class LaxFridrichsEnergy< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
   : public LaxFridrichsEnergyBase< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public:
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef LaxFridrichsEnergyBase< MeshType, Real, Index > BaseType;
      
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
      Real operator()( const MeshFunction& e,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 3, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 
 
         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts< 0,  0, -1 >(); 
         const IndexType& center = entity.getIndex(); 
         const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south = neighborEntities.template getEntityIndex<  0, -1,  0 >();
         const IndexType& up    = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down  = neighborEntities.template getEntityIndex<  0,  0, -1 >();
         
         const RealType& pressure_west  = this->pressure.template getData< DeviceType >()[ west ];
         const RealType& pressure_east  = this->pressure.template getData< DeviceType >()[ east ];
         const RealType& pressure_north = this->pressure.template getData< DeviceType >()[ north ];
         const RealType& pressure_south = this->pressure.template getData< DeviceType >()[ south ];
         const RealType& pressure_up    = this->pressure.template getData< DeviceType >()[ up ];
         const RealType& pressure_down  = this->pressure.template getData< DeviceType >()[ down ];
         
         const RealType& velocity_x_east  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ east ];
         const RealType& velocity_x_west  = this->velocity.template getData< DeviceType >()[ 0 ].template getData< DeviceType >()[ west ];
         const RealType& velocity_y_north = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ north ];
         const RealType& velocity_y_south = this->velocity.template getData< DeviceType >()[ 1 ].template getData< DeviceType >()[ south ];
         const RealType& velocity_z_up    = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ up ];
         const RealType& velocity_z_down  = this->velocity.template getData< DeviceType >()[ 2 ].template getData< DeviceType >()[ down ];         
         
         return 1.0 / ( 6.0 * this->tau ) * this->artificialViscosity *
                 ( e[ west ] + e[ east ] + e[ south ] + e[ north ] + e[ up ] + e[ down ] - 6.0 * e[ center ] ) 
                - 0.5 * ( ( ( ( e[ east ] + pressure_east ) * velocity_x_east )
                           -( ( e[ west ] + pressure_west ) * velocity_x_west ) ) * hxInverse
                        + ( ( ( e[ north ] + pressure_north ) * velocity_y_north )
                           -( ( e[ south ] + pressure_south ) * velocity_y_south ) ) * hyInverse
                        + ( ( ( e[ up ] + pressure_up ) * velocity_z_up )
                           -( ( e[ down ] + pressure_down ) * velocity_z_down ) ) * hzInverse );
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

} //namespace TNL
