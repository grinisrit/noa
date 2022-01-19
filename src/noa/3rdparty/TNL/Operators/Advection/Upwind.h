// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#ifndef Upwind_H
#define Upwind_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/Pointers/SharedPointer.h>

namespace TNL {
   namespace Operators {
      namespace Advection {
   

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType,
          typename VelocityFunction = Functions::MeshFunction< Mesh > >
class Upwind
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename VelocityFunction >
class Upwind< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimensions, VelocityFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer< VelocityFieldType, DeviceType > VelocityFieldPointer;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
      }
      
      Upwind()
         : artificialViscosity( 1.0 ) {}
      
      Upwind( const VelocityFieldPointer& velocityField )
         : artificialViscosity( 1.0 ), velocityField( velocityField ) {}
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->artificialViscosity = parameters.getParameter< double >( prefix + "numerical-viscosity" );
         return true;
      }

      void setViscosity(const Real& artificalViscosity)
      {
         this->artificialViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setVelocityField( const VelocityFieldPointer& velocityField )
      {
         this->velocityField = velocityField;
      }
      
      const VelocityFieldPointer& getVelocityField() const
      {
         return this->velocityField;
      }
      
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
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
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         const RealType& speedX = FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 0 ], entity, time );
         return ( - 0.5 * ( speedX + std::abs(speedX) ) * ( u[ center ] - u[ west ] ) * hxInverse ) 
                - ( 0.5 * ( speedX - std::abs(speedX) ) * ( u[ east ] - u[ center ] ) * hxInverse ) ;
      }
      
   protected:
            
      RealType tau;
      
      RealType artificialViscosity;
      
      VelocityFieldPointer velocityField;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename VelocityFunction >
class Upwind< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimensions, VelocityFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer< VelocityFieldType, DeviceType > VelocityFieldPointer;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
      }      
      
      Upwind()
         : artificialViscosity( 1.0 ) {}

      Upwind( const VelocityFieldPointer& velocityField )
         : artificialViscosity( 1.0 ), velocityField( velocityField ) {}      
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->artificialViscosity = parameters.getParameter< double >( prefix + "numerical-viscosity" );
         return true;
      }

      void setViscosity(const Real& artificalViscosity)
      {
         this->artificialViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setVelocityField( const VelocityFieldPointer& velocityField )
      {
         this->velocityField = velocityField;
      }
      
      const VelocityFieldPointer& getVelocityField() const
      {
         return this->velocityField;
      }
      
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
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
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         const RealType& speedX = FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 0 ]);
         const RealType& speedY = FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 1 ]);
         
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         return ( - 0.5 * ( speedX + std::abs(speedX) ) * ( u[ center ] - u[ west ] ) * hxInverse ) 
                - ( 0.5 * ( speedX - std::abs(speedX) ) * ( u[ east ] - u[ center ] ) * hxInverse )
                - ( 0.5 * ( speedY + std::abs(speedY) ) * ( u[ center ] - u[ south ] ) * hyInverse )
                - ( 0.5 * ( speedY - std::abs(speedY) ) * ( u[ north ] - u[ center ] ) * hyInverse );     
      }
      
   protected:
            
      RealType tau;
      
      RealType artificialViscosity;
      
      VelocityFieldPointer velocityField;
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename VelocityFunction >
class Upwind< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      static const int Dimensions = MeshType::getMeshDimension();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimensions, VelocityFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer< VelocityFieldType, DeviceType > VelocityFieldPointer;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
      }      
      
      Upwind()
         : artificialViscosity( 1.0 ) {}

      Upwind( const VelocityFieldPointer& velocityField )
         : artificialViscosity( 1.0 ), velocityField( velocityField ) {}
            
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->artificialViscosity = parameters.getParameter< double >( prefix + "numerical-viscosity" );
         return true;
      }

      void setViscosity(const Real& artificalViscosity)
      {
         this->artificialViscosity = artificalViscosity;
      }
      
      void setTau(const Real& tau)
      {
          this->tau = tau;
      };

      void setVelocityField( const VelocityFieldPointer& velocityField )
      {
         this->velocityField = velocityField;
      }
      
      const VelocityFieldPointer& getVelocityField() const
      {
         return this->velocityField;
      }
      
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const RealType& time = 0.0 ) const
      {
         static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimensions." ); 
         static_assert( MeshFunction::getEntitiesDimension() == 3, "Wrong preimage function" ); 
         const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 

         const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1,  0,  0 >(); 
         const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts<  0, -1,  0 >(); 
         const RealType& hzInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -1 >(); 
         const IndexType& center = entity.getIndex();
         const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
         const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
         const IndexType& north = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
         const IndexType& south = neighborEntities.template getEntityIndex<  0, -1,  0 >(); 
         const IndexType& up    = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
         const IndexType& down  = neighborEntities.template getEntityIndex<  0,  0, -1 >();
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         const RealType& speedX = FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 0 ]);
         const RealType& speedY = FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 1 ]);
         const RealType& speedZ = FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 2 ]); 
         
         typedef Functions::FunctionAdapter< MeshType, VelocityFunctionType > FunctionAdapter;
         return ( - 0.5 * ( speedX + std::abs(speedX) ) * ( u[ center ] - u[ west ] ) * hxInverse ) 
                - ( 0.5 * ( speedX - std::abs(speedX) ) * ( u[ east ] - u[ center ] ) * hxInverse )
                - ( 0.5 * ( speedY + std::abs(speedY) ) * ( u[ center ] - u[ south ] ) * hyInverse )
                - ( 0.5 * ( speedY - std::abs(speedY) ) * ( u[ north ] - u[ center ] ) * hyInverse )
                - ( 0.5 * ( speedZ + std::abs(speedZ) ) * ( u[ center ] - u[ down ] ) * hyInverse )
                - ( 0.5 * ( speedZ - std::abs(speedZ) ) * ( u[ up ] - u[ center ] ) * hyInverse );    
      }
      
   protected:
            
      RealType tau;
      
      RealType artificialViscosity;
      
      VelocityFieldPointer velocityField;
};

      }// namespace Advection
   } // namepsace Operators
} // namespace TNL

#endif	/* Upwind_H */
