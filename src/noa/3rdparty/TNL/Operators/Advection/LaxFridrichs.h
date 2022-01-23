// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#ifndef LaxFridrichs_H
#define LaxFridrichs_H

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Functions/VectorField.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>

namespace noa::TNL {
   namespace Operators {
      namespace Advection {
   

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType,
          typename VelocityFunction = Functions::MeshFunction< Mesh > >
class LaxFridrichs
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          typename VelocityFunction >
class LaxFridrichs< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef Pointers::SharedPointer<  MeshType > MeshPointer;
      static const int Dimension = MeshType::getMeshDimension();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimension, VelocityFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer<  VelocityFieldType, DeviceType > VelocityFieldPointer;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
      }
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}
      
      LaxFridrichs( const VelocityFieldPointer& velocityField )
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
         return ( 0.5 / this->tau ) * this->artificialViscosity * ( u[ west ]- 2.0 * u[ center ] + u[ east ] ) -
                FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 0 ], entity, time ) * ( u[ east ] - u[ west ] ) * hxInverse * 0.5;
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
class LaxFridrichs< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Pointers::SharedPointer<  MeshType > MeshPointer;
      static const int Dimension = MeshType::getMeshDimension();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimension, VelocityFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer<  VelocityFieldType, DeviceType > VelocityFieldPointer;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
      }      
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}

      LaxFridrichs( const VelocityFieldPointer& velocityField )
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
         return ( 0.25 / this->tau ) * this->artificialViscosity * ( u[ west ] + u[ east ] + u[ north ] + u[ south ] - 4.0 * u[ center ] ) -
                0.5 * ( FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 0 ], entity, time ) * ( u[ east ] - u[ west ] ) * hxInverse +
                        FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 1 ], entity, time ) * ( u[ north ] - u[ south ] ) * hyInverse );         
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
class LaxFridrichs< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index, VelocityFunction >
{
   public:
      
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Pointers::SharedPointer<  MeshType > MeshPointer;
      static const int Dimension = MeshType::getMeshDimension();
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunction< MeshType > MeshFunctionType;
      typedef VelocityFunction VelocityFunctionType;
      typedef Functions::VectorField< Dimension, VelocityFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer<  VelocityFieldType, DeviceType > VelocityFieldPointer;
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
      }      
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}

      LaxFridrichs( const VelocityFieldPointer& velocityField )
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
         return ( 0.25 / this->tau ) * this->artificialViscosity * ( u[ west ] + u[ east ] + u[ north ] + u[ south ] + u[ up ] + u[ down ] - 6.0 * u[ center ] ) -
                0.5 * ( FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 0 ], entity, time ) * ( u[ east ] - u[ west ] ) * hxInverse +
                        FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 1 ], entity, time ) * ( u[ north ] - u[ south ] ) * hyInverse +
                        FunctionAdapter::getValue( this->velocityField.template getData< DeviceType >()[ 2 ], entity, time ) * ( u[ up ] - u[ down ] ) * hzInverse );         
      }
      
   protected:
            
      RealType tau;
      
      RealType artificialViscosity;
      
      VelocityFieldPointer velocityField;
};

      }// namespace Advection
   } // namepsace Operators
} // namespace noa::TNL

#endif	/* LaxFridrichs_H */
