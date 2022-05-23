#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/VectorField.h>

#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsEnergy.h"
#include "LaxFridrichsMomentumX.h"
#include "LaxFridrichsMomentumY.h"
#include "LaxFridrichsMomentumZ.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichs
{
   public:
      typedef Mesh MeshType;
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< Mesh > MeshFunctionType;
      static const int Dimensions = Mesh::getMeshDimension();
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VectorFieldType;
 
      typedef LaxFridrichsContinuity< Mesh, Real, Index > ContinuityOperatorType;
      typedef LaxFridrichsMomentumX< Mesh, Real, Index > MomentumXOperatorType;
      typedef LaxFridrichsMomentumY< Mesh, Real, Index > MomentumYOperatorType;
      typedef LaxFridrichsMomentumZ< Mesh, Real, Index > MomentumZOperatorType;
      typedef LaxFridrichsEnergy< Mesh, Real, Index > EnergyOperatorType;

      typedef Pointers::SharedPointer< MeshFunctionType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< VectorFieldType > VectorFieldPointer;
      typedef Pointers::SharedPointer< MeshType > MeshPointer;
      
      typedef Pointers::SharedPointer< ContinuityOperatorType > ContinuityOperatorPointer;
      typedef Pointers::SharedPointer< MomentumXOperatorType > MomentumXOperatorPointer;
      typedef Pointers::SharedPointer< MomentumYOperatorType > MomentumYOperatorPointer;      
      typedef Pointers::SharedPointer< MomentumZOperatorType > MomentumZOperatorPointer;      
      typedef Pointers::SharedPointer< EnergyOperatorType > EnergyOperatorPointer;

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "numerical-viscosity", "Value of artificial (numerical) viscosity in the Lax-Fridrichs scheme", 1.0 );
         config.addEntry< double >( prefix + "dynamical-viscosity", "Value of dynamical (real) viscosity in the Navier-Stokes equation", 1.0 );
      }
      
      LaxFridrichs()
         : artificialViscosity( 1.0 ) {}
      
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->dynamicalViscosity = parameters.getParameter< double >( prefix + "dynamical-viscosity" );
         this->momentumXOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumYOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->momentumZOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->energyOperatorPointer->setDynamicalViscosity( dynamicalViscosity );
         this->artificialViscosity = parameters.getParameter< double >( prefix + "numerical-viscosity" );
         this->continuityOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumXOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumYOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->momentumZOperatorPointer->setArtificialViscosity( artificialViscosity );
         this->energyOperatorPointer->setArtificialViscosity( artificialViscosity );
         
         return true;
      }
      
      void setTau( const RealType& tau )
      {
         this->continuityOperatorPointer->setTau( tau );
         this->momentumXOperatorPointer->setTau( tau );
         this->momentumYOperatorPointer->setTau( tau );
         this->momentumZOperatorPointer->setTau( tau );
         this->energyOperatorPointer->setTau( tau );
      }
      
      void setPressure( const MeshFunctionPointer& pressure )
      {
         this->momentumXOperatorPointer->setPressure( pressure );
         this->momentumYOperatorPointer->setPressure( pressure );
         this->momentumZOperatorPointer->setPressure( pressure );
         this->energyOperatorPointer->setPressure( pressure );
      }
      
      void setVelocity( const VectorFieldPointer& velocity )
      {
         this->continuityOperatorPointer->setVelocity( velocity );
         this->momentumXOperatorPointer->setVelocity( velocity );
         this->momentumYOperatorPointer->setVelocity( velocity );
         this->momentumZOperatorPointer->setVelocity( velocity );
         this->energyOperatorPointer->setVelocity( velocity );
      }
      
      const ContinuityOperatorPointer& getContinuityOperator() const
      {
         return this->continuityOperatorPointer;
      }
      
      const MomentumXOperatorPointer& getMomentumXOperator() const
      {
         return this->momentumXOperatorPointer;
      }

      const MomentumYOperatorPointer& getMomentumYOperator() const
      {
         return this->momentumYOperatorPointer;
      }
      
      const MomentumZOperatorPointer& getMomentumZOperator() const
      {
         return this->momentumZOperatorPointer;
      }
      
      const EnergyOperatorPointer& getEnergyOperator() const
      {
         return this->energyOperatorPointer;
      }

   protected:
      
      ContinuityOperatorPointer continuityOperatorPointer;
      MomentumXOperatorPointer momentumXOperatorPointer;
      MomentumYOperatorPointer momentumYOperatorPointer;
      MomentumZOperatorPointer momentumZOperatorPointer;
      EnergyOperatorPointer energyOperatorPointer;  
      
      RealType artificialViscosity;
      RealType dynamicalViscosity;
};

} //namespace TNL
