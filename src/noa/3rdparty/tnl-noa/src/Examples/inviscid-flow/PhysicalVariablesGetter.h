#pragma once

#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/Functions/MeshFunctionEvaluator.h>
#include "CompressibleConservativeVariables.h"

namespace TNL {
   
template< typename Mesh >
class PhysicalVariablesGetter
{
   public:
      
      typedef Mesh MeshType;
      typedef typename MeshType::RealType RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef typename MeshType::IndexType IndexType;
      static const int Dimensions = MeshType::getMeshDimension();
      
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      typedef Pointers::SharedPointer<  MeshFunctionType > MeshFunctionPointer;
      typedef CompressibleConservativeVariables< MeshType > ConservativeVariablesType;
      typedef Pointers::SharedPointer<  ConservativeVariablesType > ConservativeVariablesPointer;
      typedef Functions::VectorField< Dimensions, MeshFunctionType > VelocityFieldType;
      typedef Pointers::SharedPointer<  VelocityFieldType > VelocityFieldPointer;
      
      class VelocityGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            VelocityGetter( MeshFunctionPointer density, 
                            MeshFunctionPointer momentum )
            : density( density ), momentum( momentum ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                        const RealType& time = 0.0 ) const
            {
               if( density.template getData< DeviceType >()( meshEntity ) == 0.0 )
                  return 0;
               else
                  return momentum.template getData< DeviceType >()( meshEntity ) / 
                         density.template getData< DeviceType >()( meshEntity );
            }
            
         protected:
            const MeshFunctionPointer density, momentum;
      };
      
      class PressureGetter : public Functions::Domain< Dimensions, Functions::MeshDomain >
      {
         public:
            typedef typename MeshType::RealType RealType;
            
            PressureGetter( MeshFunctionPointer density,
                            MeshFunctionPointer energy, 
                            VelocityFieldPointer momentum,
                            const RealType& gamma )
            : density( density ), energy( energy ), momentum( momentum ), gamma( gamma ) {}
            
            template< typename EntityType >
            __cuda_callable__
            RealType operator()( const EntityType& meshEntity,
                                 const RealType& time = 0.0 ) const
            {
               const RealType e = energy.template getData< DeviceType >()( meshEntity );
               const RealType rho = density.template getData< DeviceType >()( meshEntity );
               const RealType momentumNorm = lpNorm( momentum.template getData< DeviceType >().getVector( meshEntity ), 2.0 );
               if( rho == 0.0 )
                  return 0;
               else
                  return ( gamma - 1.0 ) * ( e - 0.5 * momentumNorm * momentumNorm / rho );
            }
            
         protected:
            const MeshFunctionPointer density, energy;
            const VelocityFieldPointer momentum;
            const RealType gamma;
      };      

      
      void getVelocity( const ConservativeVariablesPointer& conservativeVariables,
                        VelocityFieldPointer& velocity )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, VelocityGetter > evaluator;
         for( int i = 0; i < Dimensions; i++ )
         {
            Pointers::SharedPointer<  VelocityGetter, DeviceType > velocityGetter( conservativeVariables->getDensity(),
                                                                        ( *conservativeVariables->getMomentum() )[ i ] );
            evaluator.evaluate( ( *velocity )[ i ], velocityGetter );
         }
      }
      
      void getPressure( const ConservativeVariablesPointer& conservativeVariables,
                        const RealType& gamma,
                        MeshFunctionPointer& pressure )
      {
         Functions::MeshFunctionEvaluator< MeshFunctionType, PressureGetter > evaluator;
         Pointers::SharedPointer<  PressureGetter, DeviceType > pressureGetter( conservativeVariables->getDensity(),
                                                                     conservativeVariables->getEnergy(),
                                                                     conservativeVariables->getMomentum(),
                                                                     gamma );
         evaluator.evaluate( pressure, pressureGetter );
      }
      
};
   
} //namespace TNL
