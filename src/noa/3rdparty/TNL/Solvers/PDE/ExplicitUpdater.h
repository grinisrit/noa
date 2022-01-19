// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Functions/FunctionAdapter.h>
#include <TNL/Timer.h>
#include <TNL/Pointers/SharedPointer.h>
#include <type_traits>
#include <TNL/Meshes/GridDetails/Traverser_Grid1D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid2D.h>
#include <TNL/Meshes/GridDetails/Traverser_Grid3D.h>

namespace TNL {
namespace Solvers {
namespace PDE {

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class ExplicitUpdaterTraverserUserData
{
   public:

      Real time;

      const DifferentialOperator* differentialOperator;

      const BoundaryConditions* boundaryConditions;

      const RightHandSide* rightHandSide;

      MeshFunction *u, *fu;

      ExplicitUpdaterTraverserUserData()
      : time( 0.0 ),
        differentialOperator( NULL ),
        boundaryConditions( NULL ),
        rightHandSide( NULL ),
        u( NULL ),
        fu( NULL )
      {}


      /*void setUserData( const Real& time,
                        const DifferentialOperator* differentialOperator,
                        const BoundaryConditions* boundaryConditions,
                        const RightHandSide* rightHandSide,
                        MeshFunction* u,
                        MeshFunction* fu )
      {
         this->time = time;
         this->differentialOperator = differentialOperator;
         this->boundaryConditions = boundaryConditions;
         this->rightHandSide = rightHandSide;
         this->u = u;
         this->fu = fu;
      }*/
};


template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide >
class ExplicitUpdater
{
   public:
      typedef Mesh MeshType;
      typedef Pointers::SharedPointer<  MeshType > MeshPointer;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef ExplicitUpdaterTraverserUserData< RealType,
                                                MeshFunction,
                                                DifferentialOperator,
                                                BoundaryConditions,
                                                RightHandSide > TraverserUserData;
      typedef Pointers::SharedPointer<  DifferentialOperator, DeviceType > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer<  BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef Pointers::SharedPointer<  RightHandSide, DeviceType > RightHandSidePointer;
      typedef Pointers::SharedPointer<  MeshFunction, DeviceType > MeshFunctionPointer;
      typedef Pointers::SharedPointer<  TraverserUserData, DeviceType > TraverserUserDataPointer;

      void setDifferentialOperator( const DifferentialOperatorPointer& differentialOperatorPointer )
      {
         this->userData.differentialOperator = &differentialOperatorPointer.template getData< DeviceType >();
      }

      void setBoundaryConditions( const BoundaryConditionsPointer& boundaryConditionsPointer )
      {
         this->userData.boundaryConditions = &boundaryConditionsPointer.template getData< DeviceType >();
      }

      void setRightHandSide( const RightHandSidePointer& rightHandSidePointer )
      {
         this->userData.rightHandSide = &rightHandSidePointer.template getData< DeviceType >();
      }

      template< typename EntityType >
      void update( const RealType& time,
                   const RealType& tau,
                   const MeshPointer& meshPointer,
                   MeshFunctionPointer& uPointer,
                   MeshFunctionPointer& fuPointer )
      {
         static_assert( std::is_same< MeshFunction,
                                      Containers::Vector< typename MeshFunction::RealType,
                                                 typename MeshFunction::DeviceType,
                                                 typename MeshFunction::IndexType > >::value != true,
            "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector into MeshFunction in you method getExplicitUpdate."  );
         TNL_ASSERT_GT( uPointer->getData().getSize(), 0, "The first MeshFunction in the parameters was not bound." );
         TNL_ASSERT_GT( fuPointer->getData().getSize(), 0, "The second MeshFunction in the parameters was not bound." );

         TNL_ASSERT_EQ( uPointer->getData().getSize(), meshPointer->template getEntitiesCount< EntityType >(),
                        "The first MeshFunction in the parameters was not bound properly." );
         TNL_ASSERT_EQ( fuPointer->getData().getSize(), meshPointer->template getEntitiesCount< EntityType >(),
                        "The second MeshFunction in the parameters was not bound properly." );

         TNL_ASSERT_TRUE( this->userData.differentialOperator,
                          "The differential operator is not correctly set-up. Use method setDifferentialOperator() to do it." );
         TNL_ASSERT_TRUE( this->userData.rightHandSide,
                          "The right-hand side is not correctly set-up. Use method setRightHandSide() to do it." );

         this->userData.time = time;
         this->userData.u = &uPointer.template modifyData< DeviceType >();
         this->userData.fu = &fuPointer.template modifyData< DeviceType >();
         Meshes::Traverser< MeshType, EntityType > meshTraverser;
         meshTraverser.template processInteriorEntities< TraverserInteriorEntitiesProcessor >
                                                       ( meshPointer,
                                                         userData );
      }

      template< typename EntityType >
      void applyBoundaryConditions( const MeshPointer& meshPointer,
                                    const RealType& time,
                                    MeshFunctionPointer& uPointer )
      {
         TNL_ASSERT_TRUE( this->userData.boundaryConditions,
                          "The boundary conditions are not correctly set-up. Use method setBoundaryCondtions() to do it." );
         TNL_ASSERT_TRUE( &uPointer.template modifyData< DeviceType >(),
                          "The function u is not correctly set-up. It was not bound probably with DOFs." );

         this->userData.time = time;
         this->userData.u = &uPointer.template modifyData< DeviceType >();
         Meshes::Traverser< MeshType, EntityType > meshTraverser;
         meshTraverser.template processBoundaryEntities< TraverserBoundaryEntitiesProcessor >
                                           ( meshPointer,
                                             userData );

         // TODO: I think that this is not necessary
         /*if( MPI::GetSize() > 1 )
            fuPointer->template synchronize();*/

      }

      class TraverserBoundaryEntitiesProcessor
      {
         public:

            template< typename GridEntity >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const GridEntity& entity )
            {
               ( *userData.u )( entity ) = ( *userData.boundaryConditions )
                  ( *userData.u, entity, userData.time );
            }

      };

      class TraverserInteriorEntitiesProcessor
      {
         public:

            typedef typename MeshType::PointType PointType;

            template< typename EntityType >
            __cuda_callable__
            static inline void processEntity( const MeshType& mesh,
                                              TraverserUserData& userData,
                                              const EntityType& entity )
            {
               typedef Functions::FunctionAdapter< MeshType, RightHandSide > FunctionAdapter;
               ( *userData.fu )( entity ) =
                  ( *userData.differentialOperator )( *userData.u, entity, userData.time )
                  + FunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );

            }
      };

   protected:

      TraverserUserData userData;

};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

