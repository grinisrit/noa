// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/FunctionAdapter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointer.h>
#include <type_traits>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Traverser_Grid1D.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Traverser_Grid2D.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Traverser_Grid3D.h>

namespace noa::TNL {
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
   : time( 0.0 ), differentialOperator( NULL ), boundaryConditions( NULL ), rightHandSide( NULL ), u( NULL ), fu( NULL )
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
   using MeshType = Mesh;
   using MeshPointer = Pointers::SharedPointer< MeshType >;
   using RealType = typename MeshFunction::RealType;
   using DeviceType = typename MeshFunction::DeviceType;
   using IndexType = typename MeshFunction::IndexType;
   using TraverserUserData =
      ExplicitUpdaterTraverserUserData< RealType, MeshFunction, DifferentialOperator, BoundaryConditions, RightHandSide >;
   using DifferentialOperatorPointer = Pointers::SharedPointer< DifferentialOperator, DeviceType >;
   using BoundaryConditionsPointer = Pointers::SharedPointer< BoundaryConditions, DeviceType >;
   using RightHandSidePointer = Pointers::SharedPointer< RightHandSide, DeviceType >;
   using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction, DeviceType >;
   using TraverserUserDataPointer = Pointers::SharedPointer< TraverserUserData, DeviceType >;

   void
   setDifferentialOperator( const DifferentialOperatorPointer& differentialOperatorPointer )
   {
      this->userData.differentialOperator = &differentialOperatorPointer.template getData< DeviceType >();
   }

   void
   setBoundaryConditions( const BoundaryConditionsPointer& boundaryConditionsPointer )
   {
      this->userData.boundaryConditions = &boundaryConditionsPointer.template getData< DeviceType >();
   }

   void
   setRightHandSide( const RightHandSidePointer& rightHandSidePointer )
   {
      this->userData.rightHandSide = &rightHandSidePointer.template getData< DeviceType >();
   }

   template< typename EntityType >
   void
   update( const RealType& time,
           const RealType& tau,
           const MeshPointer& meshPointer,
           MeshFunctionPointer& uPointer,
           MeshFunctionPointer& fuPointer )
   {
      static_assert( std::is_same< MeshFunction,
                                   Containers::Vector< typename MeshFunction::RealType,
                                                       typename MeshFunction::DeviceType,
                                                       typename MeshFunction::IndexType > >::value
                        != true,
                     "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector "
                     "into MeshFunction in you method getExplicitUpdate." );
      TNL_ASSERT_GT( uPointer->getData().getSize(), 0, "The first MeshFunction in the parameters was not bound." );
      TNL_ASSERT_GT( fuPointer->getData().getSize(), 0, "The second MeshFunction in the parameters was not bound." );

      TNL_ASSERT_EQ( uPointer->getData().getSize(),
                     meshPointer->template getEntitiesCount< EntityType >(),
                     "The first MeshFunction in the parameters was not bound properly." );
      TNL_ASSERT_EQ( fuPointer->getData().getSize(),
                     meshPointer->template getEntitiesCount< EntityType >(),
                     "The second MeshFunction in the parameters was not bound properly." );

      TNL_ASSERT_TRUE( this->userData.differentialOperator,
                       "The differential operator is not correctly set-up. Use method setDifferentialOperator() to do it." );
      TNL_ASSERT_TRUE( this->userData.rightHandSide,
                       "The right-hand side is not correctly set-up. Use method setRightHandSide() to do it." );

      this->userData.time = time;
      this->userData.u = &uPointer.template modifyData< DeviceType >();
      this->userData.fu = &fuPointer.template modifyData< DeviceType >();
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processInteriorEntities< TraverserInteriorEntitiesProcessor >( meshPointer, userData );
   }

   template< typename EntityType >
   void
   applyBoundaryConditions( const MeshPointer& meshPointer, const RealType& time, MeshFunctionPointer& uPointer )
   {
      TNL_ASSERT_TRUE( this->userData.boundaryConditions,
                       "The boundary conditions are not correctly set-up. Use method setBoundaryCondtions() to do it." );
      TNL_ASSERT_TRUE( &uPointer.template modifyData< DeviceType >(),
                       "The function u is not correctly set-up. It was not bound probably with DOFs." );

      this->userData.time = time;
      this->userData.u = &uPointer.template modifyData< DeviceType >();
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserBoundaryEntitiesProcessor >( meshPointer, userData );

      // TODO: I think that this is not necessary
      /*if( MPI::GetSize() > 1 )
         fuPointer->template synchronize();*/
   }

   class TraverserBoundaryEntitiesProcessor
   {
   public:
      template< typename GridEntity >
      __cuda_callable__
      static inline void
      processEntity( const MeshType& mesh, TraverserUserData& userData, const GridEntity& entity )
      {
         ( *userData.u )( entity ) = ( *userData.boundaryConditions )( *userData.u, entity, userData.time );
      }
   };

   class TraverserInteriorEntitiesProcessor
   {
   public:
      using PointType = typename MeshType::PointType;

      template< typename EntityType >
      __cuda_callable__
      static inline void
      processEntity( const MeshType& mesh, TraverserUserData& userData, const EntityType& entity )
      {
         using FunctionAdapter = Functions::FunctionAdapter< MeshType, RightHandSide >;
         ( *userData.fu )( entity ) = ( *userData.differentialOperator )( *userData.u, entity, userData.time )
                                    + FunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );
      }
   };

protected:
   TraverserUserData userData;
};

}  // namespace PDE
}  // namespace Solvers
}  // namespace noa::TNL
