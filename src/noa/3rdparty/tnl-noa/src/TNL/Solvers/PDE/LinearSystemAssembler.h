// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/FunctionAdapter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Traverser.h>

namespace noa::TNL {
namespace Solvers {
namespace PDE {

template< typename Real,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename DofVector,
          typename MatrixView >
class LinearSystemAssemblerTraverserUserData
{
public:
   Real time = 0.0;

   Real tau = 0.0;

   const DifferentialOperator* differentialOperator = NULL;

   const BoundaryConditions* boundaryConditions = NULL;

   const RightHandSide* rightHandSide = NULL;

   const MeshFunction* u = NULL;

   DofVector* b = NULL;

   MatrixView matrix;

   LinearSystemAssemblerTraverserUserData( MatrixView matrix )
   : time( 0.0 ), tau( 0.0 ), differentialOperator( NULL ), boundaryConditions( NULL ), rightHandSide( NULL ), u( NULL ),
     b( NULL ), matrix( matrix )
   {}
};

template< typename Mesh,
          typename MeshFunction,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RightHandSide,
          typename TimeDiscretisation,
          typename DofVector >
class LinearSystemAssembler
{
public:
   using MeshType = typename MeshFunction::MeshType;
   using MeshPointer = typename MeshFunction::MeshPointer;
   using RealType = typename MeshFunction::RealType;
   using DeviceType = typename MeshFunction::DeviceType;
   using IndexType = typename MeshFunction::IndexType;

   template< typename MatrixView >
   using TraverserUserData = LinearSystemAssemblerTraverserUserData< RealType,
                                                                     MeshFunction,
                                                                     DifferentialOperator,
                                                                     BoundaryConditions,
                                                                     RightHandSide,
                                                                     DofVector,
                                                                     MatrixView >;

   // typedef Pointers::SharedPointer<  Matrix, DeviceType > MatrixPointer;
   using DifferentialOperatorPointer = Pointers::SharedPointer< DifferentialOperator, DeviceType >;
   using BoundaryConditionsPointer = Pointers::SharedPointer< BoundaryConditions, DeviceType >;
   using RightHandSidePointer = Pointers::SharedPointer< RightHandSide, DeviceType >;
   using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction, DeviceType >;
   using DofVectorPointer = Pointers::SharedPointer< DofVector, DeviceType >;

   void
   setDifferentialOperator( const DifferentialOperatorPointer& differentialOperatorPointer )
   {
      this->differentialOperator = &differentialOperatorPointer.template getData< DeviceType >();
   }

   void
   setBoundaryConditions( const BoundaryConditionsPointer& boundaryConditionsPointer )
   {
      this->boundaryConditions = &boundaryConditionsPointer.template getData< DeviceType >();
   }

   void
   setRightHandSide( const RightHandSidePointer& rightHandSidePointer )
   {
      this->rightHandSide = &rightHandSidePointer.template getData< DeviceType >();
   }

   template< typename EntityType, typename Matrix >
   void
   assembly( const RealType& time,
             const RealType& tau,
             const MeshPointer& meshPointer,
             const MeshFunctionPointer& uPointer,
             std::shared_ptr< Matrix >& matrixPointer,
             DofVectorPointer& bPointer )
   {
      static_assert( std::is_same< MeshFunction,
                                   Containers::Vector< typename MeshFunction::RealType,
                                                       typename MeshFunction::DeviceType,
                                                       typename MeshFunction::IndexType > >::value
                        != true,
                     "Error: I am getting Vector instead of MeshFunction or similar object. You might forget to bind DofVector "
                     "into MeshFunction in you method getExplicitUpdate." );

      // const IndexType maxRowLength = matrixPointer.template getData< Devices::Host >().getMaxRowLength();
      // TNL_ASSERT_GT( maxRowLength, 0, "maximum row length must be positive" );
      TraverserUserData< typename Matrix::ViewType > userData( matrixPointer->getView() );
      userData.time = time;
      userData.tau = tau;
      userData.differentialOperator = differentialOperator;
      userData.boundaryConditions = boundaryConditions;
      userData.rightHandSide = rightHandSide;
      userData.u = &uPointer.template getData< DeviceType >();
      userData.matrix = matrixPointer->getView();
      userData.b = &bPointer.template modifyData< DeviceType >();
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserBoundaryEntitiesProcessor< typename Matrix::ViewType > >(
         meshPointer, userData );
      meshTraverser.template processInteriorEntities< TraverserInteriorEntitiesProcessor< typename Matrix::ViewType > >(
         meshPointer, userData );
   }

   template< typename Matrix >
   struct TraverserBoundaryEntitiesProcessor
   {
      template< typename EntityType >
      __cuda_callable__
      static void
      processEntity( const MeshType& mesh, TraverserUserData< Matrix >& userData, const EntityType& entity )
      {
         ( *userData.b )[ entity.getIndex() ] = 0.0;
         userData.boundaryConditions->setMatrixElements(
            *userData.u, entity, userData.time + userData.tau, userData.tau, userData.matrix, *userData.b );
      }
   };

   template< typename Matrix >
   struct TraverserInteriorEntitiesProcessor
   {
      template< typename EntityType >
      __cuda_callable__
      static void
      processEntity( const MeshType& mesh, TraverserUserData< Matrix >& userData, const EntityType& entity )
      {
         ( *userData.b )[ entity.getIndex() ] = 0.0;
         userData.differentialOperator->setMatrixElements(
            *userData.u, entity, userData.time + userData.tau, userData.tau, userData.matrix, *userData.b );

         using RhsFunctionAdapter = Functions::FunctionAdapter< MeshType, RightHandSide >;
         using MeshFunctionAdapter = Functions::FunctionAdapter< MeshType, MeshFunction >;
         const RealType& rhs = RhsFunctionAdapter::getValue( *userData.rightHandSide, entity, userData.time );
         TimeDiscretisation::applyTimeDiscretisation( userData.matrix,
                                                      ( *userData.b )[ entity.getIndex() ],
                                                      entity.getIndex(),
                                                      MeshFunctionAdapter::getValue( *userData.u, entity, userData.time ),
                                                      userData.tau,
                                                      rhs );
      }
   };

protected:
   const DifferentialOperator* differentialOperator = NULL;

   const BoundaryConditions* boundaryConditions = NULL;

   const RightHandSide* rightHandSide = NULL;
};

}  // namespace PDE
}  // namespace Solvers
}  // namespace noa::TNL
