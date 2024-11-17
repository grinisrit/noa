// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/MeshFunctionEvaluator.h>
//#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Traverser.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Functions {

template< typename OutMeshFunction, typename InFunction >
template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::evaluate( OutMeshFunctionPointer& meshFunction,
                                                                const InFunctionPointer& function,
                                                                const RealType& time,
                                                                const RealType& outFunctionMultiplicator,
                                                                const RealType& inFunctionMultiplicator )
{
   static_assert(
      std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value,
      "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value,
                  "expected a smart pointer" );

   switch( InFunction::getDomainType() ) {
      case NonspaceDomain:
      case SpaceDomain:
      case MeshDomain:
         evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
         break;
      case MeshInteriorDomain:
         evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
         break;
      case MeshBoundaryDomain:
         evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, boundary );
         break;
   }
}

template< typename OutMeshFunction, typename InFunction >
template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::evaluateAllEntities( OutMeshFunctionPointer& meshFunction,
                                                                           const InFunctionPointer& function,
                                                                           const RealType& time,
                                                                           const RealType& outFunctionMultiplicator,
                                                                           const RealType& inFunctionMultiplicator )
{
   static_assert(
      std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value,
      "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value,
                  "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
}

template< typename OutMeshFunction, typename InFunction >
template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::evaluateInteriorEntities( OutMeshFunctionPointer& meshFunction,
                                                                                const InFunctionPointer& function,
                                                                                const RealType& time,
                                                                                const RealType& outFunctionMultiplicator,
                                                                                const RealType& inFunctionMultiplicator )
{
   static_assert(
      std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value,
      "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value,
                  "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
}

template< typename OutMeshFunction, typename InFunction >
template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::evaluateBoundaryEntities( OutMeshFunctionPointer& meshFunction,
                                                                                const InFunctionPointer& function,
                                                                                const RealType& time,
                                                                                const RealType& outFunctionMultiplicator,
                                                                                const RealType& inFunctionMultiplicator )
{
   static_assert(
      std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value,
      "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value,
                  "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, boundary );
}

template< typename OutMeshFunction, typename InFunction >
template< typename OutMeshFunctionPointer, typename InFunctionPointer >
void
MeshFunctionEvaluator< OutMeshFunction, InFunction >::evaluateEntities( OutMeshFunctionPointer& meshFunction,
                                                                        const InFunctionPointer& function,
                                                                        const RealType& time,
                                                                        const RealType& outFunctionMultiplicator,
                                                                        const RealType& inFunctionMultiplicator,
                                                                        EntitiesType entitiesType )
{
   throw Exceptions::NotImplementedError( "MeshFunctionEvaluator is not implemented with the current Grid implementation" );
   /*
   static_assert(
      std::is_same< typename std::decay< typename OutMeshFunctionPointer::ObjectType >::type, OutMeshFunction >::value,
      "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InFunctionPointer::ObjectType >::type, InFunction >::value,
                  "expected a smart pointer" );

   using MeshEntityType = typename MeshType::template EntityType< OutMeshFunction::getEntitiesDimension() >;
   using AssignmentEntitiesProcessor =
      Functions::MeshFunctionEvaluatorAssignmentEntitiesProcessor< MeshType, TraverserUserData >;
   using AdditionEntitiesProcessor = Functions::MeshFunctionEvaluatorAdditionEntitiesProcessor< MeshType, TraverserUserData >;
   // typedef typename OutMeshFunction::MeshPointer OutMeshPointer;

   TraverserUserData userData( &function.template getData< DeviceType >(),
                               time,
                               &meshFunction.template modifyData< DeviceType >(),
                               outFunctionMultiplicator,
                               inFunctionMultiplicator );
   Meshes::Traverser< MeshType, MeshEntityType > meshTraverser;
   switch( entitiesType ) {
      case all:
         if( outFunctionMultiplicator )
            meshTraverser.template processAllEntities< AdditionEntitiesProcessor >( meshFunction->getMeshPointer(), userData );
         else
            meshTraverser.template processAllEntities< AssignmentEntitiesProcessor >( meshFunction->getMeshPointer(),
                                                                                      userData );
         break;
      case interior:
         if( outFunctionMultiplicator )
            meshTraverser.template processInteriorEntities< AdditionEntitiesProcessor >( meshFunction->getMeshPointer(),
                                                                                         userData );
         else
            meshTraverser.template processInteriorEntities< AssignmentEntitiesProcessor >( meshFunction->getMeshPointer(),
                                                                                           userData );
         break;
      case boundary:
         if( outFunctionMultiplicator )
            meshTraverser.template processBoundaryEntities< AdditionEntitiesProcessor >( meshFunction->getMeshPointer(),
                                                                                         userData );
         else
            meshTraverser.template processBoundaryEntities< AssignmentEntitiesProcessor >( meshFunction->getMeshPointer(),
                                                                                           userData );
         break;
   }*/
}

}  // namespace Functions
}  // namespace noa::TNL
