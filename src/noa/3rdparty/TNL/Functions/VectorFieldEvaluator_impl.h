// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Functions/VectorFieldEvaluator.h>
#include <noa/3rdparty/TNL/Meshes/Traverser.h>

namespace noa::TNL {
namespace Functions {   

template< typename OutVectorField,
          typename InVectorField >
   template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
void
VectorFieldEvaluator< OutVectorField, InVectorField >::
evaluate( OutVectorFieldPointer& meshFunction,
          const InVectorFieldPointer& function,
          const RealType& time,
          const RealType& outFunctionMultiplicator,
          const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutVectorFieldPointer::ObjectType >::type, OutVectorField >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InVectorFieldPointer::ObjectType >::type, InVectorField >::value, "expected a smart pointer" );

   switch( InVectorField::getDomainType() )
   {
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


template< typename OutVectorField,
          typename InVectorField >
   template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
void
VectorFieldEvaluator< OutVectorField, InVectorField >::
evaluateAllEntities( OutVectorFieldPointer& meshFunction,
                     const InVectorFieldPointer& function,
                     const RealType& time,
                     const RealType& outFunctionMultiplicator,
                     const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutVectorFieldPointer::ObjectType >::type, OutVectorField >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InVectorFieldPointer::ObjectType >::type, InVectorField >::value, "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, all );
}

template< typename OutVectorField,
          typename InVectorField >
   template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
void
VectorFieldEvaluator< OutVectorField, InVectorField >::
evaluateInteriorEntities( OutVectorFieldPointer& meshFunction,
                          const InVectorFieldPointer& function,
                          const RealType& time,
                          const RealType& outFunctionMultiplicator,
                          const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutVectorFieldPointer::ObjectType >::type, OutVectorField >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InVectorFieldPointer::ObjectType >::type, InVectorField >::value, "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, interior );
}

template< typename OutVectorField,
          typename InVectorField >
   template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
void
VectorFieldEvaluator< OutVectorField, InVectorField >::
evaluateBoundaryEntities( OutVectorFieldPointer& meshFunction,
                          const InVectorFieldPointer& function,
                          const RealType& time,
                          const RealType& outFunctionMultiplicator,
                          const RealType& inFunctionMultiplicator )
{
   static_assert( std::is_same< typename std::decay< typename OutVectorFieldPointer::ObjectType >::type, OutVectorField >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InVectorFieldPointer::ObjectType >::type, InVectorField >::value, "expected a smart pointer" );

   return evaluateEntities( meshFunction, function, time, outFunctionMultiplicator, inFunctionMultiplicator, boundary );
}



template< typename OutVectorField,
          typename InVectorField >
   template< typename OutVectorFieldPointer, typename InVectorFieldPointer >
void
VectorFieldEvaluator< OutVectorField, InVectorField >::
evaluateEntities( OutVectorFieldPointer& meshFunction,
                  const InVectorFieldPointer& function,
                  const RealType& time,
                  const RealType& outFunctionMultiplicator,
                  const RealType& inFunctionMultiplicator,
                  EntitiesType entitiesType )
{
   static_assert( std::is_same< typename std::decay< typename OutVectorFieldPointer::ObjectType >::type, OutVectorField >::value, "expected a smart pointer" );
   static_assert( std::is_same< typename std::decay< typename InVectorFieldPointer::ObjectType >::type, InVectorField >::value, "expected a smart pointer" );

   typedef typename MeshType::template EntityType< OutVectorField::getEntitiesDimension() > MeshEntityType;
   typedef Functions::VectorFieldEvaluatorAssignmentEntitiesProcessor< MeshType, TraverserUserData > AssignmentEntitiesProcessor;
   typedef Functions::VectorFieldEvaluatorAdditionEntitiesProcessor< MeshType, TraverserUserData > AdditionEntitiesProcessor;
   //typedef typename OutVectorField::MeshPointer OutMeshPointer;
   typedef Pointers::SharedPointer< TraverserUserData, DeviceType > TraverserUserDataPointer;
   
   SharedPointer< TraverserUserData, DeviceType >
      userData( &function.template getData< DeviceType >(),
                time,
                &meshFunction.template modifyData< DeviceType >(),
                outFunctionMultiplicator,
                inFunctionMultiplicator );
   Meshes::Traverser< MeshType, MeshEntityType > meshTraverser;
   switch( entitiesType )
   {
      case all:
         if( outFunctionMultiplicator )
            meshTraverser.template processAllEntities< TraverserUserData,
                                                       AdditionEntitiesProcessor >
                                                     ( meshFunction->getMeshPointer(),
                                                       userData );
         else
            meshTraverser.template processAllEntities< TraverserUserData,
                                                       AssignmentEntitiesProcessor >
                                                    ( meshFunction->getMeshPointer(),
                                                      userData );
         break;
      case interior:
         if( outFunctionMultiplicator )
            meshTraverser.template processInteriorEntities< TraverserUserData,
                                                            AdditionEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         else
            meshTraverser.template processInteriorEntities< TraverserUserData,
                                                            AssignmentEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         break;
      case boundary:
         if( outFunctionMultiplicator )
            meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                            AdditionEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         else
            meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                            AssignmentEntitiesProcessor >
                                                          ( meshFunction->getMeshPointer(),
                                                            userData );
         break;
   }
}

} // namespace Functions
} // namespace noa::TNL
