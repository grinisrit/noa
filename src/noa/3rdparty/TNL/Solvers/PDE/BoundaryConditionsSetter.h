// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT


#pragma once

#include <noa/3rdparty/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/TNL/Functions/FunctionAdapter.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/TNL/Meshes/Traverser.h>

namespace noaTNL {
namespace Solvers {
namespace PDE {

template< typename Real,
          typename DofVector,
          typename BoundaryConditions >
class BoundaryConditionsSetterTraverserUserData
{
   public:

      const Real time;

      const BoundaryConditions* boundaryConditions;

      DofVector *u;

      BoundaryConditionsSetterTraverserUserData(
         const Real& time,
         const BoundaryConditions* boundaryConditions,
         DofVector* u )
      : time( time ),
        boundaryConditions( boundaryConditions ),
        u( u )
      {};
};


template< typename MeshFunction,
          typename BoundaryConditions >
class BoundaryConditionsSetter
{
   public:
      typedef typename MeshFunction::MeshType MeshType;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::DeviceType DeviceType;
      typedef typename MeshFunction::IndexType IndexType;
      typedef BoundaryConditionsSetterTraverserUserData<
         RealType,
         MeshFunction,
         BoundaryConditions > TraverserUserData;
      typedef Pointers::SharedPointer<  MeshType, DeviceType > MeshPointer;
      typedef Pointers::SharedPointer<  BoundaryConditions, DeviceType > BoundaryConditionsPointer;
      typedef Pointers::SharedPointer<  MeshFunction, DeviceType > MeshFunctionPointer;

      template< typename EntityType = typename MeshType::Cell >
      static void apply( const BoundaryConditionsPointer& boundaryConditions,
                         const RealType& time,
                         MeshFunctionPointer& u )
      {
         Pointers::SharedPointer<  TraverserUserData, DeviceType >
            userData( time,
                      &boundaryConditions.template getData< DeviceType >(),
                      &u.template modifyData< DeviceType >() );
         Meshes::Traverser< MeshType, EntityType > meshTraverser;
         meshTraverser.template processBoundaryEntities< TraverserUserData,
                                                         TraverserBoundaryEntitiesProcessor >
                                                       ( u->getMeshPointer(),
                                                         userData );
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
               ( *userData.u )( entity ) = userData.boundaryConditions->operator()
               ( *userData.u,
                 entity,
                 userData.time );
            }

      };
};

} // namespace PDE
} // namespace Solvers
} // namespace noaTNL


