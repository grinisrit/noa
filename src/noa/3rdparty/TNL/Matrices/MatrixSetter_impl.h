// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Traverser.h>

namespace noa::TNL {
namespace Matrices {   

template< typename Mesh,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RowsCapacitiesType >
   template< typename EntityType >
void
MatrixSetter< Mesh, DifferentialOperator, BoundaryConditions, RowsCapacitiesType >::
getCompressedRowLengths( const MeshPointer& meshPointer,
                          const DifferentialOperatorPointer& differentialOperatorPointer,
                          const BoundaryConditionsPointer& boundaryConditionsPointer,
                          RowsCapacitiesTypePointer& rowLengthsPointer ) const
{
   {
      TraverserUserData
         userData( &differentialOperatorPointer.template getData< DeviceType >(),
                   &boundaryConditionsPointer.template getData< DeviceType >(),
                   &rowLengthsPointer.template modifyData< DeviceType >() );
      Meshes::Traverser< MeshType, EntityType > meshTraverser;
      meshTraverser.template processBoundaryEntities< TraverserBoundaryEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
      meshTraverser.template processInteriorEntities< TraverserInteriorEntitiesProcessor >
                                                    ( meshPointer,
                                                      userData );
   }
}

} // namespace Matrices
} // namespace noa::TNL
