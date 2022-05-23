// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include "HeatEquationEocProblem.h"

namespace noa::TNL {
namespace Problems {

template< typename Mesh, typename BoundaryCondition, typename RightHandSide, typename DifferentialOperator >
bool
HeatEquationEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::setup(
   const Config::ParameterContainer& parameters,
   const String& prefix )
{
   if( ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, prefix )
       || ! this->rightHandSidePointer->setup( parameters ) )
      return false;
   this->explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer );
   this->explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer );
   this->explicitUpdater.setRightHandSide( this->rightHandSidePointer );
   this->systemAssembler.setDifferentialOperator( this->differentialOperatorPointer );
   this->systemAssembler.setBoundaryConditions( this->boundaryConditionPointer );
   this->systemAssembler.setRightHandSide( this->rightHandSidePointer );
   return true;
}

}  // namespace Problems
}  // namespace noa::TNL
