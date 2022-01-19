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

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
MeanCurvatureFlowEocProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator  >::
setup( const Config::ParameterContainer& parameters )
{
   if( ! this->boundaryCondition.setup( parameters ) ||
       ! this->rightHandSide.setup( parameters ) ||
       ! this->differentialOperator.nonlinearDiffusionOperator.operatorQ.setEps(parameters.getParameter< double >("eps")) )
      return false;
 
   return true;
}

} // namespace Problems
} // namespace TNL
