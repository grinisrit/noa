// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
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

#include <noa/3rdparty/tnl-noa/src/TNL/Problems/MeanCurvatureFlowProblem.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ.h>

namespace noa::TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = NonlinearDiffusion<
             Mesh,
             tnlOneSideDiffNonlinearOperator<
                Mesh,
                tnlOneSideDiffOperatorQ< Mesh, typename BoundaryCondition::RealType, typename BoundaryCondition::IndexType >,
                typename BoundaryCondition::RealType,
                typename BoundaryCondition::IndexType >,
             typename BoundaryCondition::RealType,
             typename BoundaryCondition::IndexType > >
class MeanCurvatureFlowEocProblem
: public MeanCurvatureFlowProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >
{
public:
   bool
   setup( const Config::ParameterContainer& parameters );
};

}  // namespace Problems
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Problems/MeanCurvatureFlowEocProblem_impl.h>
