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

#include <noa/3rdparty/tnl-noa/src/TNL/Problems/HeatEquationProblem.h>

namespace noa::TNL {
namespace Problems {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator = Operators::LinearDiffusion< Mesh, typename BoundaryCondition::RealType > >
class HeatEquationEocProblem : public HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >
{
public:
   typedef HeatEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator > BaseType;

   using typename BaseType::MeshPointer;

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix );
};

}  // namespace Problems
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Problems/HeatEquationEocProblem_impl.h>
