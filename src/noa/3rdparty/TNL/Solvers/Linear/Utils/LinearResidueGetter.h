// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL {
namespace Solvers {
namespace Linear {

struct LinearResidueGetter
{
   template< typename Matrix, typename Vector1, typename Vector2 >
   static typename Matrix::RealType
   getResidue( const Matrix& matrix,
               const Vector1& x,
               const Vector2& b,
               typename Matrix::RealType bNorm = 0 );
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#include "LinearResidueGetter.hpp"
