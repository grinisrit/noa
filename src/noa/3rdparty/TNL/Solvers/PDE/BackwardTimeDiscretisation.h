// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT


#pragma once

#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

class BackwardTimeDiscretisation
{
    public:
 
        template< typename RealType,
                  typename IndexType,
                  typename MatrixType >
        __cuda_callable__ static void applyTimeDiscretisation( MatrixType& matrix,
                                                               RealType& b,
                                                               const IndexType index,
                                                               const RealType& u,
                                                               const RealType& tau,
                                                               const RealType& rhs )
        {
            b += u + tau * rhs;
            matrix.addElement( index, index, 1.0, 1.0 );
        }
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL

