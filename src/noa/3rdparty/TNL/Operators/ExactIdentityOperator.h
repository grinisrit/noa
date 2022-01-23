// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/String.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Operators/Operator.h>

namespace noa::TNL {
namespace Operators {

template< int Dimension >
class ExactIdentityOperator
   : public Functions::Domain< Dimension, Functions::SpaceDomain >
{
   public:
 
      template< typename Function >
      __cuda_callable__
      typename Function::RealType
         operator()( const Function& function,
                     const typename Function::PointType& v,
                     const typename Function::RealType& time = 0.0 ) const
      {
         return function( v, time );
      }
 
      template< typename Function,
                int XDerivative = 0,
                int YDerivative = 0,
                int ZDerivative = 0 >
      __cuda_callable__
      typename Function::RealType
         getPartialDerivative( const Function& function,
                               const typename Function::PointType& v,
                               const typename Function::RealType& time = 0.0 ) const
      {
         static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
            "Partial derivative must be non-negative integer." );
 
         return function.template getPartialDerivative< XDerivative, YDerivative, ZDerivative >( v, time );
      }
};

} // namespace Operators
} // namespace noa::TNL

