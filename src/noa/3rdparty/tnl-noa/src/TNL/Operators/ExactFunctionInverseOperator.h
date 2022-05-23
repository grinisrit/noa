// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/Operator.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/ExactIdentityOperator.h>

namespace noa::TNL {
namespace Operators {

template< int Dimension, typename InnerOperator = ExactIdentityOperator< Dimension > >
class ExactFunctionInverseOperator : public Functions::Domain< Dimension, Functions::SpaceDomain >
{
public:
   InnerOperator&
   getInnerOperator()
   {
      return this->innerOperator;
   }

   const InnerOperator&
   getInnerOperator() const
   {
      return this->innerOperator;
   }

   template< typename Function >
   __cuda_callable__
   typename Function::RealType
   operator()( const Function& function,
               const typename Function::PointType& v,
               const typename Function::RealType& time = 0.0 ) const
   {
      typedef typename Function::RealType RealType;
      return 1.0 / innerOperator( function, v, time );
   }

   template< typename Function, int XDerivative = 0, int YDerivative = 0, int ZDerivative = 0 >
   __cuda_callable__
   typename Function::RealType
   getPartialDerivative( const Function& function,
                         const typename Function::PointType& v,
                         const typename Function::RealType& time = 0.0 ) const
   {
      static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
                     "Partial derivative must be non-negative integer." );
      static_assert( XDerivative + YDerivative + ZDerivative < 2,
                     "Partial derivative of higher order then 1 are not implemented yet." );
      typedef typename Function::RealType RealType;

      if( XDerivative == 1 ) {
         const RealType f = innerOperator( function, v, time );
         const RealType f_x = innerOperator.template getPartialDerivative< Function, 1, 0, 0 >( function, v, time );
         return -f_x / ( f * f );
      }
      if( YDerivative == 1 ) {
         const RealType f = innerOperator( function, v, time );
         const RealType f_y = innerOperator.template getPartialDerivative< Function, 0, 1, 0 >( function, v, time );
         return -f_y / ( f * f );
      }
      if( ZDerivative == 1 ) {
         const RealType f = innerOperator( function, v, time );
         const RealType f_z = innerOperator.template getPartialDerivative< Function, 0, 0, 1 >( function, v, time );
         return -f_z / ( f * f );
      }
   }

protected:
   InnerOperator innerOperator;
};

}  // namespace Operators
}  // namespace noa::TNL
