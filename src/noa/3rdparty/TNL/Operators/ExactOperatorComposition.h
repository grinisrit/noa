// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Operators {

template< typename OuterOperator,
          typename InnerOperator >
class ExactOperatorComposition
{
   public:
 
      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::PointType& v,
                                              const typename Function::RealType& time = 0.0 ) const
      {
         return OuterOperator( innerOperator( function, v, time), v, time );
      }
 
   protected:
 
      InnerOperator innerOperator;
 
      OuterOperator outerOperator;
};

} // namespace Operators
} // namespace noa::TNL

