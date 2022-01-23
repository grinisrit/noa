// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Operators {   

template< int Dimension,
          int XDerivative,
          int YDerivative,
          int ZDerivative >
class ExactDifference
   : public Functions::Domain< Dimension, Functions::SpaceDomain >
{
   public:
 
      template< typename Function >
      __cuda_callable__
      typename Function::RealType operator()(
         const Function& function,
         const typename Function::PointType& vertex,
         const typename Function::RealType& time = 0 ) const
      {
         return function.template getPartialDerivative<
            XDerivative,
            YDerivative,
            ZDerivative >(
            vertex,
            time );
      }
};

} // namespace Operators
} // namespace noa::TNL

