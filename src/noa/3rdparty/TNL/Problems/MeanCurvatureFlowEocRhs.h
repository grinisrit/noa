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

#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noaTNL {
namespace Problems {

template< typename ExactOperator,
          typename TestFunction,
          int Dimension >
class MeanCurvatureFlowEocRhs : public Domain< Dimension, SpaceDomain >
{
   public:

      typedef ExactOperator ExactOperatorType;
      typedef TestFunction TestFunctionType;
      typedef typename TestFunctionType::RealType RealType;
      typedef StaticVector< Dimension, RealType > PointType;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         if( ! testFunction.setup( parameters, prefix ) )
            return false;
         return true;
      };

      template< typename Point,
                typename Real >
      __cuda_callable__
      Real operator()( const Point& vertex,
                       const Real& time ) const
      {
         return testFunction.getTimeDerivative( vertex, time )
                - exactOperator( testFunction, vertex, time );
      };

   protected:
 
      ExactOperator exactOperator;

      TestFunction testFunction;
};

} // namespace Problems
} // namespace noaTNL
