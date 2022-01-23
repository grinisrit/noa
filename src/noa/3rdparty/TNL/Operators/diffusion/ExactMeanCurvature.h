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

#include <noa/3rdparty/TNL/Operators/diffusion/ExactNonlinearDiffusion.h>
#include <noa/3rdparty/TNL/Operators/ExactFunctionInverseOperator.h>
#include <noa/3rdparty/TNL/Operators/geometric/ExactGradientNorm.h>

namespace noa::TNL {
namespace Operators {   

template< int Dimension,
          typename InnerOperator = ExactIdentityOperator< Dimension > >
class ExactMeanCurvature
: public Functions::Domain< Dimension, Functions::SpaceDomain >
{
   public:
 
      typedef ExactGradientNorm< Dimension > ExactGradientNormType;
      typedef ExactFunctionInverseOperator< Dimension, ExactGradientNormType > FunctionInverse;
      typedef ExactNonlinearDiffusion< Dimension, FunctionInverse > NonlinearDiffusion;
 
      template< typename Real >
      void setRegularizationEpsilon( const Real& eps)
      {
         nonlinearDiffusion.getNonlinearity().getInnerOperator().setRegularizationEpsilon( eps );
      }
 
      template< typename Function >
      __cuda_callable__
      typename Function::RealType
         operator()( const Function& function,
                     const typename Function::PointType& v,
                     const typename Function::RealType& time = 0.0 ) const
      {
         return this->nonlinearDiffusion( function, v, time );
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
         static_assert( XDerivative + YDerivative + ZDerivative < 1, "Partial derivative of higher order then 1 are not implemented yet." );
         typedef typename Function::RealType RealType;
 
         if( XDerivative == 1 )
         {
         }
         if( YDerivative == 1 )
         {
         }
         if( ZDerivative == 1 )
         {
         }
      }
 
 
   protected:
 
      ExactGradientNormType gradientNorm;
 
      FunctionInverse functionInverse;
 
      NonlinearDiffusion nonlinearDiffusion;
 
};

} // namespace Operators
} // namespace noa::TNL

