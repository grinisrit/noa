// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noaTNL {
namespace Functions {
namespace Analytic {   

template< int Dimension,
          typename Real >
__cuda_callable__
Constant< Dimension, Real >::
Constant()
: constant( 0.0 )
{
}

template< int Dimension,
          typename Real >
void
Constant< Dimension, Real >::
setConstant( const RealType& constant )
{
   this->constant = constant;
}

template< int Dimension,
          typename Real >
const Real&
Constant< Dimension, Real >::
getConstant() const
{
   return this->constant;
}

template< int FunctionDimension,
          typename Real >
void
Constant< FunctionDimension, Real >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   config.addEntry     < double >( prefix + "constant", "Value of the constant function.", 0.0 );
}

template< int Dimension,
          typename Real >
bool
Constant< Dimension, Real >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   this->setConstant( parameters.getParameter< double >( prefix + "constant") );
   return true;
}

template< int Dimension,
          typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
Constant< Dimension, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   if( XDiffOrder || YDiffOrder || ZDiffOrder )
      return 0.0;
   return constant;
}

} // namespace Analytic
} // namespace Functions
} // namespace noaTNL
