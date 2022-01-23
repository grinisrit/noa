// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Functions/Analytic/Twins.h>

namespace noa::TNL {
namespace Functions {
namespace Analytic {   

template< typename Real,
          int Dimension >
bool
TwinsBase< Real, Dimension >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   return true;
}


/***
 * 1D
 */

template< typename Real >
Twins< 1, Real >::Twins()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Point >
__cuda_callable__
Real
Twins< 1, Real >::getPartialDerivative( const Point& v,
                                                   const Real& time ) const
{
   //const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
Twins< 1, Real >::
operator()( const PointType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 2D
 */
template< typename Real >
Twins< 2, Real >::Twins()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Point >
__cuda_callable__
Real
Twins< 2, Real >::
getPartialDerivative( const Point& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return -0.5 * ::sin( M_PI * x) * ::sin( M_PI * x) * ( 1 - ( y - 2 ) * ( y - 2 ) ) * ( 1 - ::tanh( 10 * ( ::sqrt( x * x + y * y ) - 0.6 ) ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
Twins< 2, Real >::
operator()( const PointType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 3D
 */
template< typename Real >
Twins< 3, Real >::Twins()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Point >
__cuda_callable__
Real
Twins< 3, Real >::
getPartialDerivative( const Point& v,
                      const Real& time ) const
{
   //const RealType& x = v.x();
   //const RealType& y = v.y();
   //const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
Twins< 3, Real >::
operator()( const PointType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

} // namespace Analytic
} // namespace Functions
} // namespace noa::TNL

