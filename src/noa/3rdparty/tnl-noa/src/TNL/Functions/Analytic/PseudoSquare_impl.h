// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/PseudoSquare.h>

namespace noa::TNL {
namespace Functions {
namespace Analytic {

template< typename Real, int Dimension >
bool
PseudoSquareBase< Real, Dimension >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   this->height = parameters.getParameter< double >( prefix + "height" );

   return true;
}

/***
 * 1D
 */

template< typename Real >
PseudoSquare< 1, Real >::PseudoSquare() = default;

template< typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
__cuda_callable__
Real
PseudoSquare< 1, Real >::getPartialDerivative( const PointType& v, const Real& time ) const
{
   // const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
PseudoSquare< 1, Real >::operator()( const PointType& v, const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

/****
 * 2D
 */
template< typename Real >
PseudoSquare< 2, Real >::PseudoSquare() = default;

template< typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
__cuda_callable__
Real
PseudoSquare< 2, Real >::getPartialDerivative( const PointType& v, const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return x * x + y * y - this->height - ::cos( 2 * x * y ) * ::cos( 2 * x * y );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
PseudoSquare< 2, Real >::operator()( const PointType& v, const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

/****
 * 3D
 */
template< typename Real >
PseudoSquare< 3, Real >::PseudoSquare() = default;

template< typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
__cuda_callable__
Real
PseudoSquare< 3, Real >::getPartialDerivative( const PointType& v, const Real& time ) const
{
   // const RealType& x = v.x();
   // const RealType& y = v.y();
   // const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
PseudoSquare< 3, Real >::operator()( const PointType& v, const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

}  // namespace Analytic
}  // namespace Functions
}  // namespace noa::TNL
