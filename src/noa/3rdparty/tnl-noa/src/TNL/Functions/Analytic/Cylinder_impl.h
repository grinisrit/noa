// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Cylinder.h>

namespace noa::TNL {
namespace Functions {
namespace Analytic {

template< typename Real, int Dimension >
bool
CylinderBase< Real, Dimension >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   this->diameter = parameters.getParameter< double >( prefix + "diameter" );
   return true;
}

template< typename Real, int Dimension >
void
CylinderBase< Real, Dimension >::setDiameter( const Real& sigma )
{
   this->diameter = diameter;
}

template< typename Real, int Dimension >
const Real&
CylinderBase< Real, Dimension >::getDiameter() const
{
   return this->diameter;
}

/***
 * 1D
 */

template< typename Real >
Cylinder< 1, Real >::Cylinder() = default;

template< typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Point >
__cuda_callable__
Real
Cylinder< 1, Real >::getPartialDerivative( const Point& v, const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return ( ( x * x - this->diameter ) < 0 ) - ( ( x * x - this->diameter ) > 0 ) + 1;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
Cylinder< 1, Real >::operator()( const PointType& v, const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

/****
 * 2D
 */

template< typename Real >
Cylinder< 2, Real >::Cylinder() = default;

template< typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Point >
__cuda_callable__
Real
Cylinder< 2, Real >::getPartialDerivative( const Point& v, const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return ( ( x * x + y * y - this->diameter ) < 0 ) - ( ( x * x + y * y - this->diameter ) > 0 ) + 1;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
Cylinder< 2, Real >::operator()( const PointType& v, const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

/****
 * 3D
 */
template< typename Real >
Cylinder< 3, Real >::Cylinder() = default;

template< typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder, typename Point >
__cuda_callable__
Real
Cylinder< 3, Real >::getPartialDerivative( const Point& v, const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return ( ( x * x + y * y + z * z - this->diameter ) < 0 ) - ( ( x * x + y * y + z * z - this->diameter ) > 0 ) + 1;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
Cylinder< 3, Real >::operator()( const PointType& v, const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

}  // namespace Analytic
}  // namespace Functions
}  // namespace noa::TNL
