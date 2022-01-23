// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once 

#include <noa/3rdparty/TNL/Functions/Analytic/Paraboloid.h>

namespace noa::TNL {
   namespace Functions {
      namespace Analytic {

template< int dimensions, typename Real >
ParaboloidBase< dimensions, Real >::ParaboloidBase()
: xCenter( 0 ), yCenter( 0 ), zCenter( 0 ),
  coefficient( 1 ), radius ( 0 )
{
}

template< int dimensions, typename Real >
bool ParaboloidBase< dimensions, Real >::setup( const Config::ParameterContainer& parameters,
        								 const String& prefix)
{
   this->xCenter = parameters.getParameter< double >( "x-center" );
   this->yCenter = parameters.getParameter< double >( "y-center" );
   this->zCenter = parameters.getParameter< double >( "z-center" );
   this->coefficient = parameters.getParameter< double >( "coefficient" );
   this->radius = parameters.getParameter< double >( "radius" );

   return true;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setXCenter( const Real& xCenter )
{
   this->xCenter = xCenter;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getXCenter() const
{
   return this->xCenter;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setYCenter( const Real& yCenter )
{
   this->yCenter = yCenter;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getYCenter() const
{
   return this->yCenter;
}
template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setZCenter( const Real& zCenter )
{
   this->zCenter = zCenter;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getZCenter() const
{
   return this->zCenter;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setCoefficient( const Real& amplitude )
{
   this->coefficient = coefficient;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getCoefficient() const
{
   return this->coefficient;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setOffset( const Real& offset )
{
   this->radius = offset;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getOffset() const
{
   return this->radius;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
Paraboloid< 1, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->coefficient * ( ( x - this -> xCenter ) * ( x - this -> xCenter ) - this->radius*this->radius );
   if( XDiffOrder == 1 )
      return 2.0 * this->coefficient * ( x - this -> xCenter );
   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
Paraboloid< 2, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->coefficient * ( ( x - this -> xCenter ) * ( x - this -> xCenter )
    		  	  	  	         + ( y - this -> yCenter ) * ( y - this -> yCenter ) - this->radius*this->radius );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0)
	   return 2.0 * this->coefficient * ( x - this -> xCenter );
   if( YDiffOrder == 1 && XDiffOrder == 0)
	   return 2.0 * this->coefficient * ( y - this -> yCenter );
   if( XDiffOrder == 2 && YDiffOrder == 0)
	   return 2.0 * this->coefficient;
   if( YDiffOrder == 2 && XDiffOrder == 0)
	   return 2.0 * this->coefficient;
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
Paraboloid< 3, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   const Real& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->coefficient * ( ( x - this -> xCenter ) * ( x - this -> xCenter )
    		  	  	  	         + ( y - this -> yCenter ) * ( y - this -> yCenter )
    		  	  	  	         + ( z - this -> zCenter ) * ( z - this -> zCenter ) - this->radius*this->radius );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient * ( x - this -> xCenter );
   if( YDiffOrder == 1 && XDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient * ( y - this -> yCenter );
   if( ZDiffOrder == 1 && XDiffOrder == 0 && YDiffOrder == 0)
	   return 2.0 * this->coefficient * ( z - this -> zCenter );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient;
   if( YDiffOrder == 2 && XDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient;
   if( ZDiffOrder == 2 && XDiffOrder == 0 && YDiffOrder == 0)
	   return 2.0 * this->coefficient;
   return 0.0;
}
         
      } // namespace Analytic
   } // namedspace Functions
} // namespace noa::TNL
