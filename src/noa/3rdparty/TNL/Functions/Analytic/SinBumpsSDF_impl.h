// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once 

#include <noa/3rdparty/TNL/Functions/Analytic/SinBumpsSDF.h>

namespace noa::TNL {
   namespace Functions {
      namespace Analytic {

template< typename Point >
void SinBumpsSDFBase< Point >::setWaveLength( const Point& waveLength )
{
   this->waveLength = waveLength;
}

template< typename Point >
const Point& SinBumpsSDFBase< Point >::getWaveLength() const
{
   return this->waveLength;
}

template< typename Point >
void SinBumpsSDFBase< Point >::setAmplitude( const typename Point::RealType& amplitude )
{
   this->amplitude = amplitude;
}

template< typename Point >
const typename Point::RealType& SinBumpsSDFBase< Point >::getAmplitude() const
{
   return this->amplitude;
}

template< typename Point >
void SinBumpsSDFBase< Point >::setPhase( const Point& phase )
{
   this->phase = phase;
}

template< typename Point >
const Point& SinBumpsSDFBase< Point >::getPhase() const
{
   return this->phase;
}

/***
 * 1D
 */

template< typename Real >
SinBumpsSDF< 1, Real >::SinBumpsSDF()
{
}

template< typename Real >
bool SinBumpsSDF< 1, Real >::setup( const Config::ParameterContainer& parameters,
        const String& prefix)
{
   this->amplitude = parameters.getParameter< double >( prefix+"amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix+"wave-length-x" );
   while(this->waveLength.x() > 2.0*M_PI)
	   this->waveLength.x() -= 2.0*M_PI;
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->phase.x() = parameters.getParameter< double >( prefix+"phase-x" );
   return true;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinBumpsSDF< 1, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   RealType xp = abs( x ) + sign( x ) * this->phase.x() * this->waveLength.x() / ( 2.0*M_PI );
   if( this->wavesNumber.x() != 0.0 && xp > this->wavesNumber.x() * this->waveLength.x() )
      return 0.0;
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return sign( xp - round( (2.0 * xp ) / this->waveLength.x() ) * this->waveLength.x() / 2.0 )
          * ( xp- round((2.0 * xp)/this->waveLength.x())* this->waveLength.x()/2.0)
          * sign( ::sin(this-> phase.x() + 2.0 * M_PI * x / this->waveLength.x()));
   if( XDiffOrder == 1 )
      return 1.0;
   return 0.0;
}

/****
 * 2D
 */

template< typename Real >
SinBumpsSDF< 2, Real >::SinBumpsSDF()
{
}

template< typename Real >
bool SinBumpsSDF< 2, Real >::setup( const Config::ParameterContainer& parameters,
        const String& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix+"amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix+"wave-length-x" );
   this->waveLength.y() = parameters.getParameter< double >( prefix+"wave-length-y" );
   while(this->waveLength.x() > 2.0*M_PI)
	   this->waveLength.x() -= 2.0*M_PI;
   while(this->waveLength.y() > 2.0*M_PI)
	   this->waveLength.y() -= 2.0*M_PI;
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->wavesNumber.y() = ceil( parameters.getParameter< double >( prefix+"waves-number-y" ) );
   this->phase.x() = parameters.getParameter< double >( prefix+"phase-x" );
   this->phase.y() = parameters.getParameter< double >( prefix+"phase-y" );
   return true;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinBumpsSDF< 2, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
	   const RealType& x = v.x();
	   const RealType& y = v.y();
	   RealType xp = ::sqrt(x*x) + sign( x ) * this->phase.x() * this->waveLength.x() / (2.0*M_PI);
	   RealType yp = ::sqrt(y*y) + sign( y ) * this->phase.y() * this->waveLength.y() / (2.0*M_PI);
	   if( ( xp > this->wavesNumber.x()*this->waveLength.x() && this->wavesNumber.x() != 0.0 )  ||
			 ( yp > this->wavesNumber.y()*this->waveLength.y() && this->wavesNumber.y() != 0.0 ) )
		   return 0.0;
	   const RealType sx = sign(xp - round((2.0 * xp)/this->waveLength.x())* this->waveLength.x()/2.0)
	  		  		    *(xp - round((2.0 * xp)/this->waveLength.x())* this->waveLength.x()/2.0);
	   const RealType sy = sign(yp - round((2.0 * yp)/this->waveLength.y())* this->waveLength.y()/2.0)
	  		  		    *(yp - round((2.0 * yp)/this->waveLength.y())* this->waveLength.y()/2.0);
	   RealType sxy;
	   if(sx < sy)
		   sxy = sx;
	   else
		   sxy = sy;
	   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
	   {
		      return sxy * sign( ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() )
		      	  	  	       * ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() ) );
	   }
	   return 0.0;
}

/****
 * 3D
 */

template< typename Real >
SinBumpsSDF< 3, Real >::SinBumpsSDF()
{
}

template< typename Real >
bool SinBumpsSDF< 3, Real >::setup( const Config::ParameterContainer& parameters,
        const String& prefix )
{
   this->amplitude = parameters.getParameter< double >( prefix+"amplitude" );
   this->waveLength.x() = parameters.getParameter< double >( prefix+"wave-length-x" );
   this->waveLength.y() = parameters.getParameter< double >( prefix+"wave-length-y" );
   this->waveLength.z() = parameters.getParameter< double >( prefix+"wave-length-z" );
   while(this->waveLength.x() > 2.0*M_PI)
	   this->waveLength.x() -= 2.0*M_PI;
   while(this->waveLength.y() > 2.0*M_PI)
	   this->waveLength.y() -= 2.0*M_PI;
   while(this->waveLength.z() > 2.0*M_PI)
	   this->waveLength.z() -= 2.0*M_PI;
   this->wavesNumber.x() = ceil( parameters.getParameter< double >( prefix+"waves-number-x" ) );
   this->wavesNumber.y() = ceil( parameters.getParameter< double >( prefix+"waves-number-y" ) );
   this->wavesNumber.z() = ceil( parameters.getParameter< double >( prefix+"waves-number-z" ) );
   this->phase.x() = parameters.getParameter< double >( prefix+"phase-x" );
   this->phase.y() = parameters.getParameter< double >( prefix+"phase-y" );
   this->phase.z() = parameters.getParameter< double >(prefix+"phase-z" );
   return true;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinBumpsSDF< 3, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
	   const RealType& x = v.x();
	   const RealType& y = v.y();
	   const RealType& z = v.z();
	   RealType xp = ::sqrt(x*x) + sign(x)*(this->phase.x())*(this->waveLength.x())/(2.0*M_PI);
	   RealType yp = ::sqrt(y*y) + sign(y)*(this->phase.y())*(this->waveLength.y())/(2.0*M_PI);
	   RealType zp = ::sqrt(z*z) + sign(z)*(this->phase.z())*(this->waveLength.z())/(2.0*M_PI);
	   if ( ( xp > this->wavesNumber.x()*this->waveLength.x() && this->wavesNumber.x() != 0.0 ) ||
			(yp > this->wavesNumber.y()*this->waveLength.y() && this->wavesNumber.y() != 0.0 ) ||
			(::sqrt(z*z) > this->wavesNumber.z()*this->waveLength.z() && this->wavesNumber.z() != 0.0 ) )
		   return 0.0;
	   const RealType sx = sign(xp - round((2.0 * xp)/this->waveLength.x())* this->waveLength.x()/2.0)
	  		  		    *(xp - round((2.0 * xp)/this->waveLength.x())* this->waveLength.x()/2.0);
	   const RealType sy = sign(yp - round((2.0 * yp)/this->waveLength.y())* this->waveLength.y()/2.0)
	  		  		    *(yp - round((2.0 * yp)/this->waveLength.y())* this->waveLength.y()/2.0);
	   const RealType sz = sign(zp - round((2.0 * zp)/this->waveLength.z())* this->waveLength.z()/2.0)
	  		  		    *(zp - round((2.0 * zp)/this->waveLength.z())* this->waveLength.z()/2.0);
	   RealType sxyz;
	   if(sx <= sy && sx <= sz)
		   sxyz = sx;
	   else if ( sy <= sx && sy <= sz)
		   sxyz = sy;
	   else
		   sxyz = sz;
	   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
	   {
	      return sxyz * sign( ::sin( this->phase.x() + 2.0 * M_PI * x / this->waveLength.x() )
	      	  	     	  	* ::sin( this->phase.y() + 2.0 * M_PI * y / this->waveLength.y() )
	      	  	  	  	  	* ::sin( this->phase.z() + 2.0 * M_PI * z / this->waveLength.z() ) );
	   }
	   return 0.0;
}

      } // namespace Analytic
   } // namespace Fucntions
} // namespace noa::TNL
