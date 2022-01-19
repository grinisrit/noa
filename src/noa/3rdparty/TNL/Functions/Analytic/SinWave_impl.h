// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Functions/Analytic/SinWave.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< int dimensions, typename Real >
SinWaveBase< dimensions, Real >::SinWaveBase()
: waveLength( 1.0 ),
  amplitude( 1.0 ),
  phase( 0 ),
  wavesNumber( 0 )
{
}

template< int dimensions, typename Real >
bool SinWaveBase< dimensions, Real >::setup( const Config::ParameterContainer& parameters,
                                           const String& prefix )
{
   this->waveLength = parameters.getParameter< double >( prefix + "wave-length" );
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->phase = parameters.getParameter< double >( prefix + "phase" );
   parameters.getParameter< double >( prefix + "waves-number" );
   return true;
}

template< int dimensions, typename Real >
void SinWaveBase< dimensions, Real >::setWaveLength( const Real& waveLength )
{
   this->waveLength = waveLength;
}

template< int dimensions, typename Real >
Real SinWaveBase< dimensions, Real >::getWaveLength() const
{
   return this->waveLength;
}

template< int dimensions, typename Real >
void SinWaveBase< dimensions, Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< int dimensions, typename Real >
Real SinWaveBase< dimensions, Real >::getAmplitude() const
{
   return this->amplitude;
}

template< int dimensions, typename Real >
void SinWaveBase< dimensions, Real >::setPhase( const Real& phase )
{
   this->phase = phase;
}

template< int dimensions, typename Real >
Real SinWaveBase< dimensions, Real >::getPhase() const
{
   return this->phase;
}



template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinWave< 1, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
   {
      RealType arg = 2.0 * M_PI * x  / this->waveLength;
      if( this->wavesNumber )
      {
         if( abs( arg ) > this->wavesNumber )
            arg = sign( x ) * this->wavesNumber;
      }
      //cout << "arg = " << arg << " amplitude = " << this->amplitude << " -> " << this->amplitude * ::sin( this->phase + arg ) << std::endl;
      return this->amplitude * ::sin( this->phase + arg );
   }
   if( XDiffOrder == 1 )
      return 2.0 * M_PI / this->waveLength * this->amplitude * ::cos( this->phase + 2.0 * M_PI * x / this->waveLength );
   if( XDiffOrder == 2 )
      return -4.0 * M_PI * M_PI / ( this->waveLength * this->waveLength ) * this->amplitude * ::sin( this->phase + 2.0 * M_PI * x / this->waveLength );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
SinWave< 1, Real >::
operator()( const PointType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}



template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinWave< 2, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if ( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0)
   {
      return this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 )
      return 2.0 * M_PI * x / ( this->waveLength * ::sqrt( x * x + y * y ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength );
   if( XDiffOrder == 2 && YDiffOrder == 0 )
      return 2.0 * M_PI * x * x / ( this->waveLength * ::sqrt( x * x + y * y ) * ::sqrt( x * x + y * y ) * ::sqrt( x * x + y * y ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength ) - 4.0 * M_PI * M_PI * x * x / ( this->waveLength * this->waveLength * ( x * x + y * y ) ) * this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 1 )
      return 2.0 * M_PI * y / ( this->waveLength * ::sqrt( x * x + y * y ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 2 )
      return 2.0 * M_PI * y * y / ( this->waveLength * ::sqrt( x * x + y * y ) * ::sqrt( x * x + y * y ) * ::sqrt( x * x + y * y ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength ) - 4.0 * M_PI * M_PI * y * y / ( this->waveLength * this->waveLength * ( x * x + y * y ) ) * this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength );
   if( XDiffOrder == 1 && YDiffOrder == 1 )
      return -4.0 * M_PI * M_PI * x * y / ( this->waveLength * this->waveLength * (x * x + y * y ) )* this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength )
             - 2.0 * M_PI * this->amplitude * x * y * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y ) / this->waveLength ) / ( this->waveLength  * ::sqrt( (x * x + y * y )  * (x * x + y * y ) * (x * x + y * y ) ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
SinWave< 2, Real >::
operator()( const PointType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
SinWave< 3, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 2.0 * M_PI * x / ( this->waveLength * ::sqrt( x * x + y * y + z * z ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 2.0 * M_PI * ( y * y + z * z ) / ( this->waveLength * ::sqrt( x * x + y * y + z * z ) * ::sqrt( x * x + y * y + z * z ) * ::sqrt( x * x + y * y + z * z ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength ) - 4.0 * M_PI * M_PI * x * x / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 0 )
      return 2.0 * M_PI * y / ( this->waveLength * ::sqrt( x * x + y * y + z * z ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 2 && ZDiffOrder == 0 )
      return 2.0 * M_PI * ( x * x + z * z ) / ( this->waveLength * ::sqrt( x * x + y * y + z * z ) * ::sqrt( x * x + y * y + z * z ) * ::sqrt( x * x + y * y + z * z ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength ) - 4.0 * M_PI * M_PI * y * y / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 1 )
      return 2.0 * M_PI * z / ( this->waveLength * ::sqrt( x * x + y * y + z * z ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 2 )
      return 2.0 * M_PI * ( x * x + y * y ) / ( this->waveLength * ::sqrt( x * x + y * y + z * z ) * ::sqrt( x * x + y * y + z * z ) * ::sqrt( x * x + y * y + z * z ) ) * this->amplitude * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength ) - 4.0 * M_PI * M_PI * z * z / ( this->waveLength * this->waveLength * ( x * x + y * y + z * z ) ) * this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength );
   if( XDiffOrder == 1 && YDiffOrder == 1 && ZDiffOrder == 0 )
      return -4.0 * M_PI * M_PI * x * y / ( this->waveLength * this->waveLength * (x * x + y * y + z * z ) )* this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength )
             - 2.0 * M_PI * this->amplitude * x * y * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength ) / ( this->waveLength  * ::sqrt( (x * x + y * y + z * z )  * (x * x + y * y + z * z ) * (x * x + y * y + z * z ) ) );
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 1 )
      return -4.0 * M_PI * M_PI * x * z / ( this->waveLength * this->waveLength * (x * x + y * y + z * z ) )* this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength )
             - 2.0 * M_PI * this->amplitude * x * z * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength ) / ( this->waveLength  * ::sqrt( (x * x + y * y + z * z )  * (x * x + y * y + z * z ) * (x * x + y * y + z * z ) ) );
   if( XDiffOrder == 0 && YDiffOrder == 1 && ZDiffOrder == 1 )
      return -4.0 * M_PI * M_PI * z * y / ( this->waveLength * this->waveLength * (x * x + y * y + z * z ) )* this->amplitude * ::sin( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength )
             - 2.0 * M_PI * this->amplitude * z * y * ::cos( this->phase + 2.0 * M_PI * ::sqrt( x * x + y * y + z * z ) / this->waveLength ) / ( this->waveLength  * ::sqrt( (x * x + y * y + z * z )  * (x * x + y * y + z * z ) * (x * x + y * y + z * z ) ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
SinWave< 3, Real >::
operator()( const PointType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL
