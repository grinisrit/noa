// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Functions/Analytic/SinWaveSDF.h>

namespace TNL {
namespace Functions {
namespace Analytic {

template< int dimensions, typename Real >
SinWaveSDFBase< dimensions, Real >::SinWaveSDFBase()
: waveLength( 1.0 ),
  amplitude( 1.0 ),
  phase( 0 ),
  wavesNumber( 0 )
{
}

template< int dimensions, typename Real >
bool SinWaveSDFBase< dimensions, Real >::setup( const Config::ParameterContainer& parameters,
                                           const String& prefix )
{
   this->waveLength = parameters.getParameter< double >( prefix + "wave-length" );
   this->amplitude = parameters.getParameter< double >( prefix + "amplitude" );
   this->phase = parameters.getParameter< double >( prefix + "phase" );
   while(this->phase >2.0*M_PI)
      this->phase -= 2.0*M_PI;
   this->wavesNumber = ceil( parameters.getParameter< double >( prefix + "waves-number" ) );
   return true;
}

template< int dimensions, typename Real >
void SinWaveSDFBase< dimensions, Real >::setWaveLength( const Real& waveLength )
{
   this->waveLength = waveLength;
}

template< int dimensions, typename Real >
Real SinWaveSDFBase< dimensions, Real >::getWaveLength() const
{
   return this->waveLength;
}

template< int dimensions, typename Real >
void SinWaveSDFBase< dimensions, Real >::setAmplitude( const Real& amplitude )
{
   this->amplitude = amplitude;
}

template< int dimensions, typename Real >
Real SinWaveSDFBase< dimensions, Real >::getAmplitude() const
{
   return this->amplitude;
}

template< int dimensions, typename Real >
void SinWaveSDFBase< dimensions, Real >::setPhase( const Real& phase )
{
   this->phase = phase;
}

template< int dimensions, typename Real >
Real SinWaveSDFBase< dimensions, Real >::getPhase() const
{
   return this->phase;
}

template< int dimensions, typename Real >
void SinWaveSDFBase< dimensions, Real >::setWavesNumber( const Real& wavesNumber )
{
   this->wavesNumber = wavesNumber;
}

template< int dimensions, typename Real >
Real SinWaveSDFBase< dimensions, Real >::getWavesNumber() const
{
   return this->wavesNumber;
}

template< int dimensions, typename Real >
__cuda_callable__
Real SinWaveSDFBase< dimensions, Real >::sinWaveFunctionSDF( const Real& r ) const
{
   if( this->wavesNumber == 0.0 || r < this->wavesNumber * this->waveLength )
      return sign( r - round( 2.0 * r / this->waveLength ) * this->waveLength / 2.0 )
             * ( r - round( 2.0 * r / this->waveLength ) * this->waveLength / 2.0 )
             * sign( ::sin( 2.0 * M_PI * r / this->waveLength ) );
   else
      return r - this->wavesNumber * this->waveLength;   
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
SinWaveSDF< 1, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   const RealType& x = v.x();
   const RealType distance = ::sqrt( x * x ) + this->phase * this->waveLength / (2.0*M_PI);
   if( XDiffOrder == 0 )
      return this->sinWaveFunctionSDF( distance );
   TNL_ASSERT_TRUE( false, "TODO: implement this" );
   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
SinWaveSDF< 2, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   if( ZDiffOrder != 0 )
      return 0.0;

   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType distance  = ::sqrt( x * x + y * y ) + this->phase * this->waveLength / (2.0*M_PI);
   if( XDiffOrder == 0 && YDiffOrder == 0)
      return this->sinWaveFunctionSDF( distance );
   TNL_ASSERT_TRUE( false, "TODO: implement this" );
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
SinWaveSDF< 3, Real >::
getPartialDerivative( const PointType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   const RealType distance  = ::sqrt( x * x +  y * y + z * z ) +  this->phase * this->waveLength / (2.0*M_PI);
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return this->sinWaveFunctionSDF( distance );
   TNL_ASSERT_TRUE( false, "TODO: implement this" );
   return 0.0;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL
