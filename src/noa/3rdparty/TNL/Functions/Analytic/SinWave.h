// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/****
 * Tomas Sobotik
 */
#pragma once

#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noaTNL {
namespace Functions {
namespace Analytic {   

template< int dimensions,
          typename Real = double >
class SinWaveBase : public Domain< dimensions, SpaceDomain >
{
   public:
 
      SinWaveBase();

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      void setWaveLength( const Real& waveLength );
      
      Real getWaveLength() const;

      void setAmplitude( const Real& amplitude );

      Real getAmplitude() const;

      void setPhase( const Real& phase );

      Real getPhase() const;

      void setWavesNumber( const Real& wavesNumber );

      Real getWavesNumber() const;

   protected:
      
      bool isInsideWaves( const Real& distance ) const;

      Real waveLength, amplitude, phase, wavesNumber;
};

template< int Dimension, typename Real >
class SinWave
{
};

template< typename Real >
class SinWave< 1, Real > : public SinWaveBase< 1, Real >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< 1, RealType > PointType;

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;

};

template< typename Real >
class SinWave< 2, Real > : public SinWaveBase< 2, Real >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< 2, RealType > PointType;
 
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class SinWave< 3, Real > : public SinWaveBase< 3, Real >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< 3, RealType > PointType;


 
      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const PointType& v,
                         const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< int Dimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const SinWave< Dimension, Real >& f )
{
   str << "Sin Wave. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase()
       << " waves number = " << f.getWavesNumber();
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace noaTNL

#include <noa/3rdparty/TNL/Functions/Analytic/SinWave_impl.h>

