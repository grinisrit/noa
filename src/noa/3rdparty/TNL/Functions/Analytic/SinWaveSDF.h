// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noa::TNL {
   namespace Functions {
      namespace Analytic {

template< int dimensions,
          typename Real = double >
class SinWaveSDFBase : public Functions::Domain< dimensions, SpaceDomain >
{
   public:

      SinWaveSDFBase();

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

      __cuda_callable__
      Real sinWaveFunctionSDF( const Real& r ) const;
      
      Real waveLength, amplitude, phase, wavesNumber;
};

template< int Dimensions, typename Real >
class SinWaveSDF
{
};

template< typename Real >
class SinWaveSDF< 1, Real > : public SinWaveSDFBase< 1, Real >
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
class SinWaveSDF< 2, Real > : public SinWaveSDFBase< 2, Real >
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
class SinWaveSDF< 3, Real > : public SinWaveSDFBase< 3, Real >
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

template< int Dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const SinWaveSDF< Dimensions, Real >& f )
{
   str << "SDF Sin Wave SDF. function: amplitude = " << f.getAmplitude()
       << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase()
       << " # of waves = " << f.getWavesNumber();
   return str;
}
        
      } // namespace Analytic
   } // namespace Functions 
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Functions/Analytic/SinWaveSDF_impl.h>
