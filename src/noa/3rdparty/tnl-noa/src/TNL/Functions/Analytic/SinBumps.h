// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/****
 * Tomas Sobotik
 */

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Functions {
namespace Analytic {

template< typename Point >
class SinBumpsBase : public Domain< Point::getSize(), SpaceDomain >
{
public:
   using PointType = Point;
   using RealType = typename Point::RealType;
   enum
   {
      Dimension = PointType::getSize()
   };

   void
   setWaveLength( const PointType& waveLength );

   const PointType&
   getWaveLength() const;

   void
   setAmplitude( const RealType& amplitude );

   const RealType&
   getAmplitude() const;

   void
   setPhase( const PointType& phase );

   const PointType&
   getPhase() const;

   void
   setWavesNumber( const PointType& wavesNumber );

   const PointType&
   getWavesNumber() const;

protected:
   RealType amplitude;

   PointType waveLength, wavesNumber, phase;
};

template< int Dimension, typename Real >
class SinBumps
{};

template< typename Real >
class SinBumps< 1, Real > : public SinBumpsBase< Containers::StaticVector< 1, Real > >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 1, RealType >;

   SinBumps();

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const;
};

template< typename Real >
class SinBumps< 2, Real > : public SinBumpsBase< Containers::StaticVector< 2, Real > >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 2, RealType >;

   SinBumps();

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const;
};

template< typename Real >
class SinBumps< 3, Real > : public SinBumpsBase< Containers::StaticVector< 3, Real > >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 3, RealType >;

   SinBumps();

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const;
};

template< int Dimension, typename Real >
std::ostream&
operator<<( std::ostream& str, const SinBumps< Dimension, Real >& f )
{
   str << "Sin Bumps. function: amplitude = " << f.getAmplitude() << " wavelength = " << f.getWaveLength()
       << " phase = " << f.getPhase();
   return str;
}

}  // namespace Analytic
}  // namespace Functions
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/SinBumps_impl.h>
