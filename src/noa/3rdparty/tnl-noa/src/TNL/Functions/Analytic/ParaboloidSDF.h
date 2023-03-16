// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Functions {
namespace Analytic {

template< int dimensions, typename Real = double >
class ParaboloidSDFBase : public Functions::Domain< dimensions, SpaceDomain >
{
public:
   ParaboloidSDFBase();

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" );

   void
   setXCenter( const Real& waveLength );

   Real
   getXCenter() const;

   void
   setYCenter( const Real& waveLength );

   Real
   getYCenter() const;

   void
   setZCenter( const Real& waveLength );

   Real
   getZCenter() const;

   void
   setCoefficient( const Real& coefficient );

   Real
   getCoefficient() const;

   void
   setOffset( const Real& offset );

   Real
   getOffset() const;

protected:
   Real xCenter, yCenter, zCenter, coefficient, radius;
};

template< int Dimensions, typename Real >
class ParaboloidSDF
{};

template< typename Real >
class ParaboloidSDF< 1, Real > : public ParaboloidSDFBase< 1, Real >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 1, RealType >;

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const;
};

template< typename Real >
class ParaboloidSDF< 2, Real > : public ParaboloidSDFBase< 2, Real >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 2, RealType >;

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const;
};

template< typename Real >
class ParaboloidSDF< 3, Real > : public ParaboloidSDFBase< 3, Real >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< 3, RealType >;

   template< int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const PointType& v, const Real& time = 0.0 ) const;

   __cuda_callable__
   RealType
   operator()( const PointType& v, const Real& time = 0.0 ) const;
};

template< int Dimensions, typename Real >
std::ostream&
operator<<( std::ostream& str, const ParaboloidSDF< Dimensions, Real >& f )
{
   str << "SDF Paraboloid SDF function: amplitude = " << f.getCoefficient() << " offset = " << f.getOffset();
   return str;
}

}  // namespace Analytic
}  // namespace Functions
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/ParaboloidSDF_impl.h>
