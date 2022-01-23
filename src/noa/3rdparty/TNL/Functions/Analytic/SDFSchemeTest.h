// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>
#include <functions/tnlSDFSinWaveFunction.h>
#include <functions/tnlSDFSinWaveFunctionSDF.h>
#include <functions/tnlSDFSinBumps.h>
#include <functions/tnlSDFSinBumpsSDF.h>
#include <functions/tnlExpBumpFunction.h>
#include <functions/tnlSDFParaboloid.h>
#include <functions/tnlSDFParaboloidSDF.h>

namespace noa::TNL {
   namespace Functions {
      namespace Analytic {

template< typename function, typename Real = double >
class SDFSchemeTestBase
{
   public:

   SDFSchemeTestBase();

   bool setup( const Config::ParameterContainer& parameters,
           const String& prefix = "" );


   	function f;
};

template< typename function, int Dimensions, typename Real >
class SDFSchemeTest
{

};

template< typename function, int Dimensions, typename Real >
class SDFSchemeTest< function, 1, Real > : public SDFSchemeTestBase< function, Real >
{
   public:


   enum { Dimensions = 1 };
   typedef Point PointType;
   typedef typename PointType::RealType RealType;

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getValue( const PointType& v,
           const Real& time = 0.0 ) const;



};

template< typename function, int Dimensions, typename Real >
class SDFSchemeTest< function, 2, Real > : public SDFSchemeTestBase< function, Real >
{
   public:


   enum { Dimensions = 2 };
   typedef Point PointType;
   typedef typename PointType::RealType RealType;

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getValue( const PointType& v,
           const Real& time = 0.0 ) const;


};

template< typename function, int Dimensions, typename Real >
class SDFSchemeTest< function, 3, Real > : public SDFSchemeTestBase< function,  Real >
{
   public:


   enum { Dimensions = 3 };
   typedef Point PointType;
   typedef typename PointType::RealType RealType;

   template< int XDiffOrder = 0,
             int YDiffOrder = 0,
             int ZDiffOrder = 0 >
   RealType getValue( const PointType& v,
           const Real& time = 0.0 ) const;

};

      } // namespace Analytic
   } // namespace Functions
} // namespace noa::TNL

#include <functions/SDFSchemeTest_impl.h>
