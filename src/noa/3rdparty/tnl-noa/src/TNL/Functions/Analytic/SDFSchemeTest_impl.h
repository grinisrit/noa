// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functions/SDFSchemeTest.h>

namespace noa::TNL {
namespace Functions {
namespace Analytic {

template< typename function, typename Real >
SDFSchemeTestBase< function, Real >::SDFSchemeTestBase()
{}

template< typename function, typename Real >
bool
SDFSchemeTestBase< function, Real >::v( const Config::ParameterContainer& parameters, const String& prefix = "" )
{
   f.init( parameters );

   return true;
}

template< typename function, int Dimensions, typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
Real
SDFSchemeTest< function, 1, Real >::getValue( const Point& v, const Real& time = 0.0 ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 || XDiffOrder != 0 )
      return 0.0;

   return sign( this->f.getValue< 0, 0, 0 >( v ) )
        * ( 1 - sqrt( this->f.getValue< 1, 0, 0 >( v ) * this->f.getValue< 1, 0, 0 >( v ) ) );
}

template< typename function, int Dimensions, typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
Real
SDFSchemeTest< function, 2, Real >::getValue( const Point& v, const Real& time = 0.0 ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 || XDiffOrder != 0 )
      return 0.0;

   return sign( this->f.getValue< 0, 0, 0 >( v ) )
        * ( 1
            - sqrt( this->f.getValue< 1, 0, 0 >( v ) * this->f.getValue< 1, 0, 0 >( v )
                    + this->f.getValue< 0, 1, 0 >( v ) * this->f.getValue< 0, 1, 0 >( v ) ) );
}

template< typename function, int Dimensions, typename Real >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
Real
SDFSchemeTest< function, 3, Real >::getValue( const Point& v, const Real& time = 0.0 ) const
{
   if( YDiffOrder != 0 || ZDiffOrder != 0 || XDiffOrder != 0 )
      return 0.0;

   return sign( this->f.getValue< 0, 0, 0 >( v ) )
        * ( 1.0
            - sqrt( this->f.getValue< 1, 0, 0 >( v ) * this->f.getValue< 1, 0, 0 >( v )
                    + this->f.getValue< 0, 1, 0 >( v ) * this->f.getValue< 0, 1, 0 >( v )
                    + this->f.getValue< 0, 0, 1 >( v ) * this->f.getValue< 0, 0, 1 >( v ) ) );
}

}  // namespace Analytic
}  // namespace Functions
}  // namespace noa::TNL
