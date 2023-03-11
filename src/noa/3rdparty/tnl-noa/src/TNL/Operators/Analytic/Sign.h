// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Domain.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

namespace noa::TNL {
namespace Operators {
namespace Analytic {

template< int Dimensions, typename Real >
class Sign : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
public:
   typedef Real RealType;
   typedef Containers::StaticVector< Dimensions, RealType > PointType;

   Sign() : positiveValue( 1.0 ), negativeValue( -1.0 ), zeroValue( 0.0 ) {}

   static void
   configSetup( Config::ConfigDescription& config, const String& prefix = "" )
   {
      config.addEntry< double >( prefix + "positive-value", "Value returned for positive argument.", 1.0 );
      config.addEntry< double >( prefix + "negative-value", "Value returned for negative argument.", -1.0 );
      config.addEntry< double >( prefix + "zero-value", "Value returned for zero argument.", 0.0 );
   }

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" )
   {
      this->positiveValue = parameters.getParameter< double >( prefix + "positive-value" );
      this->negativeValue = parameters.getParameter< double >( prefix + "negative-value" );
      this->zeroValue = parameters.getParameter< double >( prefix + "zero-value" );
      return true;
   };

   void
   setPositiveValue( const RealType& value )
   {
      this->positiveValue = value;
   }

   const RealType&
   getPositiveValue() const
   {
      return this->positiveValue;
   }

   void
   setNegativeValue( const RealType& value )
   {
      this->negativeValue = value;
   }

   const RealType&
   getNegativeValue() const
   {
      return this->negativeValue;
   }

   void
   setZeroValue( const RealType& value )
   {
      this->zeroValue = value;
   }

   const RealType&
   getZeroValue() const
   {
      return this->zeroValue;
   }

   template< typename Function >
   __cuda_callable__
   RealType
   operator()( const Function& function, const PointType& vertex, const RealType& time = 0 ) const
   {
      const RealType aux = function( vertex, time );
      if( aux > 0.0 )
         return this->positiveValue;
      else if( aux < 0.0 )
         return this->negativeValue;
      return this->zeroValue;
   }

   template< typename Function, int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const Function& function, const PointType& vertex, const RealType& time = 0 ) const
   {
      if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
         return this->operator()( function, vertex, time );
      return 0.0;
   }

protected:
   RealType positiveValue, negativeValue, zeroValue;
};

}  // namespace Analytic
}  // namespace Operators
}  // namespace noa::TNL
