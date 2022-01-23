// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Functions/Domain.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Operators {
namespace Analytic {   
   
   
template< int Dimensions,
          typename Real = double >
class Heaviside : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
      
      typedef Real RealType;
      typedef Containers::StaticVector< Dimensions, 
                                        RealType > PointType;
      
      Heaviside() : multiplicator( 1.0 ) {}
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "multiplicator", "Outer multiplicator of the Heaviside operator - -1.0 turns the function graph upside/down.", 1.0 );
      }      
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->multiplicator = parameters.getParameter< double >( prefix + "multiplicator" );
         return true;
      };
      
      
      template< typename Function >
      __cuda_callable__
      RealType operator()( const Function& function,
                           const PointType& vertex,
                           const RealType& time = 0 ) const
      {
         const RealType aux = function( vertex, time );
         if( aux > 0.0 )
            return 1.0;
         return 0.0;
      }
      
      template< typename Function,
                int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      RealType getPartialDerivative( const Function& function,
                                     const PointType& vertex,
                                     const RealType& time = 0 ) const
      {
         if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
            return this->operator()( function, vertex, time );
         return 0.0;
      }
      
   protected:
      
      RealType multiplicator;
      
};

} // namespace Analytic
} // namespace Operators
} // namespace noa::TNL