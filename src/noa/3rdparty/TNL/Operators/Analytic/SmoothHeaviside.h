// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Operators {
namespace Analytic {   
   
   
template< int Dimensions,
          typename Real = double >
class SmoothHeaviside : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
   public:
      
      typedef Real RealType;
      typedef Containers::StaticVector< Dimensions, 
                                        RealType > PointType;
      
      SmoothHeaviside() : sharpness( 1.0 ) {}
      
      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         config.addEntry< double >( prefix + "sharpness", "sharpness of smoothening", 1.0 );
      }

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         this->sharpness = parameters.getParameter< double >( prefix + "sharpness" );
         return true;
      }
      
      
      void setSharpness( const RealType& sharpness )
      {
         this->sharpness = sharpness;
      }
      
      __cuda_callable__
      const RealType getShaprness() const
      {
         return this->sharpness;
      }
      
      template< typename Function >
      __cuda_callable__
      RealType operator()( const Function& function,
                           const PointType& vertex,
                           const RealType& time = 0 ) const
      {
         const RealType aux = function( vertex, time );
         return 1.0 / ( 1.0 + std::exp( -2.0 * sharpness * aux ) );
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
         // TODO: implement the rest
      }
      
   protected:

      RealType sharpness;
};

} // namespace Analytic
} // namespace Operators
} // namespace TNL
