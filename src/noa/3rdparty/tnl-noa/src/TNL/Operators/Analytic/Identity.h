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
class Identity : public Functions::Domain< Dimensions, Functions::SpaceDomain >
{
public:
   using RealType = Real;
   using PointType = Containers::StaticVector< Dimensions, RealType >;

   bool
   setup( const Config::ParameterContainer& parameters, const String& prefix = "" )
   {
      return true;
   }

   template< typename Function >
   __cuda_callable__
   RealType
   operator()( const Function& function, const PointType& vertex, const RealType& time = 0 ) const
   {
      return function( vertex, time );
   }

   template< typename Function, int XDiffOrder = 0, int YDiffOrder = 0, int ZDiffOrder = 0 >
   __cuda_callable__
   RealType
   getPartialDerivative( const Function& function, const PointType& vertex, const RealType& time = 0 ) const
   {
      return function.template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
   }
};

}  // namespace Analytic
}  // namespace Operators
}  // namespace noa::TNL
