// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once 

#include <iostream>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Functions {
namespace Analytic {   

template< int dimensions,
          typename Real = double >
class Constant : public Domain< dimensions, NonspaceDomain >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< dimensions, RealType > PointType;
 
      __cuda_callable__
      Constant();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setConstant( const RealType& constant );

      const RealType& getConstant() const;

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__ inline
      RealType getPartialDerivative( const PointType& v,
                                     const Real& time = 0.0 ) const;

      __cuda_callable__ inline
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const
      {
         return constant;
      }
 
       __cuda_callable__ inline
      RealType getValue( const Real& time = 0.0 ) const
      {
          return constant;
      }

   protected:

      RealType constant;
};

template< int dimensions,
          typename Real >
std::ostream& operator << ( std::ostream& str, const Constant< dimensions, Real >& f )
{
   str << "Constant function: constant = " << f.getConstant();
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Functions/Analytic/Constant_impl.h>

