// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once 

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< int dimensions,
          typename Real >
class ExpBumpBase : public Domain< dimensions, SpaceDomain >
{
   public:
 
      typedef Real RealType;
 
      ExpBumpBase();
 
      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setAmplitude( const RealType& amplitude );

      const RealType& getAmplitude() const;

      void setSigma( const RealType& sigma );

      const RealType& getSigma() const;

   protected:

      RealType amplitude, sigma;
};

template< int Dimension,
          typename Real >
class ExpBump
{
};

template< typename Real >
class ExpBump< 1, Real > : public ExpBumpBase< 1, Real >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< 1, RealType > PointType;

      ExpBump();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
   __cuda_callable__
   RealType getPartialDerivative( const PointType& v,
                                  const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const PointType& v,
                        const RealType& time = 0.0 ) const;
};

template< typename Real >
class ExpBump< 2, Real > : public ExpBumpBase< 2, Real >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< 2, RealType > PointType;

      ExpBump();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
   __cuda_callable__ inline
   RealType getPartialDerivative( const PointType& v,
                                  const Real& time = 0.0 ) const;
 
   __cuda_callable__
   RealType operator()( const PointType& v,
                        const Real& time = 0.0 ) const;
};

template< typename Real >
class ExpBump< 3, Real > : public ExpBumpBase< 3, Real >
{
   public:
 
      typedef Real RealType;
      typedef Containers::StaticVector< 3, RealType > PointType;

      ExpBump();

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

template< int Dimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const ExpBump< Dimension, Real >& f )
{
   str << "ExpBump. function: amplitude = " << f.getAmplitude() << " sigma = " << f.getSigma();
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/ExpBump_impl.h>


