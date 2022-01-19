// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Functions/Domain.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< typename Real,
          int Dimension >
class FlowerpotBase : public Domain< Dimension, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      void setDiameter( const RealType& sigma );

      const RealType& getDiameter() const;

   protected:

      RealType diameter;
};

template< int Dimension,
          typename Real >
class Flowerpot
{
};

template< typename Real >
class Flowerpot< 1, Real > : public FlowerpotBase< Real, 1 >
{
   public:

      enum { Dimension = 1 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      Flowerpot();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class Flowerpot< 2, Real > : public FlowerpotBase< Real, 2 >
{
   public:

      enum { Dimension = 2 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      Flowerpot();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< typename Real >
class Flowerpot< 3, Real > : public FlowerpotBase< Real, 3 >
{
   public:

      enum { Dimension = 3 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      Flowerpot();

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0,
                typename Point = PointType >
      __cuda_callable__
      RealType getPartialDerivative( const Point& v,
                                     const Real& time = 0.0 ) const;
 
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const Real& time = 0.0 ) const;
 
};

template< int Dimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const Flowerpot< Dimension, Real >& f )
{
   str << "Flowerpot function.";
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL

#include <TNL/Functions/Analytic/Flowerpot_impl.h>

