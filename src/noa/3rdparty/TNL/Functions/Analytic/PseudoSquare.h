// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/TNL/Functions/Domain.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>

namespace noaTNL {
namespace Functions {
namespace Analytic {   

template< typename Real,
          int Dimension >
class PseudoSquareBase : public Domain< Dimension, SpaceDomain >
{
   public:

      typedef Real RealType;

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

   protected:

      RealType height;
};

template< int Dimension,
          typename Real >
class PseudoSquare
{
};

template< typename Real >
class PseudoSquare< 1, Real > : public PseudoSquareBase< Real, 1 >
{
   public:

      enum { Dimension = 1 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      PseudoSquare();

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

template< typename Real >
class PseudoSquare< 2, Real > : public PseudoSquareBase< Real, 2 >
{
   public:

      enum { Dimension = 2 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      PseudoSquare();

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

template< typename Real >
class PseudoSquare< 3, Real > : public PseudoSquareBase< Real, 3 >
{
   public:

      enum { Dimension = 3 };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      PseudoSquare();

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
std::ostream& operator << ( std::ostream& str, const PseudoSquare< Dimension, Real >& f )
{
   str << "Level-set pseudo square function.";
   return str;
}

} // namespace Analytic
} // namespace Functions
} // namepsace TNL

#include <noa/3rdparty/TNL/Functions/Analytic/PseudoSquare_impl.h>
