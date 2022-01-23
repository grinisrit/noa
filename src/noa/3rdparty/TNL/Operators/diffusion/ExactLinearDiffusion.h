// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Szekely Ondrej, ondra.szekely@gmail.com
 */

#pragma once

#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Operators {   

template< int Dimension >
class ExactLinearDiffusion
{};

template<>
class ExactLinearDiffusion< 1 > : public Functions::Domain< 1, Functions::SpaceDomain >
{
   public:

      static const int Dimension = 1;
 
      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::PointType& v,
                                              const typename Function::RealType& time = 0.0 ) const;
};

template<>
class ExactLinearDiffusion< 2 > : public Functions::Domain< 2, Functions::SpaceDomain >
{
   public:
 
      static const int Dimension = 2;

      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::PointType& v,
                                              const typename Function::RealType& time = 0.0 ) const;
};

template<>
class ExactLinearDiffusion< 3 > : public Functions::Domain< 3 >
{
   public:
 
      static const int Dimension = 3;

      template< typename Function >
      __cuda_callable__ inline
      typename Function::RealType operator()( const Function& function,
                                              const typename Function::PointType& v,
                                              const typename Function::RealType& time = 0.0 ) const;
};

} // namespace Operators
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Operators/diffusion/ExactLinearDiffusion_impl.h>
