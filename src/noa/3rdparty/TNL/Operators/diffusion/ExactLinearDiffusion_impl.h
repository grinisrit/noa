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

namespace noa::TNL {
namespace Operators {

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
ExactLinearDiffusion< 1 >::
operator()( const Function& function,
            const typename Function::PointType& v,
            const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time );
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
ExactLinearDiffusion< 2 >::
operator()( const Function& function,
            const typename Function::PointType& v,
          const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 2, 0 >( v, time );
}

template< typename Function >
__cuda_callable__ inline
typename Function::RealType
ExactLinearDiffusion< 3 >::
operator()( const Function& function,
            const typename Function::PointType& v,
            const typename Function::RealType& time ) const
{
   return function.template getPartialDerivative< 2, 0, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 2, 0 >( v, time ) +
          function.template getPartialDerivative< 0, 0, 2 >( v, time );

}

} // namespace Operators
} // namespace noa::TNL
