// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Operators {

template< int Dimension, typename Real = double >
class ExactGradientNorm
{};

/****
 * 1D
 */
template< typename Real >
class ExactGradientNorm< 1, Real > : public Functions::Domain< 1, Functions::SpaceDomain >
{
public:
   ExactGradientNorm() : epsilonSquare( 0.0 ){};

   void
   setRegularizationEpsilon( const Real& epsilon )
   {
      this->epsilonSquare = epsilon * epsilon;
   }

   template< typename Function >
   __cuda_callable__
   typename Function::RealType
   operator()( const Function& function,
               const typename Function::PointType& v,
               const typename Function::RealType& time = 0.0 ) const
   {
      typedef typename Function::RealType RealType;
      const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
      return ::sqrt( this->epsilonSquare + f_x * f_x );
   }

   template< typename Function, int XDerivative = 0, int YDerivative = 0, int ZDerivative = 0 >
   __cuda_callable__
   typename Function::RealType
   getPartialDerivative( const Function& function,
                         const typename Function::PointType& v,
                         const typename Function::RealType& time = 0.0 ) const
   {
      static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
                     "Partial derivative must be non-negative integer." );
      static_assert( XDerivative < 2, "Partial derivative of higher order then 1 are not implemented yet." );
      typedef typename Function::RealType RealType;

      if( XDerivative == 1 ) {
         const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         const RealType f_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time );
         const RealType Q = ::sqrt( this->epsilonSquare + f_x * f_x );
         return ( f_x * f_xx ) / Q;
      }
      if( XDerivative == 0 )
         return this->operator()( function, v, time );
      if( YDerivative != 0 || ZDerivative != 0 )
         return 0.0;
   }

protected:
   Real epsilonSquare;
};

/****
 * 2D
 */
template< typename Real >
class ExactGradientNorm< 2, Real > : public Functions::Domain< 2, Functions::SpaceDomain >
{
public:
   ExactGradientNorm() : epsilonSquare( 0.0 ){};

   void
   setRegularizationEpsilon( const Real& epsilon )
   {
      this->epsilonSquare = epsilon * epsilon;
   }

   template< typename Function >
   __cuda_callable__
   typename Function::RealType
   operator()( const Function& function,
               const typename Function::PointType& v,
               const typename Function::RealType& time = 0.0 ) const
   {
      typedef typename Function::RealType RealType;
      const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
      const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
      return ::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y );
   }

   template< typename Function, int XDerivative = 0, int YDerivative = 0, int ZDerivative = 0 >
   __cuda_callable__
   typename Function::RealType
   getPartialDerivative( const Function& function,
                         const typename Function::PointType& v,
                         const typename Function::RealType& time = 0.0 ) const
   {
      static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
                     "Partial derivative must be non-negative integer." );
      static_assert( XDerivative < 2 && YDerivative < 2, "Partial derivative of higher order then 1 are not implemented yet." );
      typedef typename Function::RealType RealType;

      if( XDerivative == 1 && YDerivative == 0 ) {
         const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
         const RealType f_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time );
         const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
         return ( f_x * f_xx + f_y * f_xy ) / ::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y );
      }
      if( XDerivative == 0 && YDerivative == 1 ) {
         const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
         const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
         const RealType f_yy = function.template getPartialDerivative< 0, 2, 0 >( v, time );
         return ( f_x * f_xy + f_y * f_yy ) / ::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y );
      }
      if( XDerivative == 0 && YDerivative == 0 )
         return this->operator()( function, v, time );
      if( ZDerivative > 0 )
         return 0.0;
   }

protected:
   Real epsilonSquare;
};

template< typename Real >
class ExactGradientNorm< 3, Real > : public Functions::Domain< 3, Functions::SpaceDomain >
{
public:
   ExactGradientNorm() : epsilonSquare( 0.0 ){};

   void
   setRegularizationEpsilon( const Real& epsilon )
   {
      this->epsilonSquare = epsilon * epsilon;
   }

   template< typename Function >
   __cuda_callable__
   typename Function::RealType
   operator()( const Function& function,
               const typename Function::PointType& v,
               const typename Function::RealType& time = 0.0 ) const
   {
      typedef typename Function::RealType RealType;
      const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
      const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
      const RealType f_z = function.template getPartialDerivative< 0, 0, 1 >( v, time );
      return std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
   }

   template< typename Function, int XDerivative = 0, int YDerivative = 0, int ZDerivative = 0 >
   __cuda_callable__
   typename Function::RealType
   getPartialDerivative( const Function& function,
                         const typename Function::PointType& v,
                         const typename Function::RealType& time = 0.0 ) const
   {
      static_assert( XDerivative >= 0 && YDerivative >= 0 && ZDerivative >= 0,
                     "Partial derivative must be non-negative integer." );
      static_assert( XDerivative < 2 && YDerivative < 2 && ZDerivative < 2,
                     "Partial derivative of higher order then 1 are not implemented yet." );

      typedef typename Function::RealType RealType;
      if( XDerivative == 1 && YDerivative == 0 && ZDerivative == 0 ) {
         const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
         const RealType f_z = function.template getPartialDerivative< 0, 0, 1 >( v, time );
         const RealType f_xx = function.template getPartialDerivative< 2, 0, 0 >( v, time );
         const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
         const RealType f_xz = function.template getPartialDerivative< 1, 0, 1 >( v, time );
         return ( f_x * f_xx + f_y * f_xy + f_z * f_xz ) / std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
      }
      if( XDerivative == 0 && YDerivative == 1 && ZDerivative == 0 ) {
         const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
         const RealType f_z = function.template getPartialDerivative< 0, 0, 1 >( v, time );
         const RealType f_xy = function.template getPartialDerivative< 1, 1, 0 >( v, time );
         const RealType f_yy = function.template getPartialDerivative< 0, 2, 0 >( v, time );
         const RealType f_yz = function.template getPartialDerivative< 0, 1, 1 >( v, time );
         return ( f_x * f_xy + f_y * f_yy + f_z * f_yz ) / std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
      }
      if( XDerivative == 0 && YDerivative == 0 && ZDerivative == 1 ) {
         const RealType f_x = function.template getPartialDerivative< 1, 0, 0 >( v, time );
         const RealType f_y = function.template getPartialDerivative< 0, 1, 0 >( v, time );
         const RealType f_z = function.template getPartialDerivative< 0, 0, 1 >( v, time );
         const RealType f_xz = function.template getPartialDerivative< 1, 0, 1 >( v, time );
         const RealType f_yz = function.template getPartialDerivative< 0, 1, 1 >( v, time );
         const RealType f_zz = function.template getPartialDerivative< 0, 0, 2 >( v, time );
         return ( f_x * f_xz + f_y * f_yz + f_z * f_zz ) / std::sqrt( this->epsilonSquare + f_x * f_x + f_y * f_y + f_z * f_z );
      }
      if( XDerivative == 0 && YDerivative == 0 && ZDerivative == 0 )
         return this->operator()( function, v, time );
   }

protected:
   Real epsilonSquare;
};

}  // namespace Operators
}  // namespace noa::TNL
