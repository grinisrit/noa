// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>
#include <noa/3rdparty/TNL/Config/ConfigDescription.h>
#include <noa/3rdparty/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Functions {

template< int FunctionDimension,
          typename Real = double,
          typename Device = Devices::Host >
class TestFunction : public Domain< FunctionDimension, SpaceDomain >
{
   protected:

      enum TestFunctions{ constant,
                          paraboloid,
                          expBump,
                          sinBumps,
                          sinWave,
                          cylinder,
                          flowerpot,
                          twins,
                          pseudoSquare,
                          blob,
                          vectorNorm,
                          paraboloidSDF,
                          sinWaveSDF,
                          sinBumpsSDF };

      enum TimeDependence { none,
                            linear,
                            quadratic,
                            cosine };

      enum Operators { identity,
                       heaviside,
                       smoothHeaviside };

   public:

      enum{ Dimension = FunctionDimension };
      typedef Real RealType;
      typedef Containers::StaticVector< Dimension, Real > PointType;

      TestFunction();

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const Config::ParameterContainer& parameters,
                 const String& prefix = "" );

      const TestFunction& operator = ( const TestFunction& function );

      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      Real getPartialDerivative( const PointType& vertex,
                                 const Real& time = 0 ) const;

      __cuda_callable__
      Real operator()( const PointType& vertex,
                     const Real& time = 0 ) const
      {
         return this->getPartialDerivative< 0, 0, 0 >( vertex, time );
      }


      template< int XDiffOrder = 0,
                int YDiffOrder = 0,
                int ZDiffOrder = 0 >
      __cuda_callable__
      Real getTimeDerivative( const PointType& vertex,
                              const Real& time = 0 ) const;

      std::ostream& print( std::ostream& str ) const;

      ~TestFunction();

   protected:

      template< typename FunctionType >
      bool setupFunction( const Config::ParameterContainer& parameters,
                         const String& prefix = "" );
      
      template< typename OperatorType >
      bool setupOperator( const Config::ParameterContainer& parameters,
                          const String& prefix = "" );

      template< typename FunctionType >
      void deleteFunction();

      template< typename OperatorType >
      void deleteOperator();

      void deleteFunctions();

      template< typename FunctionType >
      void copyFunction( const void* function );

      template< typename FunctionType >
      std::ostream& printFunction( std::ostream& str ) const;

      void* function;

      void* operator_;

      TestFunctions functionType;
      
      Operators operatorType;

      TimeDependence timeDependence;

      Real timeScale;

};

template< int FunctionDimension,
          typename Real,
          typename Device >
std::ostream& operator << ( std::ostream& str, const TestFunction< FunctionDimension, Real, Device >& f )
{
   str << "Test function: ";
   return f.print( str );
}

} // namespace Functions
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Functions/TestFunction_impl.h>

