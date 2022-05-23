// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Allocators/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/MultiDeviceMemoryOperations.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Constant.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/ExpBump.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/SinBumps.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/SinWave.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Constant.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/ExpBump.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/SinBumps.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/SinWave.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Cylinder.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Flowerpot.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Twins.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Blob.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/PseudoSquare.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/Paraboloid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/VectorNorm.h>
/****
 * The signed distance test functions
 */
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/SinBumpsSDF.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/SinWaveSDF.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Analytic/ParaboloidSDF.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Operators/Analytic/Identity.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/Analytic/Heaviside.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/Analytic/SmoothHeaviside.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

#include "TestFunction.h"

namespace noa::TNL {
namespace Functions {

template< int FunctionDimension, typename Real, typename Device >
TestFunction< FunctionDimension, Real, Device >::TestFunction()
: function( nullptr ), operator_( nullptr ), timeDependence( none ), timeScale( 1.0 )
{}

template< int FunctionDimension, typename Real, typename Device >
void
TestFunction< FunctionDimension, Real, Device >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   config.addRequiredEntry< String >( prefix + "test-function", "Testing function." );
   config.addEntryEnum( "constant" );
   config.addEntryEnum( "paraboloid" );
   config.addEntryEnum( "exp-bump" );
   config.addEntryEnum( "sin-wave" );
   config.addEntryEnum( "sin-bumps" );
   config.addEntryEnum( "cylinder" );
   config.addEntryEnum( "flowerpot" );
   config.addEntryEnum( "twins" );
   config.addEntryEnum( "pseudoSquare" );
   config.addEntryEnum( "blob" );
   config.addEntryEnum( "paraboloid-sdf" );
   config.addEntryEnum( "sin-wave-sdf" );
   config.addEntryEnum( "sin-bumps-sdf" );
   config.addEntryEnum( "heaviside-of-vector-norm" );
   config.addEntryEnum( "smooth-heaviside-of-vector-norm" );

   config.addEntry< double >( prefix + "constant", "Value of the constant function.", 0.0 );
   config.addEntry< double >( prefix + "wave-length", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry< double >( prefix + "wave-length-x", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry< double >( prefix + "wave-length-y", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry< double >( prefix + "wave-length-z", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry< double >( prefix + "phase", "Phase of the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "phase-x", "Phase of the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "phase-y", "Phase of the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "phase-z", "Phase of the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "amplitude", "Amplitude length of the sine based test functions.", 1.0 );
   config.addEntry< double >( prefix + "waves-number", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "waves-number-x", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "waves-number-y", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "waves-number-z", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry< double >( prefix + "sigma", "Sigma for the exp based test functions.", 1.0 );
   config.addEntry< double >( prefix + "radius", "Radius for paraboloids.", 1.0 );
   config.addEntry< double >( prefix + "coefficient", "Coefficient for paraboloids.", 1.0 );
   config.addEntry< double >( prefix + "x-center", "x-center for paraboloids.", 0.0 );
   config.addEntry< double >( prefix + "y-center", "y-center for paraboloids.", 0.0 );
   config.addEntry< double >( prefix + "z-center", "z-center for paraboloids.", 0.0 );
   config.addEntry< double >( prefix + "diameter", "Diameter for the cylinder, flowerpot test functions.", 1.0 );
   config.addEntry< double >(
      prefix + "height", "Height of zero-level-set function for the blob, pseudosquare test functions.", 1.0 );
   Analytic::VectorNorm< 3, double >::configSetup( config, "vector-norm-" );
   TNL::Operators::Analytic::Heaviside< 3, double >::configSetup( config, "heaviside-" );
   TNL::Operators::Analytic::SmoothHeaviside< 3, double >::configSetup( config, "smooth-heaviside-" );
   config.addEntry< String >( prefix + "time-dependence", "Time dependence of the test function.", "none" );
   config.addEntryEnum( "none" );
   config.addEntryEnum( "linear" );
   config.addEntryEnum( "quadratic" );
   config.addEntryEnum( "cosine" );
   config.addEntry< double >( prefix + "time-scale", "Time scaling for the time dependency of the test function.", 1.0 );
}

template< int FunctionDimension, typename Real, typename Device >
template< typename FunctionType >
bool
TestFunction< FunctionDimension, Real, Device >::setupFunction( const Config::ParameterContainer& parameters,
                                                                const String& prefix )
{
   FunctionType* auxFunction = new FunctionType;
   if( ! auxFunction->setup( parameters, prefix ) ) {
      delete auxFunction;
      return false;
   }

   if( std::is_same< Device, Devices::Host >::value ) {
      this->function = auxFunction;
   }
   if( std::is_same< Device, Devices::Cuda >::value ) {
      this->function = Allocators::Cuda< FunctionType >{}.allocate( 1 );
      Algorithms::MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::copy(
         (FunctionType*) this->function, (FunctionType*) auxFunction, 1 );
      delete auxFunction;
      TNL_CHECK_CUDA_DEVICE;
   }
   return true;
}

template< int FunctionDimension, typename Real, typename Device >
template< typename OperatorType >
bool
TestFunction< FunctionDimension, Real, Device >::setupOperator( const Config::ParameterContainer& parameters,
                                                                const String& prefix )
{
   OperatorType* auxOperator = new OperatorType;
   if( ! auxOperator->setup( parameters, prefix ) ) {
      delete auxOperator;
      return false;
   }

   if( std::is_same< Device, Devices::Host >::value ) {
      this->operator_ = auxOperator;
   }
   if( std::is_same< Device, Devices::Cuda >::value ) {
      this->operator_ = Allocators::Cuda< OperatorType >{}.allocate( 1 );
      Algorithms::MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::copy(
         (OperatorType*) this->operator_, (OperatorType*) auxOperator, 1 );
      delete auxOperator;
      TNL_CHECK_CUDA_DEVICE;
   }
   return true;
}

template< int FunctionDimension, typename Real, typename Device >
bool
TestFunction< FunctionDimension, Real, Device >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   using namespace noa::TNL::Functions::Analytic;
   using namespace noa::TNL::Operators::Analytic;
   std::cout << "Test function setup ... " << std::endl;
   const String& timeDependence = parameters.getParameter< String >( prefix + "time-dependence" );
   std::cout << "Time dependence ... " << timeDependence << std::endl;
   if( timeDependence == "none" )
      this->timeDependence = none;
   if( timeDependence == "linear" )
      this->timeDependence = linear;
   if( timeDependence == "quadratic" )
      this->timeDependence = quadratic;
   if( timeDependence == "cosine" )
      this->timeDependence = cosine;

   this->timeScale = parameters.getParameter< double >( prefix + "time-scale" );

   const String& testFunction = parameters.getParameter< String >( prefix + "test-function" );
   std::cout << "Test function ... " << testFunction << std::endl;
   if( testFunction == "constant" ) {
      using FunctionType = Constant< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = constant;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "paraboloid" ) {
      using FunctionType = Paraboloid< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = paraboloid;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "exp-bump" ) {
      using FunctionType = ExpBump< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = expBump;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-bumps" ) {
      using FunctionType = SinBumps< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = sinBumps;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-wave" ) {
      using FunctionType = SinWave< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = sinWave;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "cylinder" ) {
      using FunctionType = Cylinder< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = cylinder;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "flowerpot" ) {
      using FunctionType = Flowerpot< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = flowerpot;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "twins" ) {
      using FunctionType = Twins< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = twins;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "pseudoSquare" ) {
      using FunctionType = PseudoSquare< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = pseudoSquare;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "blob" ) {
      using FunctionType = Blob< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = blob;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "paraboloid-sdf" ) {
      using FunctionType = ParaboloidSDF< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = paraboloidSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-bumps-sdf" ) {
      using FunctionType = SinBumpsSDF< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = sinBumpsSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-wave-sdf" ) {
      using FunctionType = SinWaveSDF< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = sinWaveSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "vector-norm" ) {
      using FunctionType = VectorNorm< Dimension, Real >;
      using OperatorType = Identity< Dimension, Real >;
      functionType = vectorNorm;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "heaviside-of-vector-norm" ) {
      using FunctionType = VectorNorm< Dimension, Real >;
      using OperatorType = Heaviside< Dimension, Real >;
      functionType = vectorNorm;
      operatorType = heaviside;
      return ( setupFunction< FunctionType >( parameters, prefix + "vector-norm-" )
               && setupOperator< OperatorType >( parameters, prefix + "heaviside-" ) );
   }
   if( testFunction == "smooth-heaviside-of-vector-norm" ) {
      using FunctionType = VectorNorm< Dimension, Real >;
      using OperatorType = SmoothHeaviside< Dimension, Real >;
      functionType = vectorNorm;
      operatorType = smoothHeaviside;
      return ( setupFunction< FunctionType >( parameters, prefix + "vector-norm-" )
               && setupOperator< OperatorType >( parameters, prefix + "smooth-heaviside-" ) );
   }
   std::cerr << "Unknown function " << testFunction << std::endl;
   return false;
}

template< int FunctionDimension, typename Real, typename Device >
const TestFunction< FunctionDimension, Real, Device >&
TestFunction< FunctionDimension, Real, Device >::operator=( const TestFunction& function )
{
   /*****
    * TODO: if the function is on the device we cannot do the following
    */
   abort();
   using namespace noa::TNL::Functions::Analytic;
   this->functionType = function.functionType;
   this->timeDependence = function.timeDependence;
   this->timeScale = function.timeScale;

   this->deleteFunctions();

   switch( this->functionType ) {
      case constant:
         this->copyFunction< Constant< FunctionDimension, Real > >( function.function );
         break;
      case expBump:
         this->copyFunction< ExpBump< FunctionDimension, Real > >( function.function );
         break;
      case sinBumps:
         this->copyFunction< SinBumps< FunctionDimension, Real > >( function.function );
         break;
      case sinWave:
         this->copyFunction< SinWave< FunctionDimension, Real > >( function.function );
         break;
      case cylinder:
         this->copyFunction< Cylinder< FunctionDimension, Real > >( function.function );
         break;
      case flowerpot:
         this->copyFunction< Flowerpot< FunctionDimension, Real > >( function.function );
         break;
      case twins:
         this->copyFunction< Twins< FunctionDimension, Real > >( function.function );
         break;
      case pseudoSquare:
         this->copyFunction< PseudoSquare< FunctionDimension, Real > >( function.function );
         break;
      case blob:
         this->copyFunction< Blob< FunctionDimension, Real > >( function.function );
         break;

      case paraboloidSDF:
         this->copyFunction< Paraboloid< FunctionDimension, Real > >( function.function );
         break;
      case sinBumpsSDF:
         this->copyFunction< SinBumpsSDF< FunctionDimension, Real > >( function.function );
         break;
      case sinWaveSDF:
         this->copyFunction< SinWaveSDF< FunctionDimension, Real > >( function.function );
         break;
      default:
         throw std::invalid_argument( "Unknown function type." );
         break;
   }
}

template< int FunctionDimension, typename Real, typename Device >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
__cuda_callable__
Real
TestFunction< FunctionDimension, Real, Device >::getPartialDerivative( const PointType& vertex, const Real& time ) const
{
   TNL_ASSERT_TRUE( this->function, "The test function was not set properly." );
   using namespace noa::TNL::Functions::Analytic;
   using namespace noa::TNL::Operators::Analytic;
   Real scale( 1.0 );
   switch( this->timeDependence ) {
      case none:
         break;
      case linear:
         scale = 1.0 - this->timeScale * time;
         break;
      case quadratic:
         scale = this->timeScale * time;
         scale *= scale;
         scale = 1.0 - scale;
         break;
      case cosine:
         scale = ::cos( this->timeScale * time );
         break;
   }
   switch( functionType ) {
      case constant:
         {
            using FunctionType = Constant< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case paraboloid:
         {
            using FunctionType = Paraboloid< Dimension, Real >;
            if( operatorType == identity ) {
               using OperatorType = Identity< Dimension, Real >;

               return scale
                    * ( (OperatorType*) this->operator_ )
                         ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                            *(FunctionType*) this->function, vertex, time );
            }
            if( operatorType == heaviside ) {
               using OperatorType = Heaviside< Dimension, Real >;

               return scale
                    * ( (OperatorType*) this->operator_ )
                         ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                            *(FunctionType*) this->function, vertex, time );
            }
            if( operatorType == smoothHeaviside ) {
               using OperatorType = SmoothHeaviside< Dimension, Real >;

               return scale
                    * ( (OperatorType*) this->operator_ )
                         ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                            *(FunctionType*) this->function, vertex, time );
            }
         }
      case expBump:
         {
            using FunctionType = ExpBump< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case sinBumps:
         {
            using FunctionType = SinBumps< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case sinWave:
         {
            using FunctionType = SinWave< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case cylinder:
         {
            using FunctionType = Cylinder< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case flowerpot:
         {
            using FunctionType = Flowerpot< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case twins:
         {
            using FunctionType = Twins< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case pseudoSquare:
         {
            using FunctionType = PseudoSquare< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case blob:
         {
            using FunctionType = Blob< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case vectorNorm:
         {
            using FunctionType = VectorNorm< Dimension, Real >;
            if( operatorType == identity ) {
               using OperatorType = Identity< Dimension, Real >;

               return scale
                    * ( (OperatorType*) this->operator_ )
                         ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                            *(FunctionType*) this->function, vertex, time );
            }
            if( operatorType == heaviside ) {
               using OperatorType = Heaviside< Dimension, Real >;

               return scale
                    * ( (OperatorType*) this->operator_ )
                         ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                            *(FunctionType*) this->function, vertex, time );
            }
            if( operatorType == smoothHeaviside ) {
               using OperatorType = SmoothHeaviside< Dimension, Real >;

               return scale
                    * ( (OperatorType*) this->operator_ )
                         ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                            *(FunctionType*) this->function, vertex, time );
            }
         }
      case sinBumpsSDF:
         {
            using FunctionType = SinBumpsSDF< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case sinWaveSDF:
         {
            using FunctionType = SinWaveSDF< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }
      case paraboloidSDF:
         {
            using FunctionType = ParaboloidSDF< Dimension, Real >;
            using OperatorType = Identity< Dimension, Real >;

            return scale
                 * ( (OperatorType*) this->operator_ )
                      ->template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >(
                         *(FunctionType*) this->function, vertex, time );
         }

      default:
         return 0.0;
   }
}

template< int FunctionDimension, typename Real, typename Device >
template< int XDiffOrder, int YDiffOrder, int ZDiffOrder >
__cuda_callable__
Real
TestFunction< FunctionDimension, Real, Device >::getTimeDerivative( const PointType& vertex, const Real& time ) const
{
   using namespace noa::TNL::Functions::Analytic;
   Real scale( 0.0 );
   switch( timeDependence ) {
      case none:
         break;
      case linear:
         scale = -this->timeScale;
         break;
      case quadratic:
         scale = -2.0 * this->timeScale * this->timeScale * time;
         break;
      case cosine:
         scale = -this->timeScale * ::sin( this->timeScale * time );
         break;
   }
   switch( functionType ) {
      case constant:
         {
            using FunctionType = Constant< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         }
      case paraboloid:
         {
            using FunctionType = Paraboloid< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         }
      case expBump:
         {
            using FunctionType = ExpBump< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         }
      case sinBumps:
         {
            using FunctionType = SinBumps< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         }
      case sinWave:
         {
            using FunctionType = SinWave< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
            break;
         }
      case cylinder:
         {
            using FunctionType = Cylinder< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
            break;
         }
      case flowerpot:
         {
            using FunctionType = Flowerpot< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
            break;
         }
      case twins:
         {
            using FunctionType = Twins< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
            break;
         }
      case pseudoSquare:
         {
            using FunctionType = PseudoSquare< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
            break;
         }
      case blob:
         {
            using FunctionType = Blob< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
            break;
         }

      case paraboloidSDF:
         {
            using FunctionType = ParaboloidSDF< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         }
      case sinBumpsSDF:
         {
            using FunctionType = SinBumpsSDF< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         }
      case sinWaveSDF:
         {
            using FunctionType = SinWaveSDF< Dimension, Real >;
            return scale
                 * ( (FunctionType*) function )
                      ->template getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         }
      default:
         return 0.0;
   }
}

template< int FunctionDimension, typename Real, typename Device >
template< typename FunctionType >
void
TestFunction< FunctionDimension, Real, Device >::deleteFunction()
{
   if( std::is_same< Device, Devices::Host >::value ) {
      if( function )
         delete(FunctionType*) function;
   }
   if( std::is_same< Device, Devices::Cuda >::value ) {
      if( function )
         Allocators::Cuda< FunctionType >{}.deallocate( (FunctionType*) function, 1 );
   }
}

template< int FunctionDimension, typename Real, typename Device >
template< typename OperatorType >
void
TestFunction< FunctionDimension, Real, Device >::deleteOperator()
{
   if( std::is_same< Device, Devices::Host >::value ) {
      if( operator_ )
         delete(OperatorType*) operator_;
   }
   if( std::is_same< Device, Devices::Cuda >::value ) {
      if( operator_ )
         Allocators::Cuda< OperatorType >{}.deallocate( (OperatorType*) operator_, 1 );
   }
}

template< int FunctionDimension, typename Real, typename Device >
void
TestFunction< FunctionDimension, Real, Device >::deleteFunctions()
{
   using namespace noa::TNL::Functions::Analytic;
   using namespace noa::TNL::Operators::Analytic;
   switch( functionType ) {
      case constant:
         {
            using FunctionType = Constant< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case paraboloid:
         {
            using FunctionType = Paraboloid< Dimension, Real >;
            deleteFunction< FunctionType >();
            if( operatorType == identity )
               deleteOperator< Identity< Dimension, Real > >();
            if( operatorType == heaviside )
               deleteOperator< Heaviside< Dimension, Real > >();
            break;
         }
      case expBump:
         {
            using FunctionType = ExpBump< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case sinBumps:
         {
            using FunctionType = SinBumps< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case sinWave:
         {
            using FunctionType = SinWave< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case cylinder:
         {
            using FunctionType = Cylinder< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case flowerpot:
         {
            using FunctionType = Flowerpot< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case twins:
         {
            using FunctionType = Twins< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case pseudoSquare:
         {
            using FunctionType = PseudoSquare< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case blob:
         {
            using FunctionType = Blob< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case vectorNorm:
         {
            using FunctionType = VectorNorm< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }

      case paraboloidSDF:
         {
            using FunctionType = ParaboloidSDF< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case sinBumpsSDF:
         {
            using FunctionType = SinBumpsSDF< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
      case sinWaveSDF:
         {
            using FunctionType = SinWaveSDF< Dimension, Real >;
            deleteFunction< FunctionType >();
            deleteOperator< Identity< Dimension, Real > >();
            break;
         }
   }
}

template< int FunctionDimension, typename Real, typename Device >
template< typename FunctionType >
void
TestFunction< FunctionDimension, Real, Device >::copyFunction( const void* function )
{
   if( std::is_same< Device, Devices::Host >::value ) {
      FunctionType* f = new FunctionType;
      *f = *(FunctionType*) function;
   }
   if( std::is_same< Device, Devices::Cuda >::value ) {
      throw Exceptions::NotImplementedError( "Assignment operator is not implemented for CUDA." );
   }
}

template< int FunctionDimension, typename Real, typename Device >
template< typename FunctionType >
std::ostream&
TestFunction< FunctionDimension, Real, Device >::printFunction( std::ostream& str ) const
{
   if( std::is_same< Device, Devices::Host >::value ) {
      FunctionType* f = (FunctionType*) this->function;
      str << *f;
      return str;
   }
   if( std::is_same< Device, Devices::Cuda >::value ) {
      FunctionType f;
      Algorithms::MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::copy( &f, (FunctionType*) this->function, 1 );
      str << f;
      return str;
   }
}

template< int FunctionDimension, typename Real, typename Device >
std::ostream&
TestFunction< FunctionDimension, Real, Device >::print( std::ostream& str ) const
{
   using namespace noa::TNL::Functions::Analytic;
   str << " timeDependence = " << this->timeDependence;
   str << " functionType = " << this->functionType;
   str << " function = " << this->function << "; ";
   switch( functionType ) {
      case constant:
         return printFunction< Constant< Dimension, Real > >( str );
      case expBump:
         return printFunction< ExpBump< Dimension, Real > >( str );
      case sinBumps:
         return printFunction< SinBumps< Dimension, Real > >( str );
      case sinWave:
         return printFunction< SinWave< Dimension, Real > >( str );
      case cylinder:
         return printFunction< Cylinder< Dimension, Real > >( str );
      case flowerpot:
         return printFunction< Flowerpot< Dimension, Real > >( str );
      case twins:
         return printFunction< Twins< Dimension, Real > >( str );
      case pseudoSquare:
         return printFunction< PseudoSquare< Dimension, Real > >( str );
      case blob:
         return printFunction< Blob< Dimension, Real > >( str );
   }
   return str;
}

template< int FunctionDimension, typename Real, typename Device >
TestFunction< FunctionDimension, Real, Device >::~TestFunction()
{
   deleteFunctions();
}

}  // namespace Functions
}  // namespace noa::TNL
