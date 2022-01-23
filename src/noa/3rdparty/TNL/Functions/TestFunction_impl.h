// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Allocators/Cuda.h>
#include <noa/3rdparty/TNL/Algorithms/MultiDeviceMemoryOperations.h>

#include <noa/3rdparty/TNL/Functions/Analytic/Constant.h>
#include <noa/3rdparty/TNL/Functions/Analytic/ExpBump.h>
#include <noa/3rdparty/TNL/Functions/Analytic/SinBumps.h>
#include <noa/3rdparty/TNL/Functions/Analytic/SinWave.h>
#include <noa/3rdparty/TNL/Functions/Analytic/Constant.h>
#include <noa/3rdparty/TNL/Functions/Analytic/ExpBump.h>
#include <noa/3rdparty/TNL/Functions/Analytic/SinBumps.h>
#include <noa/3rdparty/TNL/Functions/Analytic/SinWave.h>
#include <noa/3rdparty/TNL/Functions/Analytic/Cylinder.h>
#include <noa/3rdparty/TNL/Functions/Analytic/Flowerpot.h>
#include <noa/3rdparty/TNL/Functions/Analytic/Twins.h>
#include <noa/3rdparty/TNL/Functions/Analytic/Blob.h>
#include <noa/3rdparty/TNL/Functions/Analytic/PseudoSquare.h>
#include <noa/3rdparty/TNL/Functions/Analytic/Paraboloid.h>
#include <noa/3rdparty/TNL/Functions/Analytic/VectorNorm.h>
/****
 * The signed distance test functions
 */
#include <noa/3rdparty/TNL/Functions/Analytic/SinBumpsSDF.h>
#include <noa/3rdparty/TNL/Functions/Analytic/SinWaveSDF.h>
#include <noa/3rdparty/TNL/Functions/Analytic/ParaboloidSDF.h>

#include <noa/3rdparty/TNL/Operators/Analytic/Identity.h>
#include <noa/3rdparty/TNL/Operators/Analytic/Heaviside.h>
#include <noa/3rdparty/TNL/Operators/Analytic/SmoothHeaviside.h>

#include <noa/3rdparty/TNL/Exceptions/NotImplementedError.h>

#include "TestFunction.h"

namespace noa::TNL {
namespace Functions {

template< int FunctionDimension,
          typename Real,
          typename Device >
TestFunction< FunctionDimension, Real, Device >::
TestFunction()
: function( 0 ),
  operator_( 0 ),
  timeDependence( none ),
  timeScale( 1.0 )
{
}

template< int FunctionDimension,
          typename Real,
          typename Device >
void
TestFunction< FunctionDimension, Real, Device >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
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

   config.addEntry     < double >( prefix + "constant", "Value of the constant function.", 0.0 );
   config.addEntry     < double >( prefix + "wave-length", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "wave-length-x", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "wave-length-y", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "wave-length-z", "Wave length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "phase", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "phase-x", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "phase-y", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "phase-z", "Phase of the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "amplitude", "Amplitude length of the sine based test functions.", 1.0 );
   config.addEntry     < double >( prefix + "waves-number", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "waves-number-x", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "waves-number-y", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "waves-number-z", "Cut-off for the sine based test functions.", 0.0 );
   config.addEntry     < double >( prefix + "sigma", "Sigma for the exp based test functions.", 1.0 );
	config.addEntry     < double >( prefix + "radius", "Radius for paraboloids.", 1.0 );
   config.addEntry     < double >( prefix + "coefficient", "Coefficient for paraboloids.", 1.0 );
   config.addEntry     < double >( prefix + "x-center", "x-center for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "y-center", "y-center for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "z-center", "z-center for paraboloids.", 0.0 );
   config.addEntry     < double >( prefix + "diameter", "Diameter for the cylinder, flowerpot test functions.", 1.0 );
   config.addEntry     < double >( prefix + "height", "Height of zero-level-set function for the blob, pseudosquare test functions.", 1.0 );
   Analytic::VectorNorm< 3, double >::configSetup( config, "vector-norm-" );
   noa::TNL::Operators::Analytic::Heaviside< 3, double >::configSetup( config, "heaviside-" );
   noa::TNL::Operators::Analytic::SmoothHeaviside< 3, double >::configSetup( config, "smooth-heaviside-" );
   config.addEntry     < String >( prefix + "time-dependence", "Time dependence of the test function.", "none" );
      config.addEntryEnum( "none" );
      config.addEntryEnum( "linear" );
      config.addEntryEnum( "quadratic" );
      config.addEntryEnum( "cosine" );
   config.addEntry     < double >( prefix + "time-scale", "Time scaling for the time dependency of the test function.", 1.0 );

}

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< typename FunctionType >
bool
TestFunction< FunctionDimension, Real, Device >::
setupFunction( const Config::ParameterContainer& parameters,
               const String& prefix )
{
   FunctionType* auxFunction = new FunctionType;
   if( ! auxFunction->setup( parameters, prefix ) )
   {
      delete auxFunction;
      return false;
   }

   if( std::is_same< Device, Devices::Host >::value )
   {
      this->function = auxFunction;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      this->function = Allocators::Cuda< FunctionType >{}.allocate( 1 );
      Algorithms::MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::copy( (FunctionType*) this->function, (FunctionType*) auxFunction, 1 );
      delete auxFunction;
      TNL_CHECK_CUDA_DEVICE;
   }
   return true;
}

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< typename OperatorType >
bool
TestFunction< FunctionDimension, Real, Device >::
setupOperator( const Config::ParameterContainer& parameters,
               const String& prefix )
{
   OperatorType* auxOperator = new OperatorType;
   if( ! auxOperator->setup( parameters, prefix ) )
   {
      delete auxOperator;
      return false;
   }

   if( std::is_same< Device, Devices::Host >::value )
   {
      this->operator_ = auxOperator;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      this->operator_ = Allocators::Cuda< OperatorType >{}.allocate( 1 );
      Algorithms::MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::copy( (OperatorType*) this->operator_, (OperatorType*) auxOperator, 1 );
      delete auxOperator;
      TNL_CHECK_CUDA_DEVICE;
   }
   return true;
}

template< int FunctionDimension,
          typename Real,
          typename Device >
bool
TestFunction< FunctionDimension, Real, Device >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   using namespace noa::TNL::Functions::Analytic;
   using namespace noa::TNL::Operators::Analytic;
   std::cout << "Test function setup ... " << std::endl;
   const String& timeDependence =
            parameters.getParameter< String >(
                     prefix +
                     "time-dependence" );
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
   if( testFunction == "constant" )
   {
      typedef Constant< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = constant;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "paraboloid" )
   {
      typedef Paraboloid< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = paraboloid;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }   
   if( testFunction == "exp-bump" )
   {
      typedef ExpBump< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = expBump;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-bumps" )
   {
      typedef SinBumps< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = sinBumps;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "sin-wave" )
   {
      typedef SinWave< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = sinWave;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "cylinder" )
   {
      typedef Cylinder< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = cylinder;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "flowerpot" )
   {
      typedef Flowerpot< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = flowerpot;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "twins" )
   {
      typedef Twins< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = twins;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "pseudoSquare" )
   {
      typedef PseudoSquare< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = pseudoSquare;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "blob" )
   {
      typedef Blob< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = blob;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "paraboloid-sdf" )
   {
      typedef ParaboloidSDF< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = paraboloidSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }   
   if( testFunction == "sin-bumps-sdf" )
   {
      typedef SinBumpsSDF< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = sinBumpsSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }   
   if( testFunction == "sin-wave-sdf" )
   {
      typedef SinWaveSDF< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = sinWaveSDF;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "vector-norm" )
   {
      typedef VectorNorm< Dimension, Real > FunctionType;
      typedef Identity< Dimension, Real > OperatorType;
      functionType = vectorNorm;
      operatorType = identity;
      return ( setupFunction< FunctionType >( parameters, prefix ) && 
               setupOperator< OperatorType >( parameters, prefix ) );
   }
   if( testFunction == "heaviside-of-vector-norm" )
   {
      typedef VectorNorm< Dimension, Real > FunctionType;
      typedef Heaviside< Dimension, Real > OperatorType;
      functionType = vectorNorm;
      operatorType = heaviside;
      return ( setupFunction< FunctionType >( parameters, prefix + "vector-norm-" ) && 
               setupOperator< OperatorType >( parameters, prefix + "heaviside-" ) );
   }
   if( testFunction == "smooth-heaviside-of-vector-norm" )
   {
      typedef VectorNorm< Dimension, Real > FunctionType;
      typedef SmoothHeaviside< Dimension, Real > OperatorType;
      functionType = vectorNorm;
      operatorType = smoothHeaviside;
      return ( setupFunction< FunctionType >( parameters, prefix + "vector-norm-" ) && 
               setupOperator< OperatorType >( parameters, prefix + "smooth-heaviside-" ) );
   }
   std::cerr << "Unknown function " << testFunction << std::endl;
   return false;
}

template< int FunctionDimension,
          typename Real,
          typename Device >
const TestFunction< FunctionDimension, Real, Device >&
TestFunction< FunctionDimension, Real, Device >::
operator = ( const TestFunction& function )
{
   /*****
    * TODO: if the function is on the device we cannot do the following
    */
   abort();
   using namespace noa::TNL::Functions::Analytic;
   this->functionType   = function.functionType;
   this->timeDependence = function.timeDependence;
   this->timeScale      = function.timeScale;

   this->deleteFunctions();

   switch( this->functionType )
   {
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

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
TestFunction< FunctionDimension, Real, Device >::
getPartialDerivative( const PointType& vertex,
          const Real& time ) const
{
   TNL_ASSERT_TRUE( this->function, "The test function was not set properly." );
   using namespace noa::TNL::Functions::Analytic;
   using namespace noa::TNL::Operators::Analytic;
   Real scale( 1.0 );
   switch( this->timeDependence )
   {
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
   switch( functionType )
   {
      case constant:
      {
         typedef Constant< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;

         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case paraboloid:
      {
         typedef Paraboloid< Dimension, Real > FunctionType;
         if( operatorType == identity )
         {
            typedef Identity< Dimension, Real > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
         if( operatorType == heaviside )
         {
            typedef Heaviside< Dimension, Real > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
         if( operatorType == smoothHeaviside )
         {
            typedef SmoothHeaviside< Dimension, Real > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
      }
      case expBump:
      {
         typedef ExpBump< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case sinBumps:
      {
         typedef SinBumps< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case sinWave:
      {
         typedef SinWave< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case cylinder:
      {
         typedef Cylinder< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case flowerpot:
      {
         typedef Flowerpot< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case twins:
      {
         typedef Twins< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case pseudoSquare:
      {
         typedef PseudoSquare< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case blob:
      {
         typedef Blob< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case vectorNorm:
      {
         typedef VectorNorm< Dimension, Real > FunctionType;
         if( operatorType == identity )
         {
            typedef Identity< Dimension, Real > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
         if( operatorType == heaviside )
         {
            typedef Heaviside< Dimension, Real > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
         if( operatorType == smoothHeaviside )
         {
            typedef SmoothHeaviside< Dimension, Real > OperatorType;

            return scale * ( ( OperatorType* ) this->operator_ )->
                      template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
         }
      }      
      case sinBumpsSDF:
      {
         typedef SinBumpsSDF< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
                  
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case sinWaveSDF:
      {
         typedef SinWaveSDF< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      case paraboloidSDF:
      {
         typedef ParaboloidSDF< Dimension, Real > FunctionType;
         typedef Identity< Dimension, Real > OperatorType;
         
         return scale * ( ( OperatorType* ) this->operator_ )->
                   template getPartialDerivative< FunctionType, XDiffOrder, YDiffOrder, ZDiffOrder >( * ( FunctionType*) this->function, vertex, time );
      }
      
      default:
         return 0.0;
   }
}

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
TestFunction< FunctionDimension, Real, Device >::
getTimeDerivative( const PointType& vertex,
                   const Real& time ) const
{
   using namespace noa::TNL::Functions::Analytic;
   Real scale( 0.0 );
   switch( timeDependence )
   {
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
   switch( functionType )
   {
      case constant:
      {
         typedef Constant< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      case paraboloid:
      {
         typedef Paraboloid< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      case expBump:
      {
         typedef ExpBump< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      case sinBumps:
      {
         typedef SinBumps< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      case sinWave:
      {
         typedef SinWave< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      }
      case cylinder:
      {
         typedef Cylinder< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      }
      case flowerpot:
      {
         typedef Flowerpot< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      }
      case twins:
      {
         typedef Twins< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      }
      case pseudoSquare:
      {
         typedef PseudoSquare< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      }
      case blob:
      {
         typedef Blob< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
         break;
      }


      case paraboloidSDF:
      {
         typedef ParaboloidSDF< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      case sinBumpsSDF:
      {
         typedef SinBumpsSDF< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      case sinWaveSDF:
      {
         typedef SinWaveSDF< Dimension, Real > FunctionType;
         return scale * ( ( FunctionType* ) function )->template
                  getPartialDerivative< XDiffOrder, YDiffOrder, ZDiffOrder >( vertex, time );
      }
      default:
         return 0.0;
   }
}

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< typename FunctionType >
void
TestFunction< FunctionDimension, Real, Device >::
deleteFunction()
{
   if( std::is_same< Device, Devices::Host >::value )
   {
      if( function )
         delete ( FunctionType * ) function;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      if( function )
         Allocators::Cuda< FunctionType >{}.deallocate( (FunctionType*) function, 1 );
   }
}

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< typename OperatorType >
void
TestFunction< FunctionDimension, Real, Device >::
deleteOperator()
{
   if( std::is_same< Device, Devices::Host >::value )
   {
      if( operator_ )
         delete ( OperatorType * ) operator_;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      if( operator_ )
         Allocators::Cuda< OperatorType >{}.deallocate( (OperatorType*) operator_, 1 );
   }
}

template< int FunctionDimension,
          typename Real,
          typename Device >
void
TestFunction< FunctionDimension, Real, Device >::
deleteFunctions()
{
   using namespace noa::TNL::Functions::Analytic;
   using namespace noa::TNL::Operators::Analytic;
   switch( functionType )
   {
      case constant:
      {
         typedef Constant< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case paraboloid:
      {
         typedef Paraboloid< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         if( operatorType == identity )
            deleteOperator< Identity< Dimension, Real > >();
         if( operatorType == heaviside )
            deleteOperator< Heaviside< Dimension, Real > >();
         break;
      }
      case expBump:
      {
         typedef ExpBump< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case sinBumps:
      {
         typedef SinBumps< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case sinWave:
      {
         typedef SinWave< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case cylinder:
      {
         typedef Cylinder< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case flowerpot:
      {
         typedef Flowerpot< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case twins:
      {
         typedef Twins< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case pseudoSquare:
      {
         typedef PseudoSquare< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case blob:
      {
         typedef Blob< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case vectorNorm:
      {
         typedef VectorNorm< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      
      
      case paraboloidSDF:
      {
         typedef ParaboloidSDF< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case sinBumpsSDF:
      {
         typedef SinBumpsSDF< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
      case sinWaveSDF:
      {
         typedef SinWaveSDF< Dimension, Real> FunctionType;
         deleteFunction< FunctionType >();
         deleteOperator< Identity< Dimension, Real > >();
         break;
      }
   }
}

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< typename FunctionType >
void
TestFunction< FunctionDimension, Real, Device >::
copyFunction( const void* function )
{
   if( std::is_same< Device, Devices::Host >::value )
   {
      FunctionType* f = new FunctionType;
      *f = * ( FunctionType* )function;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      throw Exceptions::NotImplementedError( "Assignment operator is not implemented for CUDA." );
   }
}

template< int FunctionDimension,
          typename Real,
          typename Device >
   template< typename FunctionType >
std::ostream&
TestFunction< FunctionDimension, Real, Device >::
printFunction( std::ostream& str ) const
{
   if( std::is_same< Device, Devices::Host >::value )
   {
      FunctionType* f = ( FunctionType* ) this->function;
      str << *f;
      return str;
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
      FunctionType f;
      Algorithms::MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::copy( &f, (FunctionType*) this->function, 1 );
      str << f;
      return str;
   }
}

template< int FunctionDimension,
          typename Real,
          typename Device >
std::ostream&
TestFunction< FunctionDimension, Real, Device >::
print( std::ostream& str ) const
{
   using namespace noa::TNL::Functions::Analytic;
   str << " timeDependence = " << this->timeDependence;
   str << " functionType = " << this->functionType;
   str << " function = " << this->function << "; ";
   switch( functionType )
   {
      case constant:
         return printFunction< Constant< Dimension, Real> >( str );
      case expBump:
         return printFunction< ExpBump< Dimension, Real> >( str );
      case sinBumps:
         return printFunction< SinBumps< Dimension, Real> >( str );
      case sinWave:
         return printFunction< SinWave< Dimension, Real> >( str );
      case cylinder:
         return printFunction< Cylinder< Dimension, Real> >( str );
      case flowerpot:
         return printFunction< Flowerpot< Dimension, Real> >( str );
      case twins:
         return printFunction< Twins< Dimension, Real> >( str );
      case pseudoSquare:
         return printFunction< PseudoSquare< Dimension, Real> >( str );
      case blob:
         return printFunction< Blob< Dimension, Real> >( str );
   }
   return str;
}

template< int FunctionDimension,
          typename Real,
          typename Device >
TestFunction< FunctionDimension, Real, Device >::
~TestFunction()
{
   deleteFunctions();
}

} // namespace Functions
} // namespace noa::TNL
