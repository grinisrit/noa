// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Containers/ndarray/Executors.h>

namespace noaTNL {
namespace Containers {

namespace __ndarray_impl {

#ifndef __NVCC__
template< typename Output,
          typename Func,
          typename... Input >
void nd_map_view( Output output, Func f, const Input... input )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input::getDimension()...} ),
                  "all arrays must be of the same dimension" );

   // without mutable, the operator() would be const so output would be const as well
   // https://stackoverflow.com/a/2835645/4180822
   auto wrapper = [=] __cuda_callable__ ( auto... indices ) mutable {
      static_assert( sizeof...( indices ) == Output::getDimension(),
                     "wrong number of indices passed to the wrapper lambda function" );
      output( indices... ) = f( input( indices... )... );
   };

   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   using Begins = ConstStaticSizesHolder< typename Output::IndexType, output.getDimension(), 0 >;
   dispatch( Begins{}, output.getSizes(), wrapper );
}

#else

   template< typename Output,
             typename Func >
   struct nvcc_map_helper_0
   {
      Output output;
      Func f;

      nvcc_map_helper_0( Output o, Func f ) : output(o), f(f) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f();
      }
   };

   template< typename Output,
             typename Func,
             typename Input1 >
   struct nvcc_map_helper_1
   {
      Output output;
      Func f;
      Input1 input1;

      nvcc_map_helper_1( Output o, Func f, Input1 i1 ) : output(o), f(f), input1(i1) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f( input1( indices... ) );
      }
   };

   template< typename Output,
             typename Func,
             typename Input1,
             typename Input2 >
   struct nvcc_map_helper_2
   {
      Output output;
      Func f;
      Input1 input1;
      Input2 input2;

      nvcc_map_helper_2( Output o, Func f, Input1 i1, Input2 i2 ) : output(o), f(f), input1(i1), input2(i2) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f( input1( indices... ), input2( indices... ) );
      }
   };

   template< typename Output,
             typename Func,
             typename Input1,
             typename Input2,
             typename Input3 >
   struct nvcc_map_helper_3
   {
      Output output;
      Func f;
      Input1 input1;
      Input2 input2;
      Input3 input3;

      nvcc_map_helper_3( Output o, Func f, Input1 i1, Input2 i2, Input3 i3 ) : output(o), f(f), input1(i1), input2(i2), input3(i3) {}

      template< typename... Ts >
      __cuda_callable__
      void operator()( Ts... indices )
      {
         static_assert( sizeof...( indices ) == Output::getDimension(),
                        "wrong number of indices passed to the wrapper operator() function" );
         output( indices... ) = f( input1( indices... ), input2( indices... ), input3( indices... ) );
      }
   };

template< typename Output,
          typename Func >
void nd_map_view( Output output, Func f )
{
   nvcc_map_helper_0< Output, Func > wrapper( output, f );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   using Begins = ConstStaticSizesHolder< typename Output::IndexType, output.getDimension(), 0 >;
   dispatch( Begins{}, output.getSizes(), wrapper );
}

template< typename Output,
          typename Func,
          typename Input1 >
void nd_map_view( Output output, Func f, const Input1 input1 )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input1::getDimension()} ),
                  "all arrays must be of the same dimension" );

   nvcc_map_helper_1< Output, Func, Input1 > wrapper( output, f, input1 );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   using Begins = ConstStaticSizesHolder< typename Output::IndexType, output.getDimension(), 0 >;
   dispatch( Begins{}, output.getSizes(), wrapper );
}

template< typename Output,
          typename Func,
          typename Input1,
          typename Input2 >
void nd_map_view( Output output, Func f, const Input1 input1, const Input2 input2 )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input1::getDimension(), Input2::getDimension()} ),
                  "all arrays must be of the same dimension" );

   nvcc_map_helper_2< Output, Func, Input1, Input2 > wrapper( output, f, input1, input2 );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   using Begins = ConstStaticSizesHolder< typename Output::IndexType, output.getDimension(), 0 >;
   dispatch( Begins{}, output.getSizes(), wrapper );
}

template< typename Output,
          typename Func,
          typename Input1,
          typename Input2,
          typename Input3 >
void nd_map_view( Output output, Func f, const Input1 input1, const Input2 input2, const Input3 input3 )
{
   static_assert( all_elements_equal_to_value( Output::getDimension(), {Input1::getDimension(), Input2::getDimension(), Input3::getDimension()} ),
                  "all arrays must be of the same dimension" );

   nvcc_map_helper_3< Output, Func, Input1, Input2, Input3 > wrapper( output, f, input1, input2, input3 );
   ExecutorDispatcher< typename Output::PermutationType, typename Output::DeviceType > dispatch;
   using Begins = ConstStaticSizesHolder< typename Output::IndexType, output.getDimension(), 0 >;
   dispatch( Begins{}, output.getSizes(), wrapper );
}

#endif

} // namespace __ndarray_impl


// f must be an N-ary function, where N is the dimension of the output and input arrays:
//      output( i1, ..., iN ) = f( input1( i1, ..., iN ), ... inputM( i1, ..., iN ) )
template< typename Output,
          typename Func,
          typename... Input >
void nd_map( Output& output, Func f, const Input&... input )
{
   __ndarray_impl::nd_map_view( output.getView(), f, input.getConstView()... );
}

template< typename Output,
          typename Input >
void nd_assign( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return v; }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return v; }, input );
#endif
}

// Some mathematical functions, inspired by NumPy:
// https://docs.scipy.org/doc/numpy/reference/ufuncs.html#math-operations

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_add( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 + v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 + v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_subtract( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 - v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 - v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_multiply( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 * v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 * v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_divide( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return v1 / v2; }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return v1 / v2; }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_maximum( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return noaTNL::max( v1, v2 ); }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return noaTNL::max( v1, v2 ); }, input1, input2 );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_minimum( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return noaTNL::min( v1, v2 ); }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return noaTNL::min( v1, v2 ); }, input1, input2 );
#endif
}

template< typename Output,
          typename Input >
void nd_absolute( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return noaTNL::abs( v ); }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return noaTNL::abs( v ); }, input );
#endif
}

template< typename Output,
          typename Input >
void nd_sign( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return noaTNL::sign( v ); }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return noaTNL::sign( v ); }, input );
#endif
}

template< typename Output,
          typename Input1,
          typename Input2 >
void nd_pow( Output& output, const Input1& input1, const Input2& input2 )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v1, auto v2 ){ return noaTNL::pow( v1, v2 ); }, input1, input2 );
#else
   using value_type_1 = typename Input1::ValueType;
   using value_type_2 = typename Input2::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type_1 v1, value_type_2 v2 ){ return noaTNL::pow( v1, v2 ); }, input1, input2 );
#endif
}

template< typename Output,
          typename Input >
void nd_sqrt( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return noaTNL::sqrt( v ); }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return noaTNL::sqrt( v ); }, input );
#endif
}

template< typename Output,
          typename Input >
void nd_square( Output& output, const Input& input )
{
#ifndef __NVCC__
   nd_map( output, [] __cuda_callable__ ( auto v ){ return v*v; }, input );
#else
   using value_type = typename Input::ValueType;
   nd_map( output, [] __cuda_callable__ ( value_type v ){ return v*v; }, input );
#endif
}

} // namespace Containers
} // namespace noaTNL
