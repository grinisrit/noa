// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/Executors.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/StreamPool.h>

namespace noa::TNL {
namespace Containers {
namespace detail {

template< typename Permutation, typename LevelTag = IndexTag< 0 > >
struct SequentialBoundaryExecutor_inner
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func, typename... Indices >
   __cuda_callable__
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               std::size_t level,
               Func f,
               Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      SequentialBoundaryExecutor_inner< Permutation, IndexTag< LevelTag::value + 1 > > exec;
      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      if( level == LevelTag::value ) {
         for( auto i = begin; i < skipBegin; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
         for( auto i = skipEnd; i < end; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
      }
      else if( level > LevelTag::value ) {
         for( auto i = skipBegin; i < skipEnd; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
      }
      else {
         for( auto i = begin; i < end; i++ )
            exec( begins, skipBegins, skipEnds, ends, level, f, std::forward< Indices >( indices )..., i );
      }
   }
};

template< typename Permutation >
struct SequentialBoundaryExecutor_inner< Permutation, IndexTag< Permutation::size() - 1 > >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func, typename... Indices >
   __cuda_callable__
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               std::size_t level,
               Func f,
               Indices&&... indices )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );
      static_assert( sizeof...( indices ) == Begins::getDimension() - 1,
                     "invalid number of indices in the final step of the SequentialBoundaryExecutor" );

      using LevelTag = IndexTag< Permutation::size() - 1 >;

      const auto begin = begins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< LevelTag::value >( Permutation{} ) >();
      const auto end = ends.template getSize< get< LevelTag::value >( Permutation{} ) >();
      if( level == LevelTag::value ) {
         for( auto i = begin; i < skipBegin; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
         for( auto i = skipEnd; i < end; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
      }
      else if( level > LevelTag::value ) {
         for( auto i = skipBegin; i < skipEnd; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
      }
      else {
         for( auto i = begin; i < end; i++ )
            call_with_unpermuted_arguments< Permutation >( f, std::forward< Indices >( indices )..., i );
      }
   }
};

template< typename Permutation, std::size_t dim = Permutation::size() >
struct SequentialBoundaryExecutor
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   __cuda_callable__
   void
   operator()( const Begins& begins, const SkipBegins& skipBegins, const SkipEnds& skipEnds, const Ends& ends, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      SequentialBoundaryExecutor_inner< Permutation > exec;
      for( std::size_t level = 0; level < Permutation::size(); level++ )
         exec( begins, skipBegins, skipEnds, ends, level, f );
   }
};

template< typename Permutation >
struct SequentialBoundaryExecutor< Permutation, 0 >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   __cuda_callable__
   void
   operator()( const Begins& begins, const SkipBegins& skipBegins, const SkipEnds& skipEnds, const Ends& ends, Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      const auto begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto end = ends.template getSize< get< 0 >( Permutation{} ) >();
      for( auto i = begin; i < skipBegin; i++ )
         f( i );
      for( auto i = skipEnd; i < end; i++ )
         f( i );
   }
};

template< typename Permutation, typename Device, typename DimTag = IndexTag< Permutation::size() > >
struct ParallelBoundaryExecutor
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins, const SkipBegins& skipBegins, const SkipEnds& skipEnds, const Ends& ends, Func f )
   {
      static_assert( Permutation::size() <= 3, "ParallelBoundaryExecutor is implemented only for 1D, 2D, and 3D." );
   }
};

template< typename Permutation, typename Device >
struct ParallelBoundaryExecutor< Permutation, Device, IndexTag< 3 > >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      // nvcc does not like nested __cuda_callable__ and normal lambdas...
      Functor_call_with_unpermuted_arguments< Permutation, Device > kernel;

      const auto begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const auto begin2 = begins.template getSize< get< 2 >( Permutation{} ) >();
      const auto skipBegin0 = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin1 = skipBegins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipBegin2 = skipBegins.template getSize< get< 2 >( Permutation{} ) >();
      const auto skipEnd0 = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd1 = skipEnds.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipEnd2 = skipEnds.template getSize< get< 2 >( Permutation{} ) >();
      const auto end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const auto end1 = ends.template getSize< get< 1 >( Permutation{} ) >();
      const auto end2 = ends.template getSize< get< 2 >( Permutation{} ) >();

      Algorithms::ParallelFor3D< Device >::exec(
         begin2, begin1, begin0, skipBegin2, end1, end0, launch_configuration, kernel, f );
      Algorithms::ParallelFor3D< Device >::exec( skipEnd2, begin1, begin0, end2, end1, end0, launch_configuration, kernel, f );
      Algorithms::ParallelFor3D< Device >::exec(
         skipBegin2, begin1, begin0, skipEnd2, skipBegin1, end0, launch_configuration, kernel, f );
      Algorithms::ParallelFor3D< Device >::exec(
         skipBegin2, skipEnd1, begin0, skipEnd2, end1, end0, launch_configuration, kernel, f );
      Algorithms::ParallelFor3D< Device >::exec(
         skipBegin2, skipBegin1, begin0, skipEnd2, skipEnd1, skipBegin0, launch_configuration, kernel, f );
      Algorithms::ParallelFor3D< Device >::exec(
         skipBegin2, skipBegin1, skipEnd0, skipEnd2, skipEnd1, end0, launch_configuration, kernel, f );
   }
};

template< typename Permutation >
struct ParallelBoundaryExecutor< Permutation, Devices::Cuda, IndexTag< 3 > >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               Devices::Cuda::LaunchConfiguration launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      // nvcc does not like nested __cuda_callable__ and normal lambdas...
      Functor_call_with_unpermuted_arguments< Permutation, Devices::Cuda > kernel;

      const auto begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const auto begin2 = begins.template getSize< get< 2 >( Permutation{} ) >();
      const auto skipBegin0 = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin1 = skipBegins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipBegin2 = skipBegins.template getSize< get< 2 >( Permutation{} ) >();
      const auto skipEnd0 = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd1 = skipEnds.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipEnd2 = skipEnds.template getSize< get< 2 >( Permutation{} ) >();
      const auto end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const auto end1 = ends.template getSize< get< 1 >( Permutation{} ) >();
      const auto end2 = ends.template getSize< get< 2 >( Permutation{} ) >();

      // launch each kernel in its own stream to achieve concurrency
      cudaStream_t stream_1 = Cuda::StreamPool::getInstance().getStream( 1 );
      cudaStream_t stream_2 = Cuda::StreamPool::getInstance().getStream( 2 );
      cudaStream_t stream_3 = Cuda::StreamPool::getInstance().getStream( 3 );
      cudaStream_t stream_4 = Cuda::StreamPool::getInstance().getStream( 4 );
      cudaStream_t stream_5 = Cuda::StreamPool::getInstance().getStream( 5 );
      cudaStream_t stream_6 = Cuda::StreamPool::getInstance().getStream( 6 );

      // remember the original mode and set non-blocking for the following
      const bool blockHostUntilFinished = launch_configuration.blockHostUntilFinished;
      launch_configuration.blockHostUntilFinished = false;

      launch_configuration.stream = stream_1;
      Algorithms::ParallelFor3D< Devices::Cuda >::exec(
         begin2, begin1, begin0, skipBegin2, end1, end0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_2;
      Algorithms::ParallelFor3D< Devices::Cuda >::exec(
         skipEnd2, begin1, begin0, end2, end1, end0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_3;
      Algorithms::ParallelFor3D< Devices::Cuda >::exec(
         skipBegin2, begin1, begin0, skipEnd2, skipBegin1, end0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_4;
      Algorithms::ParallelFor3D< Devices::Cuda >::exec(
         skipBegin2, skipEnd1, begin0, skipEnd2, end1, end0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_5;
      Algorithms::ParallelFor3D< Devices::Cuda >::exec(
         skipBegin2, skipBegin1, begin0, skipEnd2, skipEnd1, skipBegin0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_6;
      Algorithms::ParallelFor3D< Devices::Cuda >::exec(
         skipBegin2, skipBegin1, skipEnd0, skipEnd2, skipEnd1, end0, launch_configuration, kernel, f );

      if( blockHostUntilFinished ) {
         // synchronize all streams
         cudaStreamSynchronize( stream_1 );
         cudaStreamSynchronize( stream_2 );
         cudaStreamSynchronize( stream_3 );
         cudaStreamSynchronize( stream_4 );
         cudaStreamSynchronize( stream_5 );
         cudaStreamSynchronize( stream_6 );
         TNL_CHECK_CUDA_DEVICE;
      }
   }
};

template< typename Permutation, typename Device >
struct ParallelBoundaryExecutor< Permutation, Device, IndexTag< 2 > >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      // nvcc does not like nested __cuda_callable__ and normal lambdas...
      Functor_call_with_unpermuted_arguments< Permutation, Device > kernel;

      const auto begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipBegin0 = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin1 = skipBegins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipEnd0 = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd1 = skipEnds.template getSize< get< 1 >( Permutation{} ) >();
      const auto end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const auto end1 = ends.template getSize< get< 1 >( Permutation{} ) >();

      Algorithms::ParallelFor2D< Device >::exec( begin1, begin0, skipBegin1, end0, launch_configuration, kernel, f );
      Algorithms::ParallelFor2D< Device >::exec( skipEnd1, begin0, end1, end0, launch_configuration, kernel, f );
      Algorithms::ParallelFor2D< Device >::exec( skipBegin1, begin0, skipEnd1, skipBegin0, launch_configuration, kernel, f );
      Algorithms::ParallelFor2D< Device >::exec( skipBegin1, skipEnd0, skipEnd1, end0, launch_configuration, kernel, f );
   }
};

template< typename Permutation >
struct ParallelBoundaryExecutor< Permutation, Devices::Cuda, IndexTag< 2 > >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               Devices::Cuda::LaunchConfiguration launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      // nvcc does not like nested __cuda_callable__ and normal lambdas...
      Functor_call_with_unpermuted_arguments< Permutation, Devices::Cuda > kernel;

      const auto begin0 = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto begin1 = begins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipBegin0 = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin1 = skipBegins.template getSize< get< 1 >( Permutation{} ) >();
      const auto skipEnd0 = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd1 = skipEnds.template getSize< get< 1 >( Permutation{} ) >();
      const auto end0 = ends.template getSize< get< 0 >( Permutation{} ) >();
      const auto end1 = ends.template getSize< get< 1 >( Permutation{} ) >();

      // launch each kernel in its own stream to achieve concurrency
      cudaStream_t stream_1 = Cuda::StreamPool::getInstance().getStream( 1 );
      cudaStream_t stream_2 = Cuda::StreamPool::getInstance().getStream( 2 );
      cudaStream_t stream_3 = Cuda::StreamPool::getInstance().getStream( 3 );
      cudaStream_t stream_4 = Cuda::StreamPool::getInstance().getStream( 4 );

      // remember the original mode and set non-blocking for the following
      const bool blockHostUntilFinished = launch_configuration.blockHostUntilFinished;
      launch_configuration.blockHostUntilFinished = false;

      launch_configuration.stream = stream_1;
      Algorithms::ParallelFor2D< Devices::Cuda >::exec( begin1, begin0, skipBegin1, end0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_2;
      Algorithms::ParallelFor2D< Devices::Cuda >::exec( skipEnd1, begin0, end1, end0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_3;
      Algorithms::ParallelFor2D< Devices::Cuda >::exec(
         skipBegin1, begin0, skipEnd1, skipBegin0, launch_configuration, kernel, f );
      launch_configuration.stream = stream_4;
      Algorithms::ParallelFor2D< Devices::Cuda >::exec( skipBegin1, skipEnd0, skipEnd1, end0, launch_configuration, kernel, f );

      if( blockHostUntilFinished ) {
         // synchronize all streams
         cudaStreamSynchronize( stream_1 );
         cudaStreamSynchronize( stream_2 );
         cudaStreamSynchronize( stream_3 );
         cudaStreamSynchronize( stream_4 );
         TNL_CHECK_CUDA_DEVICE;
      }
   }
};

template< typename Permutation, typename Device >
struct ParallelBoundaryExecutor< Permutation, Device, IndexTag< 1 > >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      const auto begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto end = ends.template getSize< get< 0 >( Permutation{} ) >();

      Algorithms::ParallelFor< Device >::exec( begin, skipBegin, launch_configuration, f );
      Algorithms::ParallelFor< Device >::exec( skipEnd, end, launch_configuration, f );
   }
};

template< typename Permutation >
struct ParallelBoundaryExecutor< Permutation, Devices::Cuda, IndexTag< 1 > >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               Devices::Cuda::LaunchConfiguration launch_configuration,
               Func f )
   {
      static_assert( Begins::getDimension() == Ends::getDimension(), "wrong begins or ends" );

      const auto begin = begins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipBegin = skipBegins.template getSize< get< 0 >( Permutation{} ) >();
      const auto skipEnd = skipEnds.template getSize< get< 0 >( Permutation{} ) >();
      const auto end = ends.template getSize< get< 0 >( Permutation{} ) >();

      // launch each kernel in its own stream to achieve concurrency
      cudaStream_t stream_1 = Cuda::StreamPool::getInstance().getStream( 1 );
      cudaStream_t stream_2 = Cuda::StreamPool::getInstance().getStream( 2 );

      // remember the original mode and set non-blocking for the following
      const bool blockHostUntilFinished = launch_configuration.blockHostUntilFinished;
      launch_configuration.blockHostUntilFinished = false;

      launch_configuration.stream = stream_1;
      Algorithms::ParallelFor< Devices::Cuda >::exec( begin, skipBegin, launch_configuration, f );
      launch_configuration.stream = stream_2;
      Algorithms::ParallelFor< Devices::Cuda >::exec( skipEnd, end, launch_configuration, f );

      if( blockHostUntilFinished ) {
         // synchronize all streams
         cudaStreamSynchronize( stream_1 );
         cudaStreamSynchronize( stream_2 );
         TNL_CHECK_CUDA_DEVICE;
      }
   }
};

// Device may be void which stands for StaticNDArray
template< typename Permutation, typename Device >
struct BoundaryExecutorDispatcher
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               const typename Device::LaunchConfiguration& launch_configuration,
               Func f )
   {
      SequentialBoundaryExecutor< Permutation >()( begins, skipBegins, skipEnds, ends, f );
   }
};

template< typename Permutation >
struct BoundaryExecutorDispatcher< Permutation, Devices::Host >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               const Devices::Host::LaunchConfiguration& launch_configuration,
               Func f )
   {
      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 )
         ParallelBoundaryExecutor< Permutation, Devices::Host >()(
            begins, skipBegins, skipEnds, ends, launch_configuration, f );
      else
         SequentialBoundaryExecutor< Permutation >()( begins, skipBegins, skipEnds, ends, f );
   }
};

template< typename Permutation >
struct BoundaryExecutorDispatcher< Permutation, Devices::Cuda >
{
   template< typename Begins, typename SkipBegins, typename SkipEnds, typename Ends, typename Func >
   void
   operator()( const Begins& begins,
               const SkipBegins& skipBegins,
               const SkipEnds& skipEnds,
               const Ends& ends,
               const Devices::Cuda::LaunchConfiguration& launch_configuration,
               Func f )
   {
      ParallelBoundaryExecutor< Permutation, Devices::Cuda >()( begins, skipBegins, skipEnds, ends, launch_configuration, f );
   }
};

}  // namespace detail
}  // namespace Containers
}  // namespace noa::TNL
