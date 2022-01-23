// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Cuda/LaunchHelpers.h>
#include <noa/3rdparty/TNL/Containers/VectorView.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Kernels/CSRScalarKernel.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/detail/LambdaAdapter.h>

namespace noa::TNL {
   namespace Algorithms {
      namespace Segments {

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduce,
          typename Keep,
          bool DispatchScalarCSR = detail::CheckFetchLambda< Index, Fetch >::hasAllParameters() >
struct CSRScalarKernelreduceSegmentsDispatcher;

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper >
struct CSRScalarKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper, true >
{

   template< typename Offsets,
             typename Real >
   static void reduce( const Offsets& offsets,
                       Index first,
                       Index last,
                       Fetch& fetch,
                       const Reduction& reduction,
                       ResultKeeper& keep,
                       const Real& zero )
   {
      auto l = [=] __cuda_callable__ ( const Index segmentIdx ) mutable {
         const Index begin = offsets[ segmentIdx ];
         const Index end = offsets[ segmentIdx + 1 ];
         Real aux( zero );
         Index localIdx( 0 );
         bool compute( true );
         for( Index globalIdx = begin; globalIdx < end && compute; globalIdx++  )
             aux = reduction( aux, fetch( segmentIdx, localIdx++, globalIdx, compute ) );
         keep( segmentIdx, aux );
      };

      if( std::is_same< Device, noa::TNL::Devices::Sequential >::value )
      {
         for( Index segmentIdx = first; segmentIdx < last; segmentIdx ++ )
            l( segmentIdx );
      }
      else if( std::is_same< Device, noa::TNL::Devices::Host >::value )
      {
#ifdef HAVE_OPENMP
        #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( Index segmentIdx = first; segmentIdx < last; segmentIdx ++ )
            l( segmentIdx );
      }
      else
         Algorithms::ParallelFor< Device >::exec( first, last, l );
   }
};

template< typename Index,
          typename Device,
          typename Fetch,
          typename Reduce,
          typename Keep >
struct CSRScalarKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduce, Keep, false >
{
   template< typename OffsetsView,
             typename Real >
   static void reduce( const OffsetsView& offsets,
                       Index first,
                       Index last,
                       Fetch& fetch,
                       const Reduce& reduction,
                       Keep& keep,
                       const Real& zero )
   {
      auto l = [=] __cuda_callable__ ( const Index segmentIdx ) mutable {
         const Index begin = offsets[ segmentIdx ];
         const Index end = offsets[ segmentIdx + 1 ];
         Real aux( zero );
         bool compute( true );
         for( Index globalIdx = begin; globalIdx < end && compute; globalIdx++  )
             aux = reduction( aux, fetch( globalIdx, compute ) );
         keep( segmentIdx, aux );
      };

      if( std::is_same< Device, noa::TNL::Devices::Sequential >::value )
      {
         for( Index segmentIdx = first; segmentIdx < last; segmentIdx ++ )
            l( segmentIdx );
      }
      else if( std::is_same< Device, noa::TNL::Devices::Host >::value )
      {
#ifdef HAVE_OPENMP
        #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( Index segmentIdx = first; segmentIdx < last; segmentIdx ++ )
            l( segmentIdx );
      }
      else
         Algorithms::ParallelFor< Device >::exec( first, last, l );

   }
};


template< typename Index,
          typename Device >
    template< typename Offsets >
void
CSRScalarKernel< Index, Device >::
init( const Offsets& offsets )
{
}

template< typename Index,
          typename Device >
void
CSRScalarKernel< Index, Device >::
reset()
{
}

template< typename Index,
          typename Device >
auto
CSRScalarKernel< Index, Device >::
getView() -> ViewType
{
    return *this;
}

template< typename Index,
          typename Device >
auto
CSRScalarKernel< Index, Device >::
getConstView() const -> ConstViewType
{
    return *this;
};

template< typename Index,
          typename Device >
noa::TNL::String
CSRScalarKernel< Index, Device >::
getKernelType()
{
    return "Scalar";
}

template< typename Index,
          typename Device >
    template< typename OffsetsView,
              typename Fetch,
              typename Reduction,
              typename ResultKeeper,
              typename Real,
              typename... Args >
void
CSRScalarKernel< Index, Device >::
reduceSegments( const OffsetsView& offsets,
                   Index first,
                   Index last,
                   Fetch& fetch,
                   const Reduction& reduction,
                   ResultKeeper& keeper,
                   const Real& zero,
                   Args... args )
{
   CSRScalarKernelreduceSegmentsDispatcher< Index, Device, Fetch, Reduction, ResultKeeper >::reduce(
      offsets, first, last, fetch, reduction, keeper, zero );
   /*
    auto l = [=] __cuda_callable__ ( const IndexType segmentIdx, Args... args ) mutable {
        const IndexType begin = offsets[ segmentIdx ];
        const IndexType end = offsets[ segmentIdx + 1 ];
        Real aux( zero );
        IndexType localIdx( 0 );
        bool compute( true );
        for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
            aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
        keeper( segmentIdx, aux );
    };

     if( std::is_same< DeviceType, noa::TNL::Devices::Host >::value )
    {
#ifdef HAVE_OPENMP
        #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
        for( Index segmentIdx = first; segmentIdx < last; segmentIdx ++ )
            l( segmentIdx, args... );
        {
            const IndexType begin = offsets[ segmentIdx ];
            const IndexType end = offsets[ segmentIdx + 1 ];
            Real aux( zero );
            IndexType localIdx( 0 );
            bool compute( true );
            for( IndexType globalIdx = begin; globalIdx < end && compute; globalIdx++  )
                aux = reduction( aux, detail::FetchLambdaAdapter< IndexType, Fetch >::call( fetch, segmentIdx, localIdx++, globalIdx, compute ) );
            keeper( segmentIdx, aux );
        }
    }
    else
        Algorithms::ParallelFor< Device >::exec( first, last, l, args... );*/
}
      } // namespace Segments
   }  // namespace Algorithms
} // namespace noa::TNL
