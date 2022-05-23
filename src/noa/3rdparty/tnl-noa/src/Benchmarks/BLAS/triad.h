// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>

#include <TNL/Containers/Array.h>
#include <TNL/Allocators/CudaHost.h>
#include <TNL/Allocators/CudaManaged.h>

namespace TNL {
namespace Benchmarks {

template< typename Real = double,
          typename Index = int >
void
benchmarkTriad( Benchmark<> & benchmark,
                const long & size )
{
   using HostAllocator = Allocators::Host< Real >;
   using CudaAllocator = Allocators::Cuda< Real >;
   using CudaHostAllocator = Allocators::CudaHost< Real >;
   using CudaManagedAllocator = Allocators::CudaManaged< Real >;

   double datasetSize = (double) size * sizeof(Real) / oneGB;
   benchmark.setOperation( "triad", 3 * datasetSize );

   const Real scalar = 3.1415926535;

   // pageable
   {
      using HostArray = Containers::Array< Real, Devices::Host, Index, HostAllocator >;
      using CudaArray = Containers::Array< Real, Devices::Cuda, Index, CudaAllocator >;

      HostArray a_h;
      HostArray b_h;
      HostArray c_h;
      CudaArray a_d;
      CudaArray b_d;
      CudaArray c_d;
      a_h.setSize( size );
      b_h.setSize( size );
      c_h.setSize( size );
      a_d.setSize( size );
      b_d.setSize( size );
      c_d.setSize( size );

      auto reset = [&]()
      {
         a_h.setValue( 1.0 );
         b_h.setValue( 1.0 );
         c_h.setValue( 1.0 );
         a_d.setValue( 0.0 );
         b_d.setValue( 0.0 );
         c_d.setValue( 0.0 );
      };

      auto triad = [&]()
      {
         b_d = b_h;
         c_d = c_h;

         auto a_v = a_d.getView();
         const auto b_v = b_d.getConstView();
         const auto c_v = c_d.getConstView();
         auto kernel = [=] __cuda_callable__ ( Index i ) mutable
         {
            a_v[i] = b_v[i] + scalar * c_v[i];
         };
         Algorithms::ParallelFor< Devices::Cuda >::exec( (long) 0, size, kernel );

         a_h = a_d;
      };

      benchmark.time< Devices::Cuda >( reset, "pageable", triad );
   }

   // pinned
   {
      using HostArray = Containers::Array< Real, Devices::Host, Index, CudaHostAllocator >;
      using CudaArray = Containers::Array< Real, Devices::Cuda, Index, CudaAllocator >;

      HostArray a_h;
      HostArray b_h;
      HostArray c_h;
      CudaArray a_d;
      CudaArray b_d;
      CudaArray c_d;
      a_h.setSize( size );
      b_h.setSize( size );
      c_h.setSize( size );
      a_d.setSize( size );
      b_d.setSize( size );
      c_d.setSize( size );

      auto reset = [&]()
      {
         a_h.setValue( 1.0 );
         b_h.setValue( 1.0 );
         c_h.setValue( 1.0 );
         a_d.setValue( 0.0 );
         b_d.setValue( 0.0 );
         c_d.setValue( 0.0 );
      };

      auto triad = [&]()
      {
         b_d = b_h;
         c_d = c_h;

         auto a_v = a_d.getView();
         const auto b_v = b_d.getConstView();
         const auto c_v = c_d.getConstView();
         auto kernel = [=] __cuda_callable__ ( Index i ) mutable
         {
            a_v[i] = b_v[i] + scalar * c_v[i];
         };
         Algorithms::ParallelFor< Devices::Cuda >::exec( (long) 0, size, kernel );

         a_h = a_d;
      };

      benchmark.time< Devices::Cuda >( reset, "pinned", triad );
   }

   // zero-copy
   {
      using HostArray = Containers::Array< Real, Devices::Host, Index, CudaHostAllocator >;

      HostArray a_h;
      HostArray b_h;
      HostArray c_h;
      a_h.setSize( size );
      b_h.setSize( size );
      c_h.setSize( size );

      auto reset = [&]()
      {
         a_h.setValue( 1.0 );
         b_h.setValue( 1.0 );
         c_h.setValue( 1.0 );
      };

      auto a_v = a_h.getView();
      const auto b_v = b_h.getConstView();
      const auto c_v = c_h.getConstView();
      auto kernel = [=] __cuda_callable__ ( Index i ) mutable
      {
         a_v[i] = b_v[i] + scalar * c_v[i];
      };
      auto triad = [&]()
      {
         Algorithms::ParallelFor< Devices::Cuda >::exec( (long) 0, size, kernel );
      };

      benchmark.time< Devices::Cuda >( reset, "zero-copy", triad );
   }

   // unified memory
   {
      using Array = Containers::Array< Real, Devices::Host, Index, CudaManagedAllocator >;

      Array a;
      Array b;
      Array c;
      a.setSize( size );
      b.setSize( size );
      c.setSize( size );

      auto reset = [&]()
      {
         a.setValue( 1.0 );
         b.setValue( 1.0 );
         c.setValue( 1.0 );
      };

      auto a_v = a.getView();
      const auto b_v = b.getConstView();
      const auto c_v = c.getConstView();
      auto kernel = [=] __cuda_callable__ ( Index i ) mutable
      {
         a_v[i] = b_v[i] + scalar * c_v[i];
      };
      auto triad = [&]()
      {
         Algorithms::ParallelFor< Devices::Cuda >::exec( (long) 0, size, kernel );
      };

      benchmark.time< Devices::Cuda >( reset, "unified memory", triad );
   }

   // TODO: unified memory with AccessedBy hint and/or prefetching - see https://github.com/cwpearson/triad-gpu-bandwidth
}

} // namespace Benchmarks
} // namespace TNL
