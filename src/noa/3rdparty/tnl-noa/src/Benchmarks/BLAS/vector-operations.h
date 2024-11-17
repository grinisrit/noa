// Implemented by: Jakub Klinkovsky

#pragma once

#include <cstdlib>     // srand48
#include <algorithm>   // std::max_element, std::min_element, std::transform, etc.
#include <numeric>     // std::reduce, std::transform_reduce, std::partial_sum, std::inclusive_scan, std::exclusive_scan
#include <execution>   // std::execution policies
#include <functional>  // std::function

#if defined( HAVE_TBB ) && defined( __cpp_lib_parallel_algorithm )
   #define STDEXEC std::execution::par_unseq,
#else
   #define STDEXEC
#endif

#include <TNL/Benchmarks/Benchmarks.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/scan.h>
#include "CommonVectorOperations.h"
#include "VectorOperations.h"

#ifdef HAVE_BLAS
   #include "blasWrappers.h"
#endif

#ifdef __CUDACC__
   #include "cublasWrappers.h"
#endif

#ifdef HAVE_THRUST
   #include <thrust/reduce.h>
   #include <thrust/transform_reduce.h>
   #include <thrust/inner_product.h>
   #include <thrust/scan.h>
   #include <thrust/extrema.h>
   #include <thrust/execution_policy.h>
   #include <thrust/device_ptr.h>
#endif

namespace TNL {
namespace Benchmarks {

template< typename Real = double, typename Index = int >
class VectorOperationsBenchmark
{
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, Index >;
   using SequentialView = Containers::VectorView< Real, Devices::Sequential, Index >;
   using HostView = Containers::VectorView< Real, Devices::Host, Index >;
   using CudaView = Containers::VectorView< Real, Devices::Cuda, Index >;

   Benchmark<>& benchmark;
   long size = 0;
   double datasetSize = 0;

   HostVector hostVector;
   HostVector hostVector2;
   HostVector hostVector3;
   HostVector hostVector4;
   CudaVector deviceVector;
   CudaVector deviceVector2;
   CudaVector deviceVector3;
   CudaVector deviceVector4;

   HostView hostView;
   HostView hostView2;
   HostView hostView3;
   HostView hostView4;
   CudaView deviceView;
   CudaView deviceView2;
   CudaView deviceView3;
   CudaView deviceView4;

   Real resultHost;
   Real resultDevice;

   // reset functions
   std::function< void() > reset1;
   std::function< void() > reset2;
   std::function< void() > reset3;
   std::function< void() > reset4;
   std::function< void() > resetAll;

#ifdef __CUDACC__
   cublasHandle_t cublasHandle;
#endif

public:
   VectorOperationsBenchmark( Benchmark<>& benchmark, const long& size )
   : benchmark( benchmark ), size( size ), datasetSize( (double) size * sizeof( Real ) / oneGB )
   {
      hostVector.setSize( size );
      hostVector2.setSize( size );
      hostVector3.setSize( size );
      hostVector4.setSize( size );
#ifdef __CUDACC__
      deviceVector.setSize( size );
      deviceVector2.setSize( size );
      deviceVector3.setSize( size );
      deviceVector4.setSize( size );
#endif

      hostView.bind( hostVector );
      hostView2.bind( hostVector2 );
      hostView3.bind( hostVector3 );
      hostView4.bind( hostVector4 );
#ifdef __CUDACC__
      deviceView.bind( deviceVector );
      deviceView2.bind( deviceVector2 );
      deviceView3.bind( deviceVector3 );
      deviceView4.bind( deviceVector4 );
#endif

      // reset functions
      // (Make sure to always use some in benchmarks, even if it's not necessary
      // to assure correct result - it helps to clear cache and avoid optimizations
      // of the benchmark loop.)
      reset1 = [ & ]()
      {
         hostVector.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector.setValue( 1.0 );
#endif
         // A relatively harmless call to keep the compiler from realizing we
         // don't actually do any useful work with the result of the reduction.
         srand( static_cast< unsigned int >( resultHost ) );
         resultHost = resultDevice = 0.0;
      };
      reset2 = [ & ]()
      {
         hostVector2.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector2.setValue( 1.0 );
#endif
      };
      reset3 = [ & ]()
      {
         hostVector3.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector3.setValue( 1.0 );
#endif
      };
      reset4 = [ & ]()
      {
         hostVector4.setValue( 1.0 );
#ifdef __CUDACC__
         deviceVector4.setValue( 1.0 );
#endif
      };

      resetAll = [ & ]()
      {
         reset1();
         reset2();
         reset3();
         reset4();
      };

      resetAll();

#ifdef __CUDACC__
      cublasCreate( &cublasHandle );
#endif
   }

   ~VectorOperationsBenchmark()
   {
#ifdef __CUDACC__
      cublasDestroy( cublasHandle );
#endif
   }

   // verifying the results is useful to ensure that we compute the same thing
   // even with the 3rdparty libraries, where the compute functions may be more
   // complicated than just one function call
   void
   verify( const std::string& performer, Real result, Real expected, Real tolerance = 1e-6 )
   {
      if( TNL::abs( result - expected ) > tolerance ) {
         std::cerr << "ERROR: result " << result << " computed by " << performer << " is not equal to the expected value "
                   << expected << std::endl;
      }
   }

   void
   verify( const std::string& performer, const HostVector& result, Real expected, Real tolerance = 1e-6 )
   {
      for( Index i = 0; i < size; i++ ) {
         if( TNL::abs( result[ i ] - expected ) > tolerance ) {
            std::cerr << "ERROR: " << i << "-th value of the result vector " << result[ i ] << " computed by " << performer
                      << " is not equal to the expected value " << expected << std::endl;
         }
      }
   }

   void
   verify( const std::string& performer, const CudaVector& result, Real expected, Real tolerance = 1e-6 )
   {
      hostVector = result;
      verify( performer, hostVector, expected, tolerance );
      reset1();
   }

   void
   max()
   {
      benchmark.setOperation( "max", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMax( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, 1.0 );

      auto computeET = [ & ]()
      {
         using TNL::max;
         resultHost = max( hostView );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, 1.0 );

      auto computeSTL = [ & ]()
      {
         resultHost = *std::max_element( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::max_element", computeSTL );
      verify( "CPU std::max_element", resultHost, 1.0 );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMax( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, 1.0 );

      auto computeCudaET = [ & ]()
      {
         using TNL::max;
         resultDevice = max( deviceView );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, 1.0 );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         resultHost = *thrust::max_element( thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::max_element", computeThrust );
      verify( "CPU thrust::max_element", resultHost, 1.0 );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         resultDevice = *thrust::max_element( thrust::device,
                                              thrust::device_pointer_cast( deviceVector.getData() ),
                                              thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::max_element", computeThrustDevice );
      verify( "GPU thrust::max_element", resultDevice, 1.0 );
   #endif
#endif
   }

   void
   min()
   {
      benchmark.setOperation( "min", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMin( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, 1.0 );

      auto computeET = [ & ]()
      {
         using TNL::min;
         resultHost = min( hostView );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, 1.0 );

      auto computeSTL = [ & ]()
      {
         resultHost = *std::min_element( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::min_element", computeSTL );
      verify( "CPU std::min_element", resultHost, 1.0 );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMin( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, 1.0 );

      auto computeCudaET = [ & ]()
      {
         using TNL::min;
         resultDevice = min( deviceView );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, 1.0 );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         resultHost = *thrust::min_element( thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::min_element", computeThrust );
      verify( "CPU thrust::min_element", resultHost, 1.0 );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         resultDevice = *thrust::min_element( thrust::device,
                                              thrust::device_pointer_cast( deviceVector.getData() ),
                                              thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::min_element", computeThrustDevice );
      verify( "GPU thrust::min_element", resultDevice, 1.0 );
   #endif
#endif
   }

   void
   absMax()
   {
      benchmark.setOperation( "absMax", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMax( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, 1.0 );

      auto computeET = [ & ]()
      {
         using TNL::max;
         resultHost = max( abs( hostView ) );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, 1.0 );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         int index = blasIgamax( size, hostVector.getData(), 1 );
         resultHost = hostVector.getElement( index );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", resultHost, 1.0 );
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = *std::max_element( STDEXEC hostVector.getData(),
                                         hostVector.getData() + hostVector.getSize(),
                                         []( auto a, auto b )
                                         {
                                            return std::abs( a ) < std::abs( b );
                                         } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::max_element", computeSTL );
      verify( "CPU std::max_element", resultHost, 1.0 );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMax( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, 1.0 );

      auto computeCudaET = [ & ]()
      {
         using TNL::max;
         resultDevice = max( abs( deviceView ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, 1.0 );

      auto computeCudaCUBLAS = [ & ]()
      {
         int index = 0;
         cublasIgamax( cublasHandle, size, deviceVector.getData(), 1, &index );
         resultDevice = deviceVector.getElement( index );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", resultDevice, 1.0 );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         resultHost = *thrust::max_element( thrust::host,
                                            hostVector.getData(),
                                            hostVector.getData() + hostVector.getSize(),
                                            []( auto a, auto b )
                                            {
                                               return std::abs( a ) < std::abs( b );
                                            } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::max_element", computeThrust );
      verify( "CPU thrust::max_element", resultHost, 1.0 );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         resultDevice = *thrust::max_element( thrust::device,
                                              thrust::device_pointer_cast( deviceVector.getData() ),
                                              thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
                                              [] __cuda_callable__( Real a, Real b )
                                              {
                                                 return std::abs( a ) < std::abs( b );
                                              } );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::max_element", computeThrustDevice );
      verify( "GPU thrust::max_element", resultDevice, 1.0 );
   #endif
#endif
   }

   void
   absMin()
   {
      benchmark.setOperation( "absMin", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMin( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, 1.0 );

      auto computeET = [ & ]()
      {
         using TNL::min;
         resultHost = min( abs( hostView ) );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, 1.0 );

#if 0
   #ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         int index = blasIgamin( size, hostVector.getData(), 1 );
         resultHost = hostVector.getElement( index );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", resultHost, 1.0 );
   #endif
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = *std::min_element( STDEXEC hostVector.getData(),
                                         hostVector.getData() + hostVector.getSize(),
                                         []( auto a, auto b )
                                         {
                                            return std::abs( a ) < std::abs( b );
                                         } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::min_element", computeSTL );
      verify( "CPU std::min_element", resultHost, 1.0 );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMin( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, 1.0 );

      auto computeCudaET = [ & ]()
      {
         using TNL::min;
         resultDevice = min( abs( deviceView ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, 1.0 );

      auto computeCudaCUBLAS = [ & ]()
      {
         int index = 0;
         cublasIgamin( cublasHandle, size, deviceVector.getData(), 1, &index );
         resultDevice = deviceVector.getElement( index );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", resultDevice, 1.0 );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         resultHost = *thrust::min_element( thrust::host,
                                            hostVector.getData(),
                                            hostVector.getData() + hostVector.getSize(),
                                            []( auto a, auto b )
                                            {
                                               return std::abs( a ) < std::abs( b );
                                            } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::min_element", computeThrust );
      verify( "CPU thrust::min_element", resultHost, 1.0 );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         resultDevice = *thrust::min_element( thrust::device,
                                              thrust::device_pointer_cast( deviceVector.getData() ),
                                              thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
                                              [] __cuda_callable__( Real a, Real b )
                                              {
                                                 return std::abs( a ) < std::abs( b );
                                              } );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::min_element", computeThrustDevice );
      verify( "GPU thrust::min_element", resultDevice, 1.0 );
   #endif
#endif
   }

   void
   sum()
   {
      benchmark.setOperation( "sum", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorSum( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, size );

      auto computeET = [ & ]()
      {
         using TNL::sum;
         resultHost = sum( hostView );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, size );

      auto computeSTL = [ & ]()
      {
         resultHost = std::reduce( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::reduce", computeSTL );
      verify( "CPU std::reduce", resultHost, size );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorSum( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, size );

      auto copmuteCudaET = [ & ]()
      {
         using TNL::sum;
         resultDevice = sum( deviceView );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", copmuteCudaET );
      verify( "GPU ET", resultDevice, size );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         resultHost = thrust::reduce( thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::reduce", computeThrust );
      verify( "CPU thrust::reduce", resultHost, size );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         resultDevice = thrust::reduce( thrust::device,
                                        thrust::device_pointer_cast( deviceVector.getData() ),
                                        thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::reduce", computeThrustDevice );
      verify( "GPU thrust::reduce", resultDevice, size );
   #endif
#endif
   }

   void
   l1norm()
   {
      benchmark.setOperation( "l1 norm", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 1.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, size );

      auto computeET = [ & ]()
      {
         resultHost = lpNorm( hostView, 1.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, size );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         resultHost = blasGasum( size, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", resultHost, size );
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = std::transform_reduce( STDEXEC hostVector.getData(),
                                             hostVector.getData() + hostVector.getSize(),
                                             0,
                                             std::plus<>{},
                                             []( auto v )
                                             {
                                                return std::abs( v );
                                             } );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );
      verify( "CPU std::transform_reduce", resultHost, size );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 1.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, size );

      auto computeCudaET = [ & ]()
      {
         resultDevice = lpNorm( deviceView, 1.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, size );

      auto computeCudaCUBLAS = [ & ]()
      {
         cublasGasum( cublasHandle, size, deviceVector.getData(), 1, &resultDevice );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", resultDevice, size );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         resultHost = thrust::transform_reduce(
            thrust::host,
            hostVector.getData(),
            hostVector.getData() + hostVector.getSize(),
            []( auto v )
            {
               return std::abs( v );
            },
            0,
            std::plus<>{} );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::transform_reduce", computeThrust );
      verify( "CPU thrust::transform_reduce", resultHost, size );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         resultDevice = thrust::transform_reduce(
            thrust::device,
            thrust::device_pointer_cast( deviceVector.getData() ),
            thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
            [] __cuda_callable__( Real v )
            {
               return std::abs( v );
            },
            0,
            std::plus<>{} );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::transform_reduce", computeThrustDevice );
      verify( "GPU thrust::transform_reduce", resultDevice, size );
   #endif
#endif
   }

   void
   l2norm()
   {
      benchmark.setOperation( "l2 norm", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 2.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, std::sqrt( size ) );

      auto computeET = [ & ]()
      {
         resultHost = lpNorm( hostView, 2.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, std::sqrt( size ) );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         resultHost = blasGnrm2( size, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", resultHost, std::sqrt( size ) );
#endif

      auto computeSTL = [ & ]()
      {
         const auto sum = std::transform_reduce( STDEXEC hostVector.getData(),
                                                 hostVector.getData() + hostVector.getSize(),
                                                 0,
                                                 std::plus<>{},
                                                 []( auto v )
                                                 {
                                                    return v * v;
                                                 } );
         resultHost = std::sqrt( sum );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );
      verify( "CPU std::transform_reduce", resultHost, std::sqrt( size ) );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 2.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, std::sqrt( size ) );

      auto computeCudaET = [ & ]()
      {
         resultDevice = lpNorm( deviceView, 2.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, std::sqrt( size ) );

      auto computeCudaCUBLAS = [ & ]()
      {
         cublasGnrm2( cublasHandle, size, deviceVector.getData(), 1, &resultDevice );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", resultDevice, std::sqrt( size ) );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         const auto sum = thrust::transform_reduce(
            thrust::host,
            hostVector.getData(),
            hostVector.getData() + hostVector.getSize(),
            []( auto v )
            {
               return v * v;
            },
            0,
            std::plus<>{} );
         resultHost = std::sqrt( sum );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::transform_reduce", computeThrust );
      verify( "CPU thrust::transform_reduce", resultHost, std::sqrt( size ) );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         const auto sum = thrust::transform_reduce(
            thrust::device,
            thrust::device_pointer_cast( deviceVector.getData() ),
            thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
            [] __cuda_callable__( Real v )
            {
               return v * v;
            },
            0,
            std::plus<>{} );
         resultDevice = std::sqrt( sum );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::transform_reduce", computeThrustDevice );
      verify( "GPU thrust::transform_reduce", resultDevice, std::sqrt( size ) );
   #endif
#endif
   }

   void
   l3norm()
   {
      benchmark.setOperation( "l3 norm", datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 3.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, std::cbrt( size ) );

      auto computeET = [ & ]()
      {
         resultHost = lpNorm( hostView, 3.0 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, std::cbrt( size ) );

      auto computeSTL = [ & ]()
      {
         const auto sum = std::transform_reduce( STDEXEC hostVector.getData(),
                                                 hostVector.getData() + hostVector.getSize(),
                                                 0,
                                                 std::plus<>{},
                                                 []( auto v )
                                                 {
                                                    return std::pow( v, 3 );
                                                 } );
         resultHost = std::cbrt( sum );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );
      verify( "CPU std::transform_reduce", resultHost, std::cbrt( size ) );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 3.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, std::cbrt( size ) );

      auto computeCudaET = [ & ]()
      {
         resultDevice = lpNorm( deviceView, 3.0 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, std::cbrt( size ) );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         const auto sum = thrust::transform_reduce(
            thrust::host,
            hostVector.getData(),
            hostVector.getData() + hostVector.getSize(),
            []( auto v )
            {
               return std::pow( v, 3 );
            },
            0,
            std::plus<>{} );
         resultHost = std::cbrt( sum );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::transform_reduce", computeThrust );
      verify( "CPU thrust::transform_reduce", resultHost, std::cbrt( size ) );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         const auto sum = thrust::transform_reduce(
            thrust::device,
            thrust::device_pointer_cast( deviceVector.getData() ),
            thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
            [] __cuda_callable__( Real v )
            {
               return std::pow( v, 3 );
            },
            0,
            std::plus<>{} );
         resultDevice = std::cbrt( sum );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::transform_reduce", computeThrustDevice );
      verify( "GPU thrust::transform_reduce", resultDevice, std::cbrt( size ) );
   #endif
#endif
   }

   void
   scalarProduct()
   {
      benchmark.setOperation( "scalar product", 2 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getScalarProduct( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU legacy", computeLegacy );
      verify( "CPU legacy", resultHost, size );

      auto computeET = [ & ]()
      {
         resultHost = ( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", resultHost, size );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         resultHost = blasGdot( size, hostVector.getData(), 1, hostVector2.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", resultHost, size );
#endif

      auto computeSTL = [ & ]()
      {
         resultHost = std::transform_reduce( STDEXEC hostVector.getData(),
                                             hostVector.getData() + hostVector.getSize(),
                                             hostVector2.getData(),
                                             0,
                                             std::plus<>{},
                                             std::multiplies<>{} );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::transform_reduce", computeSTL );
      verify( "CPU std::transform_reduce", resultHost, size );

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getScalarProduct( deviceVector, deviceVector2 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", resultDevice, size );

      auto computeCudaET = [ & ]()
      {
         resultDevice = ( deviceView, deviceView2 );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", resultDevice, size );

      auto computeCudaCUBLAS = [ & ]()
      {
         cublasGdot( cublasHandle, size, deviceVector.getData(), 1, deviceVector2.getData(), 1, &resultDevice );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", resultDevice, size );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         resultHost = thrust::inner_product( thrust::host,
                                             hostVector.getData(),
                                             hostVector.getData() + hostVector.getSize(),
                                             hostVector2.getData(),
                                             0,
                                             std::plus<>{},
                                             std::multiplies<>{} );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::inner_product", computeThrust );
      verify( "CPU thrust::inner_product", resultHost, size );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         resultDevice = thrust::inner_product( thrust::device,
                                               thrust::device_pointer_cast( deviceVector.getData() ),
                                               thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
                                               thrust::device_pointer_cast( deviceVector2.getData() ),
                                               0,
                                               std::plus<>{},
                                               std::multiplies<>{} );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::inner_product", computeThrustDevice );
      verify( "GPU thrust::inner_product", resultDevice, size );
   #endif
#endif
   }

   void
   scalarMultiplication()
   {
      benchmark.setOperation( "scalar multiplication", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         hostVector *= 0.5;
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", hostVector, 0.5 );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         blasGscal( hostVector.getSize(), (Real) 0.5, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( reset1, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", hostVector, 0.5 );
#endif

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         deviceVector *= 0.5;
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector, 0.5 );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 0.5;
         cublasGscal( cublasHandle, size, &alpha, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( reset1, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", deviceVector, 0.5 );
#endif
   }

   void
   vectorAddition()
   {
      benchmark.setOperation( "vector addition", 3 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU legacy", computeLegacy );
      verify( "CPU legacy", hostVector, 2.0 );

      auto computeET = [ & ]()
      {
         hostView += hostView2;
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector, 2.0 );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", hostVector, 2.0 );
#endif

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", deviceVector, 2.0 );

      auto computeCudaET = [ & ]()
      {
         deviceView += deviceView2;
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector, 2.0 );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", deviceVector, 2.0 );
#endif
   }

   void
   twoVectorsAddition()
   {
      benchmark.setOperation( "two vectors addition", 4 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector3, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU legacy", computeLegacy );
      verify( "CPU legacy", hostVector, 3.0 );

      auto computeET = [ & ]()
      {
         hostView += hostView2 + hostView3;
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector, 3.0 );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
         blasGaxpy( size, alpha, hostVector3.getData(), 1, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", hostVector, 3.0 );
#endif

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector3, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", deviceVector, 3.0 );

      auto computeCudaET = [ & ]()
      {
         deviceView += deviceView2 + deviceView3;
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector, 3.0 );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", deviceVector, 3.0 );
#endif
   }

   void
   threeVectorsAddition()
   {
      benchmark.setOperation( "three vectors addition", 5 * datasetSize );

      auto computeLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector3, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector4, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU legacy", computeLegacy );
      verify( "CPU legacy", hostVector, 4.0 );

      auto computeET = [ & ]()
      {
         hostView += hostView2 + hostView3 + hostView4;
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector, 4.0 );

#ifdef HAVE_BLAS
      auto computeBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         blasGaxpy( size, alpha, hostVector2.getData(), 1, hostVector.getData(), 1 );
         blasGaxpy( size, alpha, hostVector3.getData(), 1, hostVector.getData(), 1 );
         blasGaxpy( size, alpha, hostVector4.getData(), 1, hostVector.getData(), 1 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU BLAS", computeBLAS );
      verify( "CPU BLAS", hostVector, 4.0 );
#endif

#ifdef __CUDACC__
      auto computeCudaLegacy = [ & ]()
      {
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector3, (Real) 1.0, (Real) 1.0 );
         Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector4, (Real) 1.0, (Real) 1.0 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", computeCudaLegacy );
      verify( "GPU legacy", deviceVector, 4.0 );

      auto computeCudaET = [ & ]()
      {
         deviceView += deviceView2 + deviceView3 + deviceView4;
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector, 4.0 );

      auto computeCudaCUBLAS = [ & ]()
      {
         const Real alpha = 1.0;
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector2.getData(), 1, deviceVector.getData(), 1 );
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector3.getData(), 1, deviceVector.getData(), 1 );
         cublasGaxpy( cublasHandle, size, &alpha, deviceVector4.getData(), 1, deviceVector.getData(), 1 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", computeCudaCUBLAS );
      verify( "cuBLAS", deviceVector, 4.0 );
#endif
   }

   void
   inclusiveScanInplace()
   {
      benchmark.setOperation( "inclusive scan (inplace)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inplaceInclusiveScan( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", hostVector[ 0 ], 1 );
      verify( "CPU ET", hostVector[ size - 1 ], size );

      auto computeSequential = [ & ]()
      {
         SequentialView view;
         view.bind( hostVector.getData(), hostVector.getSize() );
         Algorithms::inplaceInclusiveScan( view );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU sequential", computeSequential );
      verify( "CPU sequential", hostVector[ 0 ], 1 );
      verify( "CPU sequential", hostVector[ size - 1 ], size );

      auto computeSTL_partial_sum = [ & ]()
      {
         std::partial_sum( hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::partial_sum", computeSTL_partial_sum );
      verify( "CPU std::partial_sum", hostVector[ 0 ], 1 );
      verify( "CPU std::partial_sum", hostVector[ size - 1 ], size );

      auto computeSTL = [ & ]()
      {
         std::inclusive_scan( STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::inclusive_scan", computeSTL );
      verify( "CPU std::inclusive_scan", hostVector[ 0 ], 1 );
      verify( "CPU std::inclusive_scan", hostVector[ size - 1 ], size );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inplaceInclusiveScan( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector.getElement( 0 ), 1 );
      verify( "GPU ET", deviceVector.getElement( size - 1 ), size );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         thrust::inclusive_scan(
            thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::inclusive_scan", computeThrust );
      verify( "CPU thrust::inclusive_scan", hostVector[ 0 ], 1 );
      verify( "CPU thrust::inclusive_scan", hostVector[ size - 1 ], size );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         thrust::inclusive_scan( thrust::device,
                                 thrust::device_pointer_cast( deviceVector.getData() ),
                                 thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
                                 thrust::device_pointer_cast( deviceVector.getData() ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::inclusive_scan", computeThrustDevice );
      verify( "GPU thrust::inclusive_scan", deviceVector.getElement( 0 ), 1 );
      verify( "GPU thrust::inclusive_scan", deviceVector.getElement( size - 1 ), size );
   #endif
#endif
   }

   void
   inclusiveScanOneVector()
   {
      benchmark.setOperation( "inclusive scan (1 vector)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inclusiveScan( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector2[ 0 ], 1 );
      verify( "CPU ET", hostVector2[ size - 1 ], size );

      auto computeSTL_partial_sum = [ & ]()
      {
         std::partial_sum( hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData() );
      };
      benchmark.time< Devices::Sequential >( resetAll, "CPU std::partial_sum", computeSTL_partial_sum );
      verify( "CPU ET", hostVector2[ 0 ], 1 );
      verify( "CPU ET", hostVector2[ size - 1 ], size );

      auto computeSTL = [ & ]()
      {
         std::inclusive_scan(
            STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData() );
      };
      benchmark.time< Devices::Sequential >( resetAll, "CPU std::inclusive_scan", computeSTL );
      verify( "CPU ET", hostVector2[ 0 ], 1 );
      verify( "CPU ET", hostVector2[ size - 1 ], size );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inclusiveScan( deviceVector, deviceVector2 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector2.getElement( 0 ), 1 );
      verify( "GPU ET", deviceVector2.getElement( size - 1 ), size );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         thrust::inclusive_scan(
            thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::inclusive_scan", computeThrust );
      verify( "CPU thrust::inclusive_scan", hostVector2[ 0 ], 1 );
      verify( "CPU thrust::inclusive_scan", hostVector2[ size - 1 ], size );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         thrust::inclusive_scan( thrust::device,
                                 thrust::device_pointer_cast( deviceVector.getData() ),
                                 thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
                                 thrust::device_pointer_cast( deviceVector2.getData() ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::inclusive_scan", computeThrustDevice );
      verify( "GPU thrust::inclusive_scan", deviceVector2.getElement( 0 ), 1 );
      verify( "GPU thrust::inclusive_scan", deviceVector2.getElement( size - 1 ), size );
   #endif
#endif
   }

   void
   inclusiveScanTwoVectors()
   {
      benchmark.setOperation( "inclusive scan (2 vectors)", 3 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inclusiveScan( hostVector + hostVector2, hostVector3 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector3[ 0 ], 2 );
      verify( "CPU ET", hostVector3[ size - 1 ], 2 * size );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inclusiveScan( deviceVector + deviceVector2, deviceVector3 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector3.getElement( 0 ), 2 );
      verify( "GPU ET", deviceVector3.getElement( size - 1 ), 2 * size );
#endif
   }

   void
   inclusiveScanThreeVectors()
   {
      auto computeET = [ & ]()
      {
         Algorithms::inclusiveScan( hostVector + hostVector2 + hostVector3, hostVector4 );
      };
      benchmark.setOperation( "inclusive scan (3 vectors)", 4 * datasetSize );
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector4[ 0 ], 3 );
      verify( "CPU ET", hostVector4[ size - 1 ], 3 * size );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inclusiveScan( deviceVector + deviceVector2 + deviceVector3, deviceVector4 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "GPU ET", deviceVector4.getElement( 0 ), 3 );
      verify( "GPU ET", deviceVector4.getElement( size - 1 ), 3 * size );
#endif
   }

   void
   exclusiveScanInplace()
   {
      benchmark.setOperation( "exclusive scan (inplace)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::inplaceExclusiveScan( hostVector );
      };
      benchmark.time< Devices::Host >( reset1, "CPU ET", computeET );
      verify( "CPU ET", hostVector[ 0 ], 0 );
      verify( "CPU ET", hostVector[ size - 1 ], size - 1 );

      auto computeSequential = [ & ]()
      {
         SequentialView view;
         view.bind( hostVector.getData(), hostVector.getSize() );
         Algorithms::inplaceExclusiveScan( view );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU sequential", computeSequential );
      verify( "CPU sequential", hostVector[ 0 ], 0 );
      verify( "CPU sequential", hostVector[ size - 1 ], size - 1 );

      auto computeSTL = [ & ]()
      {
         std::exclusive_scan(
            STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData(), 0 );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::exclusive_scan", computeSTL );
      verify( "CPU std::exclusive_scan", hostVector[ 0 ], 0 );
      // NOTE: this fails due to https://stackoverflow.com/q/74932677
      verify( "CPU std::exclusive_scan", hostVector[ size - 1 ], size - 1 );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::inplaceExclusiveScan( deviceVector );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU ET", computeCudaET );
      verify( "CPU ET", deviceVector.getElement( 0 ), 0 );
      verify( "GPU ET", deviceVector.getElement( size - 1 ), size - 1 );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         thrust::exclusive_scan(
            thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::exclusive_scan", computeThrust );
      verify( "CPU thrust::exclusive_scan", hostVector[ 0 ], 0 );
      verify( "CPU thrust::exclusive_scan", hostVector[ size - 1 ], size - 1 );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         thrust::exclusive_scan( thrust::device,
                                 thrust::device_pointer_cast( deviceVector.getData() ),
                                 thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
                                 thrust::device_pointer_cast( deviceVector.getData() ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::exclusive_scan", computeThrustDevice );
      verify( "GPU thrust::exclusive_scan", deviceVector.getElement( 0 ), 0 );
      verify( "GPU thrust::exclusive_scan", deviceVector.getElement( size - 1 ), size - 1 );
   #endif
#endif
   }

   void
   exclusiveScanOneVector()
   {
      benchmark.setOperation( "exclusive scan (1 vector)", 2 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::exclusiveScan( hostVector, hostVector2 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector2[ 0 ], 0 );
      verify( "CPU ET", hostVector2[ size - 1 ], size - 1 );

      auto computeSTL = [ & ]()
      {
         std::exclusive_scan(
            STDEXEC hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData(), 0 );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU std::exclusive_scan", computeSTL );
      verify( "CPU std::exclusive_scan", hostVector2[ 0 ], 0 );
      verify( "CPU std::exclusive_scan", hostVector2[ size - 1 ], size - 1 );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::exclusiveScan( deviceVector, deviceVector2 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "CPU ET", deviceVector2.getElement( 0 ), 0 );
      verify( "GPU ET", deviceVector2.getElement( size - 1 ), size - 1 );
#endif

#ifdef HAVE_THRUST
      auto computeThrust = [ & ]()
      {
         thrust::exclusive_scan(
            thrust::host, hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector2.getData() );
      };
      benchmark.time< Devices::Sequential >( reset1, "CPU thrust::exclusive_scan", computeThrust );
      verify( "CPU thrust::exclusive_scan", hostVector2[ 0 ], 0 );
      verify( "CPU thrust::exclusive_scan", hostVector2[ size - 1 ], size - 1 );

   #ifdef __CUDACC__
      auto computeThrustDevice = [ & ]()
      {
         thrust::exclusive_scan( thrust::device,
                                 thrust::device_pointer_cast( deviceVector.getData() ),
                                 thrust::device_pointer_cast( deviceVector.getData() + deviceVector.getSize() ),
                                 thrust::device_pointer_cast( deviceVector2.getData() ) );
      };
      benchmark.time< Devices::Cuda >( reset1, "GPU thrust::exclusive_scan", computeThrustDevice );
      verify( "GPU thrust::exclusive_scan", deviceVector2.getElement( 0 ), 0 );
      verify( "GPU thrust::exclusive_scan", deviceVector2.getElement( size - 1 ), size - 1 );
   #endif
#endif
   }

   void
   exclusiveScanTwoVectors()
   {
      benchmark.setOperation( "exclusive scan (2 vectors)", 3 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::exclusiveScan( hostVector + hostVector2, hostVector3 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector3[ 0 ], 0 );
      verify( "CPU ET", hostVector3[ size - 1 ], 2 * ( size - 1 ) );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::exclusiveScan( deviceVector + deviceVector2, deviceVector3 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "CPU ET", deviceVector3.getElement( 0 ), 0 );
      verify( "GPU ET", deviceVector3.getElement( size - 1 ), 2 * ( size - 1 ) );
#endif
   }

   void
   exclusiveScanThreeVectors()
   {
      benchmark.setOperation( "exclusive scan (3 vectors)", 4 * datasetSize );

      auto computeET = [ & ]()
      {
         Algorithms::exclusiveScan( hostVector + hostVector2 + hostVector3, hostVector4 );
      };
      benchmark.time< Devices::Host >( resetAll, "CPU ET", computeET );
      verify( "CPU ET", hostVector4[ 0 ], 0 );
      verify( "CPU ET", hostVector4[ size - 1 ], 3 * ( size - 1 ) );

#ifdef __CUDACC__
      auto computeCudaET = [ & ]()
      {
         Algorithms::exclusiveScan( deviceVector + deviceVector2 + deviceVector3, deviceVector4 );
      };
      benchmark.time< Devices::Cuda >( resetAll, "GPU ET", computeCudaET );
      verify( "CPU ET", deviceVector4.getElement( 0 ), 0 );
      verify( "GPU ET", deviceVector4.getElement( size - 1 ), 3 * ( size - 1 ) );
#endif
   }
};

template< typename Real = double, typename Index = int >
void
benchmarkVectorOperations( Benchmark<>& benchmark, const long& size )
{
   VectorOperationsBenchmark< Real, Index > ops( benchmark, size );
   ops.max();
   ops.min();
   ops.absMax();
   ops.absMin();
   ops.sum();
   ops.l1norm();
   ops.l2norm();
   ops.l3norm();
   ops.scalarProduct();
   ops.scalarMultiplication();
   ops.vectorAddition();
   ops.twoVectorsAddition();
   ops.threeVectorsAddition();
   ops.inclusiveScanInplace();
   ops.inclusiveScanOneVector();
   ops.inclusiveScanTwoVectors();
   ops.inclusiveScanThreeVectors();
   ops.exclusiveScanInplace();
   ops.exclusiveScanOneVector();
   ops.exclusiveScanTwoVectors();
   ops.exclusiveScanThreeVectors();
}

}  // namespace Benchmarks
}  // namespace TNL
