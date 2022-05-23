// Implemented by: Jakub Klinkovsky

#pragma once

#include <cstdlib>  // srand48
#include <numeric>  // std::partial_sum

#include <TNL/Benchmarks/Benchmarks.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/scan.h>
#include "CommonVectorOperations.h"
#include "VectorOperations.h"

#ifdef HAVE_BLAS
#include "blasWrappers.h"
#endif

#ifdef HAVE_CUDA
#include "cublasWrappers.h"
#endif

namespace TNL {
namespace Benchmarks {

template< typename Real = double,
          typename Index = int >
void
benchmarkVectorOperations( Benchmark<> & benchmark,
                           const long & size )
{
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using CudaVector =  Containers::Vector< Real, Devices::Cuda, Index >;
   using SequentialView = Containers::VectorView< Real, Devices::Sequential, Index >;
   using HostView = Containers::VectorView< Real, Devices::Host, Index >;
   using CudaView =  Containers::VectorView< Real, Devices::Cuda, Index >;

   using namespace std;

   double datasetSize = (double) size * sizeof( Real ) / oneGB;

   HostVector hostVector( size );
   HostVector hostVector2( size );
   HostVector hostVector3( size );
   HostVector hostVector4( size );
   CudaVector deviceVector;
   CudaVector deviceVector2;
   CudaVector deviceVector3;
   CudaVector deviceVector4;
#ifdef HAVE_CUDA
   deviceVector.setSize( size );
   deviceVector2.setSize( size );
   deviceVector3.setSize( size );
   deviceVector4.setSize( size );
#endif

   HostView hostView( hostVector );
   HostView hostView2( hostVector2 );
   HostView hostView3( hostVector3 );
   HostView hostView4( hostVector4 );
#ifdef HAVE_CUDA
   CudaView deviceView( deviceVector ), deviceView2( deviceVector2 ), deviceView3( deviceVector3 ), deviceView4( deviceVector4 );
#endif

   Real resultHost;
   Real resultDevice;

#ifdef HAVE_CUDA
   cublasHandle_t cublasHandle;
   cublasCreate( &cublasHandle );
#endif


   // reset functions
   // (Make sure to always use some in benchmarks, even if it's not necessary
   // to assure correct result - it helps to clear cache and avoid optimizations
   // of the benchmark loop.)
   auto reset1 = [&]() {
      hostVector.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector.setValue( 1.0 );
#endif
      // A relatively harmless call to keep the compiler from realizing we
      // don't actually do any useful work with the result of the reduction.
      srand48(resultHost);
      resultHost = resultDevice = 0.0;
   };
   auto reset2 = [&]() {
      hostVector2.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector2.setValue( 1.0 );
#endif
   };
   auto reset3 = [&]() {
      hostVector3.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector3.setValue( 1.0 );
#endif
   };
   auto reset4 = [&]() {
      hostVector4.setValue( 1.0 );
#ifdef HAVE_CUDA
      deviceVector4.setValue( 1.0 );
#endif
   };


   auto resetAll = [&]() {
      reset1();
      reset2();
      reset3();
      reset4();
   };

   resetAll();

   ////
   // Max
   auto maxHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMax( hostVector );
   };
   auto maxHostET = [&]() {
      resultHost = max( hostView );
   };
   benchmark.setOperation( "max", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", maxHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", maxHostET );
#ifdef HAVE_CUDA
   auto maxCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMax( deviceVector );
   };
   auto maxCudaET = [&]() {
      resultDevice = max( deviceView );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", maxCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", maxCudaET );
#endif

   ////
   // Min
   auto minHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorMin( hostVector );
   };
   auto minHostET = [&]() {
      resultHost = min( hostView );
   };
   benchmark.setOperation( "min", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", minHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", minHostET );
#ifdef HAVE_CUDA
   auto minCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorMin( deviceVector );
   };
   auto minCudaET = [&]() {
      resultDevice = min( deviceView );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", minCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", minCudaET );
#endif

   ////
   // Absmax
   auto absMaxHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMax( hostVector );
   };
   auto absMaxHostET = [&]() {
      resultHost = max( abs( hostView ) );
   };
#ifdef HAVE_BLAS
   auto absMaxBlas = [&]() {
      int index = blasIgamax( size, hostVector.getData(), 1 );
      resultHost = hostVector.getElement( index );
   };
#endif
   benchmark.setOperation( "absMax", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", absMaxHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", absMaxHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", absMaxBlas );
#endif
#ifdef HAVE_CUDA
   auto absMaxCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMax( deviceVector );
   };
   auto absMaxCudaET = [&]() {
      resultDevice = max( abs( deviceView ) );
   };
   auto absMaxCublas = [&]() {
      int index = 0;
      cublasIgamax( cublasHandle, size,
                    deviceVector.getData(), 1,
                    &index );
      resultDevice = deviceVector.getElement( index );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", absMaxCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", absMaxCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", absMaxCublas );
#endif

   ////
   // Absmin
   auto absMinHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorAbsMin( hostVector );
   };
   auto absMinHostET = [&]() {
      resultHost = min( abs( hostView ) );
   };
/*#ifdef HAVE_BLAS
   auto absMinBlas = [&]() {
      int index = blasIgamin( size, hostVector.getData(), 1 );
      resultHost = hostVector.getElement( index );
   };
#endif*/
   benchmark.setOperation( "absMin", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", absMinHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", absMinHostET );
   //benchmark.time< Devices::Host >( reset1, "CPU BLAS", absMinBlas );
#ifdef HAVE_CUDA
   auto absMinCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorAbsMin( deviceVector );
   };
   auto absMinCudaET = [&]() {
      resultDevice = min( abs( deviceView ) );
   };
   auto absMinCublas = [&]() {
      int index = 0;
      cublasIgamin( cublasHandle, size,
                    deviceVector.getData(), 1,
                    &index );
      resultDevice = deviceVector.getElement( index );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", absMinCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", absMinCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", absMinCublas );
#endif

   ////
   // Sum
   auto sumHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorSum( hostVector );
   };
   auto sumHostET = [&]() {
      resultHost = sum( hostView );
   };
   benchmark.setOperation( "sum", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", sumHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", sumHostET );
#ifdef HAVE_CUDA
   auto sumCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorSum( deviceVector );
   };
   auto sumCudaET = [&]() {
      resultDevice = sum( deviceView );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", sumCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", sumCudaET );
#endif

   ////
   // L1 norm
   auto l1normHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 1.0 );
   };
   auto l1normHostET = [&]() {
      resultHost = lpNorm( hostView, 1.0 );
   };
#ifdef HAVE_BLAS
   auto l1normBlas = [&]() {
      resultHost = blasGasum( size, hostVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "l1 norm", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", l1normHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", l1normHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", l1normBlas );
#endif
#ifdef HAVE_CUDA
   auto l1normCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 1.0 );
   };
   auto l1normCudaET = [&]() {
      resultDevice = lpNorm( deviceView, 1.0 );
   };
   auto l1normCublas = [&]() {
      cublasGasum( cublasHandle, size,
                   deviceVector.getData(), 1,
                   &resultDevice );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", l1normCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", l1normCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", l1normCublas );
#endif

   ////
   // L2 norm
   auto l2normHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 2.0 );
   };
   auto l2normHostET = [&]() {
      resultHost = lpNorm( hostView, 2.0 );
   };
#ifdef HAVE_BLAS
   auto l2normBlas = [&]() {
      resultHost = blasGnrm2( size, hostVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "l2 norm", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", l2normHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", l2normHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", l2normBlas );
#endif
#ifdef HAVE_CUDA
   auto l2normCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 2.0 );
   };
   auto l2normCudaET = [&]() {
      resultDevice = lpNorm( deviceView, 2.0 );
   };
   auto l2normCublas = [&]() {
      cublasGnrm2( cublasHandle, size,
                   deviceVector.getData(), 1,
                   &resultDevice );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", l2normCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", l2normCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", l2normCublas );
#endif

   ////
   // L3 norm
   auto l3normHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getVectorLpNorm( hostVector, 3.0 );
   };
   auto l3normHostET = [&]() {
      resultHost = lpNorm( hostView, 3.0 );
   };
   benchmark.setOperation( "l3 norm", datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", l3normHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", l3normHostET );
#ifdef HAVE_CUDA
   auto l3normCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getVectorLpNorm( deviceVector, 3.0 );
   };
   auto l3normCudaET = [&]() {
      resultDevice = lpNorm( deviceView, 3.0 );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", l3normCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", l3normCudaET );
#endif

   ////
   // Scalar product
   auto scalarProductHost = [&]() {
      resultHost = Benchmarks::CommonVectorOperations< Devices::Host >::getScalarProduct( hostVector, hostVector2 );
   };
   auto scalarProductHostET = [&]() {
      resultHost = ( hostVector, hostVector2 );
   };
#ifdef HAVE_BLAS
   auto scalarProductBlas = [&]() {
      resultHost = blasGdot( size, hostVector.getData(), 1, hostVector2.getData(), 1 );
   };
#endif
   benchmark.setOperation( "scalar product", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU legacy", scalarProductHost );
   benchmark.time< Devices::Host >( reset1, "CPU ET", scalarProductHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", scalarProductBlas );
#endif
#ifdef HAVE_CUDA
   auto scalarProductCuda = [&]() {
      resultDevice = Benchmarks::CommonVectorOperations< Devices::Cuda >::getScalarProduct( deviceVector, deviceVector2 );
   };
   auto scalarProductCudaET = [&]() {
      resultDevice = ( deviceView, deviceView2 );
   };
   auto scalarProductCublas = [&]() {
      cublasGdot( cublasHandle, size,
                  deviceVector.getData(), 1,
                  deviceVector2.getData(), 1,
                  &resultDevice );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU legacy", scalarProductCuda );
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", scalarProductCudaET );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", scalarProductCublas );
#endif

   ////
   // Scalar multiplication
   auto multiplyHost = [&]() {
      hostVector *= 0.5;
   };
#ifdef HAVE_BLAS
   auto multiplyBlas = [&]() {
      blasGscal( hostVector.getSize(), (Real) 0.5, hostVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "scalar multiplication", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU ET", multiplyHost );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( reset1, "CPU BLAS", multiplyBlas );
#endif
#ifdef HAVE_CUDA
   auto multiplyCuda = [&]() {
      deviceVector *= 0.5;
   };
   auto multiplyCublas = [&]() {
      const Real alpha = 0.5;
      cublasGscal( cublasHandle, size,
                   &alpha,
                   deviceVector.getData(), 1 );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", multiplyCuda );
   benchmark.time< Devices::Cuda >( reset1, "cuBLAS", multiplyCublas );
#endif

   ////
   // Vector addition
   auto addVectorHost = [&]() {
      Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
   };
   auto addVectorHostET = [&]() {
      hostView += hostView2;
   };
#ifdef HAVE_BLAS
   auto addVectorBlas = [&]() {
      const Real alpha = 1.0;
      blasGaxpy( size, alpha,
                 hostVector2.getData(), 1,
                 hostVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "vector addition", 3 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU legacy", addVectorHost );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", addVectorHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( resetAll, "CPU BLAS", addVectorBlas );
#endif
#ifdef HAVE_CUDA
   auto addVectorCuda = [&]() {
      Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
   };
   auto addVectorCudaET = [&]() {
      deviceView += deviceView2;
   };
   auto addVectorCublas = [&]() {
      const Real alpha = 1.0;
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector2.getData(), 1,
                   deviceVector.getData(), 1 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", addVectorCuda );
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", addVectorCudaET );
   benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", addVectorCublas );
#endif

   ////
   // Two vectors addition
   auto addTwoVectorsHost = [&]() {
      Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
      Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector3, (Real) 1.0, (Real) 1.0 );
   };
   auto addTwoVectorsHostET = [&]() {
      hostView += hostView2 + hostView3;
   };
#ifdef HAVE_BLAS
   auto addTwoVectorsBlas = [&]() {
      const Real alpha = 1.0;
      blasGaxpy( size, alpha,
                 hostVector2.getData(), 1,
                 hostVector.getData(), 1 );
      blasGaxpy( size, alpha,
                 hostVector3.getData(), 1,
                 hostVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "two vectors addition", 4 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU legacy", addTwoVectorsHost );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", addTwoVectorsHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( resetAll, "CPU BLAS", addTwoVectorsBlas );
#endif
#ifdef HAVE_CUDA
   auto addTwoVectorsCuda = [&]() {
      Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
      Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector3, (Real) 1.0, (Real) 1.0 );
   };
   auto addTwoVectorsCudaET = [&]() {
      deviceView += deviceView2 + deviceView3;
   };
   auto addTwoVectorsCublas = [&]() {
      const Real alpha = 1.0;
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector2.getData(), 1,
                   deviceVector.getData(), 1 );
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector3.getData(), 1,
                   deviceVector.getData(), 1 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", addTwoVectorsCuda );
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", addTwoVectorsCudaET );
   benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", addTwoVectorsCublas );
#endif

   ////
   // Three vectors addition
   auto addThreeVectorsHost = [&]() {
      Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector2, (Real) 1.0, (Real) 1.0 );
      Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector3, (Real) 1.0, (Real) 1.0 );
      Benchmarks::VectorOperations< Devices::Host >::addVector( hostVector, hostVector4, (Real) 1.0, (Real) 1.0 );
   };
   auto addThreeVectorsHostET = [&]() {
      hostView += hostView2 + hostView3 + hostView4;
   };
#ifdef HAVE_BLAS
   auto addThreeVectorsBlas = [&]() {
      const Real alpha = 1.0;
      blasGaxpy( size, alpha,
                 hostVector2.getData(), 1,
                 hostVector.getData(), 1 );
      blasGaxpy( size, alpha,
                 hostVector3.getData(), 1,
                 hostVector.getData(), 1 );
      blasGaxpy( size, alpha,
                 hostVector4.getData(), 1,
                 hostVector.getData(), 1 );
   };
#endif
   benchmark.setOperation( "three vectors addition", 5 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU legacy", addThreeVectorsHost );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", addThreeVectorsHostET );
#ifdef HAVE_BLAS
   benchmark.time< Devices::Host >( resetAll, "CPU BLAS", addThreeVectorsBlas );
#endif
#ifdef HAVE_CUDA
   auto addThreeVectorsCuda = [&]() {
      Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector2, (Real) 1.0, (Real) 1.0 );
      Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector3, (Real) 1.0, (Real) 1.0 );
      Benchmarks::VectorOperations< Devices::Cuda >::addVector( deviceVector, deviceVector4, (Real) 1.0, (Real) 1.0 );
   };
   auto addThreeVectorsCudaET = [&]() {
      deviceView += deviceView2 + deviceView3 + deviceView4;
   };
   auto addThreeVectorsCublas = [&]() {
      const Real alpha = 1.0;
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector2.getData(), 1,
                   deviceVector.getData(), 1 );
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector3.getData(), 1,
                   deviceVector.getData(), 1 );
      cublasGaxpy( cublasHandle, size,
                   &alpha,
                   deviceVector4.getData(), 1,
                   deviceVector.getData(), 1 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU legacy", addThreeVectorsCuda );
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", addThreeVectorsCudaET );
   benchmark.time< Devices::Cuda >( resetAll, "cuBLAS", addThreeVectorsCublas );
#endif

   ////
   // Inplace inclusive scan
   auto inplaceInclusiveScanHost = [&]() {
      Algorithms::inplaceInclusiveScan( hostVector );
   };
   auto inplaceInclusiveScanSequential = [&]() {
      SequentialView view;
      view.bind( hostVector.getData(), hostVector.getSize() );
      Algorithms::inplaceInclusiveScan( view );
   };
   auto inplaceInclusiveScanSTL = [&]() {
      std::partial_sum( hostVector.getData(), hostVector.getData() + hostVector.getSize(), hostVector.getData() );
   };
   benchmark.setOperation( "inclusive scan (inplace)", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU ET", inplaceInclusiveScanHost );
   benchmark.time< Devices::Sequential >( reset1, "CPU sequential", inplaceInclusiveScanSequential );
   benchmark.time< Devices::Sequential >( reset1, "CPU std::partial_sum", inplaceInclusiveScanSTL );
   // TODO: there are also `std::inclusive_scan` and `std::exclusive_scan` since C++17 which are parallel,
   // add them to the benchmark when we use C++17
#ifdef HAVE_CUDA
   auto inplaceInclusiveScanCuda = [&]() {
      Algorithms::inplaceInclusiveScan( deviceVector );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", inplaceInclusiveScanCuda );
#endif

   ////
   // Inclusive scan of one vector
   auto inclusiveScanOneVectorHost = [&]() {
      Algorithms::inclusiveScan( hostVector, hostVector2 );
   };
   benchmark.setOperation( "inclusive scan (1 vector)", 2 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", inclusiveScanOneVectorHost );
#ifdef HAVE_CUDA
   auto inclusiveScanOneVectorCuda = [&]() {
      Algorithms::inclusiveScan( deviceVector, deviceVector2 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", inclusiveScanOneVectorCuda );
#endif

   ////
   // Inclusive scan of two vectors
   auto inclusiveScanTwoVectorsHost = [&]() {
      Algorithms::inclusiveScan( hostVector + hostVector2, hostVector3 );
   };
   benchmark.setOperation( "inclusive scan (2 vectors)", 3 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", inclusiveScanTwoVectorsHost );
#ifdef HAVE_CUDA
   auto inclusiveScanTwoVectorsCuda = [&]() {
      Algorithms::inclusiveScan( deviceVector + deviceVector2, deviceVector3 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", inclusiveScanTwoVectorsCuda );
#endif

   ////
   // Inclusive scan of three vectors
   auto inclusiveScanThreeVectorsHost = [&]() {
      Algorithms::inclusiveScan( hostVector + hostVector2 + hostVector3, hostVector4 );
   };
   benchmark.setOperation( "inclusive scan (3 vectors)", 4 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", inclusiveScanThreeVectorsHost );
#ifdef HAVE_CUDA
   auto inclusiveScanThreeVectorsCuda = [&]() {
      Algorithms::inclusiveScan( deviceVector + deviceVector2 + deviceVector3, deviceVector4 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", inclusiveScanThreeVectorsCuda );
#endif

   ////
   // Inplace exclusive scan
   auto inplaceExclusiveScanHost = [&]() {
      Algorithms::inplaceExclusiveScan( hostVector );
   };
   auto inplaceExclusiveScanSequential = [&]() {
      SequentialView view;
      view.bind( hostVector.getData(), hostVector.getSize() );
      Algorithms::inplaceExclusiveScan( view );
   };
   benchmark.setOperation( "exclusive scan (inplace)", 2 * datasetSize );
   benchmark.time< Devices::Host >( reset1, "CPU ET", inplaceExclusiveScanHost );
   benchmark.time< Devices::Sequential >( reset1, "CPU sequential", inplaceExclusiveScanSequential );
#ifdef HAVE_CUDA
   auto inplaceExclusiveScanCuda = [&]() {
      Algorithms::inplaceExclusiveScan( deviceVector );
   };
   benchmark.time< Devices::Cuda >( reset1, "GPU ET", inplaceExclusiveScanCuda );
#endif

   ////
   // Exclusive scan of one vector
   auto exclusiveScanOneVectorHost = [&]() {
      Algorithms::exclusiveScan( hostVector, hostVector2 );
   };
   benchmark.setOperation( "exclusive scan (1 vector)", 2 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", exclusiveScanOneVectorHost );
#ifdef HAVE_CUDA
   auto exclusiveScanOneVectorCuda = [&]() {
      Algorithms::exclusiveScan( deviceVector, deviceVector2 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", exclusiveScanOneVectorCuda );
#endif

   ////
   // Exclusive scan of two vectors
   auto exclusiveScanTwoVectorsHost = [&]() {
      Algorithms::exclusiveScan( hostVector + hostVector2, hostVector3 );
   };
   benchmark.setOperation( "exclusive scan (2 vectors)", 3 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", exclusiveScanTwoVectorsHost );
#ifdef HAVE_CUDA
   auto exclusiveScanTwoVectorsCuda = [&]() {
      Algorithms::exclusiveScan( deviceVector + deviceVector2, deviceVector3 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", exclusiveScanTwoVectorsCuda );
#endif

   ////
   // Exclusive scan of three vectors
   auto exclusiveScanThreeVectorsHost = [&]() {
      Algorithms::exclusiveScan( hostVector + hostVector2 + hostVector3, hostVector4 );
   };
   benchmark.setOperation( "exclusive scan (3 vectors)", 4 * datasetSize );
   benchmark.time< Devices::Host >( resetAll, "CPU ET", exclusiveScanThreeVectorsHost );
#ifdef HAVE_CUDA
   auto exclusiveScanThreeVectorsCuda = [&]() {
      Algorithms::exclusiveScan( deviceVector + deviceVector2 + deviceVector3, deviceVector4 );
   };
   benchmark.time< Devices::Cuda >( resetAll, "GPU ET", exclusiveScanThreeVectorsCuda );
#endif

#ifdef HAVE_CUDA
   cublasDestroy( cublasHandle );
#endif
}

} // namespace Benchmarks
} // namespace TNL
