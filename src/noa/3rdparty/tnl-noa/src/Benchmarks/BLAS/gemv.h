// Implemented by: Jakub Klinkovsky, Tomas Oberhuber

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
#include "cublasWrappers.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Host.h>

namespace TNL {
namespace Benchmarks {

template< typename Matrix >
void setMatrix( Matrix& matrix )
{
   matrix.setValue( 1.0 );
}

template< typename Real >
void
benchmarkGemv( Benchmark<> & benchmark, int rows, int columns )
{
   using HostMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Host >;
   using RowMajorCudaMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >;
   using ColumnMajorCudaMatrix = TNL::Matrices::DenseMatrix< Real, TNL::Devices::Cuda >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   HostMatrix hostMatrix;
   RowMajorCudaMatrix rowMajorCudaMatrix;
   ColumnMajorCudaMatrix columnMajorCudaMatrix;
   HostVector inHostVector;
   HostVector outHostVector;
   CudaVector inCudaVector;
   CudaVector outCudaVector1;
   CudaVector outCudaVector2;

   hostMatrix.setDimensions( rows, columns );
   inHostVector.setSize( columns );
   outHostVector.setSize( rows );

   setMatrix< HostMatrix >( hostMatrix );
   const double datasetSize = (double) ( rows * columns + rows + columns ) * sizeof(Real) / oneGB;
   benchmark.setOperation( "gemv", datasetSize );

   // reset function
   auto reset = [&]() {
      inHostVector = 1.0;
      outHostVector = 0.0;
#ifdef __CUDACC__
      inCudaVector = 1.0;
      //outCudaVector1 = 0.0;
      //outCudaVector2 = 0.0;
#endif
   };

   // compute functions
   auto spmvHost = [&]() {
      hostMatrix.vectorProduct( inHostVector, outHostVector );
   };
   benchmark.time< Devices::Host >( reset, "CPU", spmvHost );

#ifdef __CUDACC__
   columnMajorCudaMatrix.setDimensions( rows, columns );
   inCudaVector.setSize( columns );
   outCudaVector1.setSize( rows );
   outCudaVector2.setSize( rows );
   setMatrix< ColumnMajorCudaMatrix >( columnMajorCudaMatrix );

   auto columnMajorMvCuda = [&]() {
      columnMajorCudaMatrix.vectorProduct( inCudaVector, outCudaVector1 );
   };
   benchmark.time< Devices::Cuda >( reset, "GPU col", columnMajorMvCuda );

   columnMajorCudaMatrix.reset();

   rowMajorCudaMatrix.setDimensions( rows, columns );
   setMatrix< RowMajorCudaMatrix >( rowMajorCudaMatrix );

   auto rowMajorMvCuda = [&]() {
      rowMajorCudaMatrix.vectorProduct( inCudaVector, outCudaVector2 );
   };
   benchmark.time< Devices::Cuda >( reset, "GPU row", rowMajorMvCuda );

   //auto diff = TNL::max( abs( outCudaVector2 - outCudaVector1 ) );
   //std::cerr << outCudaVector1 << std::endl << outCudaVector2 << std::endl;

   rowMajorCudaMatrix.reset();
   columnMajorCudaMatrix.setDimensions( rows, columns );
   setMatrix< ColumnMajorCudaMatrix >( columnMajorCudaMatrix );

   cublasHandle_t cublasHandle;
   cublasCreate( &cublasHandle );
   auto mvCublas = [&] () {
      Real alpha = 1.0;
      Real beta = 0.0;
      cublasGemv( cublasHandle, CUBLAS_OP_N, rows, columns, &alpha,
                  columnMajorCudaMatrix.getValues().getData(), rows,
                  inCudaVector.getData(), 1, &beta,
                  outCudaVector1.getData(), 1 );
   };
   benchmark.time< Devices::Cuda >( reset, "GPU cublas", mvCublas );

   //std::cerr << "Diff. = " << diff << std::endl;
#endif
}

} // namespace Benchmarks
} // namespace TNL
