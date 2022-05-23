// Implemented by: Lukas Cejka
//      Original implemented by J. Klinkovsky in Benchmarks/BLAS
//      This is an edited copy of Benchmarks/BLAS/spmv.h by: Lukas Cejka

#pragma once

#include <cstdint>

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Benchmarks/JsonLogging.h>
#include "SpmvBenchmarkResult.h"

#include <TNL/Pointers/DevicePointer.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/AdEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/BiEllpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/LegacyMatrixReader.h>

#include <TNL/Matrices/MatrixInfo.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>

#ifdef HAVE_PETSC
#include <petscmat.h>
#endif

// Uncomment the following line to enable benchmarking the sandbox sparse matrix.
//#define WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>
#endif

using namespace TNL::Matrices;

#include <Benchmarks/SpMV/ReferenceFormats/cusparseCSRMatrix.h>
#include <Benchmarks/SpMV/ReferenceFormats/cusparseCSRMatrixLegacy.h>
#include <Benchmarks/SpMV/ReferenceFormats/LightSpMVBenchmark.h>
#include <Benchmarks/SpMV/ReferenceFormats/CSR5Benchmark.h>

namespace TNL {
   namespace Benchmarks {
      namespace SpMV {

using BenchmarkType = TNL::Benchmarks::Benchmark< JsonLogging >;

/////
// General sparse matrix aliases
//
template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Scalar = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Vector = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Hybrid = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRHybrid >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Light = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRLight >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_CSR_Adaptive = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Algorithms::Segments::CSRAdaptive >;

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_Ellpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, EllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_SlicedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, SlicedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using ChunkedEllpackSegments = Algorithms::Segments::ChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_ChunkedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, ChunkedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using BiEllpackSegments = Algorithms::Segments::BiEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SparseMatrix_BiEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, BiEllpackSegments >;

/////
// Symmetric sparse matrix aliases
//
template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Scalar = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Vector = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRVector >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Hybrid = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRHybrid >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Light = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRLight >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_CSR_Adaptive = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, Algorithms::Segments::CSRAdaptive >;

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_Ellpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, EllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_SlicedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, SlicedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using ChunkedEllpackSegments = Algorithms::Segments::ChunkedEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_ChunkedEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, ChunkedEllpackSegments >;

template< typename Device, typename Index, typename IndexAllocator >
using BiEllpackSegments = Algorithms::Segments::BiEllpack< Device, Index, IndexAllocator >;

template< typename Real, typename Device, typename Index >
using SymmetricSparseMatrix_BiEllpack = Matrices::SparseMatrix< Real, Device, Index, Matrices::SymmetricMatrix, BiEllpackSegments >;

#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
template< typename Real, typename Device, typename Index >
using SparseSandboxMatrix = Matrices::Sandbox::SparseSandboxMatrix< Real, Device, Index, Matrices::GeneralMatrix >;
#endif

/////
// Legacy formats
//
template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Scalar = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRScalar >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Vector = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light2 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight2 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light3 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight3 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light4 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight4 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light5 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight5 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Light6 = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight6 >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_Adaptive = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRAdaptive >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_MultiVector = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRMultiVector >;

template< typename Real, typename Device, typename Index >
using SparseMatrixLegacy_CSR_LightWithoutAtomic = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLightWithoutAtomic >;

template< typename Real, typename Device, typename Index >
using SlicedEllpackAlias = Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< Real, Device, Index >;

template< typename Real,
          template< typename, typename, typename > class Matrix >
void
benchmarkSpMVLegacy( BenchmarkType& benchmark,
                     const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                     const String& inputFileName,
                     bool allCpuTests,
                     bool verboseMR )
{
   using HostMatrix = Matrix< Real, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< Real, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() });

   HostMatrix hostMatrix;
   CudaMatrix cudaMatrix;

   try
   {
      SpMV::ReferenceFormats::Legacy::LegacyMatrixReader< HostMatrix >::readMtxFile( inputFileName, hostMatrix, verboseMR );
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to read the matrix:" + String(e.what()) );
      return;
   }

   const int nonzeros = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   if( allCpuTests )
   {
      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector );

      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef HAVE_CUDA
   try
   {
      cudaMatrix = hostMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
}

template< typename Real,
          typename InputMatrix,
          template< typename, typename, typename > class Matrix >
void
benchmarkSpMV( BenchmarkType& benchmark,
               const InputMatrix& inputMatrix,
               const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
               const String& inputFileName,
               bool allCpuTests,
               bool verboseMR )
{
   using HostMatrix = Matrix< Real, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< Real, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() });

   HostMatrix hostMatrix;
   try
   {
      hostMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + String(e.what()) );
      return;
   }

   const int nonzeros = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   if( allCpuTests )
   {
      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector );

      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef HAVE_CUDA
   CudaMatrix cudaMatrix;
   try
   {
      cudaMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
}

template< typename Real,
          typename InputMatrix,
          template< typename, typename, typename > class Matrix,
          typename TestReal = Real >
void
benchmarkSpMVCSRLight( BenchmarkType& benchmark,
                       const InputMatrix& inputMatrix,
                       const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                       const String& inputFileName,
                       bool allCpuTests,
                       bool verboseMR )
{
   using HostMatrix = Matrix< TestReal, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< TestReal, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() });

   HostMatrix hostMatrix;
   try
   {
      hostMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + String(e.what()) );
      return;
   }

   const int nonzeros = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   if( allCpuTests )
   {
      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector );

      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef HAVE_CUDA
   CudaMatrix cudaMatrix;
   try
   {
      cudaMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };

   {
      cudaMatrix.getSegments().getKernel().setThreadsMapping( Algorithms::Segments::CSRLightAutomaticThreads );
      String format = MatrixInfo< HostMatrix >::getFormat() + " Automatic";
      benchmark.setMetadataElement({ "format", format });

      SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
   };

   {
      cudaMatrix.getSegments().getKernel().setThreadsMapping( Algorithms::Segments::CSRLightAutomaticThreadsLightSpMV );
      String format = MatrixInfo< HostMatrix >::getFormat() + " Automatic Light";
      benchmark.setMetadataElement({ "format", format });

      SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
   };

   /*for( auto threadsPerRow : std::vector< int >{ 1, 2, 4, 8, 16, 32 } )
   {
      cudaMatrix.getSegments().getKernel().setThreadsPerSegment( threadsPerRow );
      String format = MatrixInfo< HostMatrix >::getFormat() + " " + convertToString( threadsPerRow );
      benchmark.setMetadataElement({ "format", format });

      SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
      benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
   }*/
 #endif
}


template< typename Real,
          typename InputMatrix,
          template< typename, typename, typename > class Matrix >
void
benchmarkBinarySpMV( BenchmarkType& benchmark,
                     const InputMatrix& inputMatrix,
                     const TNL::Containers::Vector< Real, Devices::Host, int >& csrResultVector,
                     const String& inputFileName,
                     bool allCpuTests,
                     bool verboseMR )
{
   using HostMatrix = Matrix< bool, TNL::Devices::Host, int >;
   using CudaMatrix = Matrix< bool, TNL::Devices::Cuda, int >;
   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;

   benchmark.setMetadataElement({ "format", MatrixInfo< HostMatrix >::getFormat() });

   HostMatrix hostMatrix;
   try
   {
      hostMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to convert the matrix to the target format:" + String(e.what()) );
      return;
   }

   const int nonzeros = hostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   /////
   // Benchmark SpMV on host
   //
   if( allCpuTests )
   {
      HostVector hostInVector( hostMatrix.getColumns() ), hostOutVector( hostMatrix.getRows() );

      auto resetHostVectors = [&]() {
         hostInVector = 1.0;
         hostOutVector = 0.0;
      };

      auto spmvHost = [&]() {
         hostMatrix.vectorProduct( hostInVector, hostOutVector );

      };
      SpmvBenchmarkResult< Real, Devices::Host, int > hostBenchmarkResults( csrResultVector, hostOutVector );
      benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvHost, hostBenchmarkResults );
   }

   /////
   // Benchmark SpMV on CUDA
   //
#ifdef HAVE_CUDA
   CudaMatrix cudaMatrix;
   try
   {
      cudaMatrix = inputMatrix;
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to copy the matrix on GPU: " + String(e.what()) );
      return;
   }

   CudaVector cudaInVector( hostMatrix.getColumns() ), cudaOutVector( hostMatrix.getRows() );

   auto resetCudaVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCuda = [&]() {
      cudaMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };
   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( csrResultVector, cudaOutVector );
   benchmark.time< Devices::Cuda >( resetCudaVectors, "GPU", spmvCuda, cudaBenchmarkResults );
 #endif
}

template< typename Real >
void
dispatchLegacy( BenchmarkType& benchmark,
                const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                const String& inputFileName,
                bool allCpuTests,
                bool verboseMR )
{
   using namespace Benchmarks::SpMV::ReferenceFormats;
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Scalar             >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Vector             >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light              >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light2             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light3             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light4             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light5             >( benchmark, hostOutVector, inputFileName, verboseMR );
   //benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Light6             >( benchmark, hostOutVector, inputFileName, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_Adaptive           >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_MultiVector        >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVLegacy< Real, SparseMatrixLegacy_CSR_LightWithoutAtomic >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVLegacy< Real, Legacy::Ellpack                           >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVLegacy< Real, SlicedEllpackAlias                        >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVLegacy< Real, Legacy::ChunkedEllpack                    >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVLegacy< Real, Legacy::BiEllpack                         >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
   // AdEllpack is broken
   //benchmarkSpMV< Real, Matrices::AdEllpack              >( benchmark, hostOutVector, inputFileName, verboseMR );
}

template< typename Real, typename HostMatrix >
void
dispatchBinary( BenchmarkType& benchmark,
                const HostMatrix& hostMatrix,
                const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                const String& inputFileName,
                bool allCpuTests,
                bool verboseMR )
{
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_CSR_Scalar              >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_CSR_Vector              >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVCSRLight< Real, HostMatrix, SparseMatrix_CSR_Light, bool >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_CSR_Adaptive            >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_Ellpack                 >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_SlicedEllpack           >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_ChunkedEllpack          >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, HostMatrix, SparseMatrix_BiEllpack               >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
}

template< typename Real >
void
dispatchSpMV( BenchmarkType& benchmark,
              const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
              const String& inputFileName,
              bool allCpuTests,
              bool verboseMR )
{
   using HostMatrixType = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host >;
   HostMatrixType hostMatrix;
   TNL::Matrices::MatrixReader< HostMatrixType >::readMtx( inputFileName, hostMatrix, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Scalar                   >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Vector                   >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   //benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Hybrid                   >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVCSRLight< Real, HostMatrixType, SparseMatrix_CSR_Light            >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_CSR_Adaptive                 >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_Ellpack                      >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_SlicedEllpack                >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_ChunkedEllpack               >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, HostMatrixType, SparseMatrix_BiEllpack                    >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   dispatchBinary< Real >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
#ifdef WITH_TNL_BENCHMARK_SPMV_SANDBOX_MATRIX
   benchmarkSpMV< Real, HostMatrixType, SparseSandboxMatrix                       >( benchmark, hostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
#endif
}

template< typename Real, typename SymmetricInputMatrix >
void
dispatchSymmetricBinary( BenchmarkType& benchmark,
                         const SymmetricInputMatrix& symmetricHostMatrix,
                         const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                         const String& inputFileName,
                         bool allCpuTests,
                         bool verboseMR )
{
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Scalar              >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Vector              >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   //benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Hybrid            >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVCSRLight< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Light, bool       >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Adaptive            >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_Ellpack                 >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_SlicedEllpack           >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_ChunkedEllpack          >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkBinarySpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_BiEllpack               >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
}

template< typename Real >
void
dispatchSymmetric( BenchmarkType& benchmark,
                   const TNL::Containers::Vector< Real, Devices::Host, int >& hostOutVector,
                   const String& inputFileName,
                   bool allCpuTests,
                   bool verboseMR )
{
   using SymmetricInputMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >;
   using InputMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >;
   SymmetricInputMatrix symmetricHostMatrix;
   try
   {
      TNL::Matrices::MatrixReader< SymmetricInputMatrix >::readMtx( inputFileName, symmetricHostMatrix, verboseMR );
   }
   catch(const std::exception& e)
   {
      benchmark.addErrorMessage( "Unable to read the symmetric matrix: " + String(e.what()) );
      return;
   }
   InputMatrix hostMatrix;
   TNL::Matrices::MatrixReader< InputMatrix >::readMtx( inputFileName, hostMatrix, verboseMR );
   // TODO: Comparison of symmetric and general matrix does not work yet.
   //if( hostMatrix != symmetricHostMatrix )
   //{
   //   std::cerr << "ERROR: Symmetric matrices do not match !!!" << std::endl;
   //}
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Scalar                    >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Vector                    >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   //benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Hybrid                   >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMVCSRLight< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Light             >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_CSR_Adaptive                  >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_Ellpack                       >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_SlicedEllpack                 >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_ChunkedEllpack                >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   benchmarkSpMV< Real, SymmetricInputMatrix, SymmetricSparseMatrix_BiEllpack                     >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
   dispatchSymmetricBinary< Real >( benchmark, symmetricHostMatrix, hostOutVector, inputFileName, allCpuTests, verboseMR );
}

template< typename Real = double,
          typename Index = int >
void
benchmarkSpmv( BenchmarkType& benchmark,
               const String& inputFileName,
               const Config::ParameterContainer& parameters,
               bool verboseMR )
{
   // The following is another workaround because of a bug in nvcc versions 10 and 11.
   // If we use the current matrix formats, not the legacy ones, we get
   // ' error: redefinition of ‘void TNL::Algorithms::__wrapper__device_stub_CudaReductionKernel...'
   // It seems that there is a problem with lambda functions identification when we create
   // two instances of TNL::Matrices::SparseMatrix. The second one comes from calling of
   // `benchmarkSpMV< Real, SparseMatrix_CSR_Scalar >( benchmark, hostOutVector, inputFileName, verboseMR );`
   // and simillar later in this function.
#define USE_LEGACY_FORMATS
#ifdef USE_LEGACY_FORMATS
   // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
   using CSRHostMatrix = SpMV::ReferenceFormats::Legacy::CSR< Real, Devices::Host, int >;
   using CSRCudaMatrix = SpMV::ReferenceFormats::Legacy::CSR< Real, Devices::Cuda, int >;
   using CusparseMatrix = TNL::CusparseCSRLegacy< Real >;
   using LightSpMVCSRHostMatrix = SpMV::ReferenceFormats::Legacy::CSR< Real, Devices::Host, uint32_t >;
#else
   // Here we use 'int' instead of 'Index' because of compatibility with cusparse.
   using CSRHostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, int >;
   using CSRCudaMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Cuda, int >;
   using CusparseMatrix = TNL::CusparseCSR< Real >;
#endif

   using HostVector = Containers::Vector< Real, Devices::Host, int >;
   using CudaVector = Containers::Vector< Real, Devices::Cuda, int >;
   using BinaryHostVector = Containers::Vector< int, Devices::Host, int >;

   CSRHostMatrix csrHostMatrix;

   ////
   // Set-up benchmark datasize
   //
   MatrixReader< CSRHostMatrix >::readMtx( inputFileName, csrHostMatrix, verboseMR );
   const int nonzeros = csrHostMatrix.getNonzeroElementsCount();
   const double datasetSize = (double) nonzeros * ( 2 * sizeof( Real ) + sizeof( int ) ) / oneGB;
   benchmark.setDatasetSize( datasetSize );

   ////
   // Perform benchmark on host with CSR as a reference CPU format
   //
   benchmark.setMetadataColumns({
      { "matrix name", inputFileName },
      { "precision", getType< Real >() },
      { "rows", convertToString( csrHostMatrix.getRows() ) },
      { "columns", convertToString( csrHostMatrix.getColumns() ) },
      { "nonzeros", convertToString( nonzeros ) },
      // NOTE: this can be easily calculated with Pandas based on the other metadata
      //{ "nonzeros per row", convertToString( ( double ) nonzeros / ( double ) csrHostMatrix.getRows() ) },
   });
   benchmark.setMetadataWidths({
      { "matrix name", 32 },
      { "format", 46 },
   });

   HostVector hostInVector( csrHostMatrix.getRows() ), hostOutVector( csrHostMatrix.getRows() );

   auto resetHostVectors = [&]() {
      hostInVector = 1.0;
      hostOutVector = 0.0;
   };

   auto spmvCSRHost = [&]() {
       csrHostMatrix.vectorProduct( hostInVector, hostOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > csrBenchmarkResults( hostOutVector, hostOutVector );
   benchmark.setMetadataElement({ "format", "CSR" });
   benchmark.time< Devices::Host >( resetHostVectors, "CPU", spmvCSRHost, csrBenchmarkResults );

#ifdef HAVE_PETSC
   Mat petscMatrix;
   Containers::Vector< PetscInt, Devices::Host, PetscInt > petscRowPointers( csrHostMatrix.getRowPointers() );
   Containers::Vector< PetscInt, Devices::Host, PetscInt > petscColumns( csrHostMatrix.getColumnIndexes() );
   Containers::Vector< PetscScalar, Devices::Host, PetscInt > petscValues( csrHostMatrix.getValues() );
   MatCreateSeqAIJWithArrays( PETSC_COMM_WORLD, //PETSC_COMM_SELF,
                              csrHostMatrix.getRows(),
                              csrHostMatrix.getColumns(),
                              petscRowPointers.getData(),
                              petscColumns.getData(),
                              petscValues.getData(),
                              &petscMatrix );
   Vec inVector, outVector;
   VecCreateSeq( PETSC_COMM_WORLD, csrHostMatrix.getColumns(), &inVector );
   VecCreateSeq( PETSC_COMM_WORLD, csrHostMatrix.getRows(), &outVector );

   auto resetPetscVectors = [&]() {
      VecSet( inVector, 1.0 );
      VecSet( outVector, 0.0 );
   };

   auto petscSpmvCSRHost = [&]() {
      MatMult( petscMatrix, inVector, outVector );
   };

   SpmvBenchmarkResult< Real, Devices::Host, int > petscBenchmarkResults( hostOutVector, hostOutVector );
   benchmark.setMetadataElement({ "format", "Petsc" });
   benchmark.time< Devices::Host >( resetPetscVectors, "CPU", petscSpmvCSRHost, petscBenchmarkResults );
#endif


#ifdef HAVE_CUDA
   ////
   // Perform benchmark on CUDA device with cuSparse as a reference GPU format
   //
   cusparseHandle_t cusparseHandle;
   cusparseCreate( &cusparseHandle );

   CSRCudaMatrix csrCudaMatrix;
   csrCudaMatrix = csrHostMatrix;

   CusparseMatrix cusparseMatrix;
   cusparseMatrix.init( csrCudaMatrix, &cusparseHandle );

   CudaVector cudaInVector( csrCudaMatrix.getColumns() ), cudaOutVector( csrCudaMatrix.getRows() );

   auto resetCusparseVectors = [&]() {
      cudaInVector = 1.0;
      cudaOutVector = 0.0;
   };

   auto spmvCusparse = [&]() {
       cusparseMatrix.vectorProduct( cudaInVector, cudaOutVector );
   };

   SpmvBenchmarkResult< Real, Devices::Cuda, int > cudaBenchmarkResults( hostOutVector, cudaOutVector );
   benchmark.setMetadataElement({ "format", "cusparse" });
   benchmark.time< Devices::Cuda >( resetCusparseVectors, "GPU", spmvCusparse, cudaBenchmarkResults );

#ifdef HAVE_CSR5
   ////
   // Perform benchmark on CUDA device with CSR5 as a reference GPU format
   //
   CudaVector cudaOutVector2( cudaOutVector );
   CSR5Benchmark::CSR5Benchmark< CSRCudaMatrix > csr5Benchmark( csrCudaMatrix, cudaInVector, cudaOutVector );

   auto csr5SpMV = [&]() {
       csr5Benchmark.vectorProduct();
   };

   benchmark.setMetadataElement({ "format", "CSR5" });
   benchmark.time< Devices::Cuda >( resetCusparseVectors, "GPU", csr5SpMV, cudaBenchmarkResults );
   std::cerr << "CSR5 error = " << max( abs( cudaOutVector - cudaOutVector2 ) ) << std::endl;
   csrCudaMatrix.reset();
#endif

   ////
   // Perform benchmark on CUDA device with LightSpMV as a reference GPU format
   //
   LightSpMVCSRHostMatrix lightSpMVCSRHostMatrix;
   lightSpMVCSRHostMatrix = csrHostMatrix;
   LightSpMVBenchmark< Real > lightSpMVBenchmark( lightSpMVCSRHostMatrix, LightSpMVBenchmarkKernelVector );
   auto resetLightSpMVVectors = [&]() {
      lightSpMVBenchmark.resetVectors();
   };

   auto spmvLightSpMV = [&]() {
       lightSpMVBenchmark.vectorProduct();
   };
   benchmark.setMetadataElement({ "format", "LightSpMV Vector" });
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "GPU", spmvLightSpMV, cudaBenchmarkResults );

   lightSpMVBenchmark.setKernelType( LightSpMVBenchmarkKernelWarp );
   benchmark.setMetadataElement({ "format", "LightSpMV Warp" });
   benchmark.time< Devices::Cuda >( resetLightSpMVVectors, "GPU", spmvLightSpMV, cudaBenchmarkResults );
#endif
   csrHostMatrix.reset();

   bool allCpuTests = parameters.getParameter< bool >( "with-all-cpu-tests" );
   /////
   // Benchmarking of TNL legacy formats
   //
   if( parameters.getParameter< bool >("with-legacy-matrices") )
      dispatchLegacy< Real >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );

   /////
   // Benchmarking TNL formats
   //
   dispatchSpMV< Real >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );

   /////
   // Benchmarking symmetric sparse matrices
   //
   if( parameters.getParameter< bool >("with-symmetric-matrices") )
      dispatchSymmetric< Real >( benchmark, hostOutVector, inputFileName, allCpuTests, verboseMR );
}

// =============== EXPLICIT TEMPLATE INSTANTIATIONS ===============
// The explicit template declarations (extern ...) are converted to definitions
// in separate source files using the eti.py script. The developer should call
// this script whenever the declarations are changed and commit the generated
// definitions in the git repository.
//
// IMPORTANT:
// - Each template instantiation must be written on exactly one line (the code
//   generator script (spmv.py) does not support parsing multiple lines).
// - Make sure that all "dispatch*" functions that are called above are
//   instantiated below.
// - Also make sure that all functions that are explicitly instantiated below
//   are actually used.
// - Explicit template instantiations cannot be guarded by #ifdef (the code
//   generator script (spmv.py) does not support parsing macros).
// - For optimum compilation performance, the explicitly instantiated functions
//   should be as independent as possible. The compilation of each explicit
//   instantiation should take about the same time so that the work load in a
//   parallel build is balanced. Functions that are not instantiated explicitly
//   will be compiled in the main unit that is compiled serially.

extern template void dispatchLegacy< float >( BenchmarkType&, const Containers::Vector< float, Devices::Host, int >&, const String&, bool, bool );
extern template void dispatchLegacy< double >( BenchmarkType&, const Containers::Vector< double, Devices::Host, int >&, const String&, bool, bool );

extern template void dispatchBinary< float >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< float, Devices::Host, int >&, const String&, bool, bool );
extern template void dispatchBinary< double >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< double, Devices::Host, int >&, const String&, bool, bool );

extern template void dispatchSpMV< float >( BenchmarkType&, const Containers::Vector< float, Devices::Host, int >&, const String&, bool, bool );
extern template void dispatchSpMV< double >( BenchmarkType&, const Containers::Vector< double, Devices::Host, int >&, const String&, bool, bool );

extern template void dispatchSymmetric< float >( BenchmarkType&, const Containers::Vector< float, Devices::Host, int >&, const String&, bool, bool );
extern template void dispatchSymmetric< double >( BenchmarkType&, const Containers::Vector< double, Devices::Host, int >&, const String&, bool, bool );

extern template void dispatchSymmetricBinary< float >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< float, Devices::Host, int >&, const String&, bool, bool );
extern template void dispatchSymmetricBinary< double >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< double, Devices::Host, int >&, const String&, bool, bool );

      } // namespace SpMV
   } // namespace Benchmarks
} // namespace TNL
