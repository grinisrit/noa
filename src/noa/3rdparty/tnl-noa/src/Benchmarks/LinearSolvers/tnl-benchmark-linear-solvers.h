// Implemented by: Jakub Klinkovsky

#pragma once

#include <set>
#include <sstream>
#include <string>
#include <random>

#ifndef NDEBUG
#include <TNL/Debugging/FPE.h>
#endif

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Matrices/DistributedMatrix.h>
#include <TNL/Matrices/SparseOperations.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Solvers/Linear/Preconditioners/Diagonal.h>
#include <TNL/Solvers/Linear/Preconditioners/ILU0.h>
#include <TNL/Solvers/Linear/Preconditioners/ILUT.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/BICGStabL.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>

#include <TNL/Benchmarks/Benchmarks.h>
#include "../DistSpMV/ordering.h"
#include "benchmarks.h"

// FIXME: nvcc 8.0 fails when cusolverSp.h is included (works fine with clang):
// /opt/cuda/include/cuda_fp16.h(3068): error: more than one instance of overloaded function "isinf" matches the argument list:
//             function "isinf(float)"
//             function "std::isinf(float)"
//             argument types are: (float)
#if defined(HAVE_CUDA) && !defined(__NVCC__)
   #include "CuSolverWrapper.h"
   #define HAVE_CUSOLVER
#endif

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

template< typename _Device, typename _Index, typename _IndexAlocator >
using SegmentsType = TNL::Algorithms::Segments::SlicedEllpack< _Device, _Index, _IndexAlocator >;

using namespace TNL;
using namespace TNL::Benchmarks;


static const std::set< std::string > valid_solvers = {
   "gmres",
   "tfqmr",
   "bicgstab",
   "bicgstab-ell",
};

static const std::set< std::string > valid_gmres_variants = {
   "CGS",
   "CGSR",
   "MGS",
   "MGSR",
   "CWY",
};

static const std::set< std::string > valid_preconditioners = {
   "jacobi",
   "ilu0",
   "ilut",
};

std::set< std::string >
parse_comma_list( const Config::ParameterContainer& parameters,
                  const char* parameter,
                  const std::set< std::string >& options )
{
   const String param = parameters.getParameter< String >( parameter );

   if( param == "all" )
      return options;

   std::stringstream ss( param.getString() );
   std::string s;
   std::set< std::string > set;

   while( std::getline( ss, s, ',' ) ) {
      if( ! options.count( s ) )
         throw std::logic_error( std::string("Invalid value in the comma-separated list for the parameter '")
                                 + parameter + "': '" + s + "'. The list contains: '" + param.getString() + "'." );

      set.insert( s );

      if( ss.peek() == ',' )
         ss.ignore();
   }

   return set;
}

// initialize all vector entries with a unioformly distributed random value from the interval [a, b]
template< typename Vector >
void set_random_vector( Vector& v, typename Vector::RealType a, typename Vector::RealType b )
{
   using RealType = typename Vector::RealType;
   using IndexType = typename Vector::IndexType;
   // random device will be used to obtain a seed for the random number engine
   std::random_device rd;
   // initialize the standard mersenne_twister_engine with rd() as the seed
   std::mt19937 gen(rd());
   // create uniform distribution
   std::uniform_real_distribution< RealType > dis(a, b);

   // create host vector
   typename Vector::template Self< RealType, Devices::Host > host_v;
   host_v.setSize( v.getSize() );

   // initialize the host vector
   auto kernel = [&] ( IndexType i )
   {
      host_v[ i ] = dis(gen);
   };
   Algorithms::ParallelFor< Devices::Host >::exec( (IndexType) 0, host_v.getSize(), kernel );

   // copy the data to the device vector
   v = host_v;
}

template< typename Matrix, typename Vector >
void
benchmarkIterativeSolvers( Benchmark<>& benchmark,
                           Config::ParameterContainer parameters,
                           const std::shared_ptr< Matrix >& matrixPointer,
                           const Vector& x0,
                           const Vector& b )
{
#ifdef HAVE_CUDA
   using CudaMatrix = typename Matrix::template Self< typename Matrix::RealType, Devices::Cuda >;
   using CudaVector = typename Vector::template Self< typename Vector::RealType, Devices::Cuda >;

   CudaVector cuda_x0, cuda_b;
   cuda_x0 = x0;
   cuda_b = b;

   auto cudaMatrixPointer = std::make_shared< CudaMatrix >();
   *cudaMatrixPointer = *matrixPointer;
#endif

   using namespace Solvers::Linear;
   using namespace Solvers::Linear::Preconditioners;

   const int ell_max = 2;
   const std::set< std::string > solvers = parse_comma_list( parameters, "solvers", valid_solvers );
   const std::set< std::string > gmresVariants = parse_comma_list( parameters, "gmres-variants", valid_gmres_variants );
   const std::set< std::string > preconditioners = parse_comma_list( parameters, "preconditioners", valid_preconditioners );
   const bool with_preconditioner_update = parameters.getParameter< bool >( "with-preconditioner-update" );

   if( preconditioners.count( "jacobi" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation("preconditioner update (Jacobi)");
         benchmarkPreconditionerUpdate< Diagonal >( benchmark, parameters, matrixPointer );
         #ifdef HAVE_CUDA
         benchmarkPreconditionerUpdate< Diagonal >( benchmark, parameters, cudaMatrixPointer );
         #endif
      }

      if( solvers.count( "gmres" ) ) {
         for( auto variant : gmresVariants ) {
            benchmark.setOperation(variant + "-GMRES (Jacobi)");
            parameters.template setParameter< String >( "gmres-variant", variant );
            benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation("TFQMR (Jacobi)");
         benchmarkSolver< TFQMR, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< TFQMR, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation("BiCGstab (Jacobi)");
         benchmarkSolver< BICGStab, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< BICGStab, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         benchmark.setOperation("BiCGstab (Jacobi)");
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation("BiCGstab(" + convertToString(ell) + ") (Jacobi)");
            benchmarkSolver< BICGStabL, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< BICGStabL, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }
   }


   if( preconditioners.count( "ilu0" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation("preconditioner update (ILU0)");
         benchmarkPreconditionerUpdate< ILU0 >( benchmark, parameters, matrixPointer );
         #ifdef HAVE_CUDA
         benchmarkPreconditionerUpdate< ILU0 >( benchmark, parameters, cudaMatrixPointer );
         #endif
      }

      if( solvers.count( "gmres" ) ) {
         for( auto variant : gmresVariants ) {
            benchmark.setOperation(variant + "-GMRES (ILU0)");
            parameters.template setParameter< String >( "gmres-variant", variant );
            benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation("TFQMR (ILU0)");
         benchmarkSolver< TFQMR, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< TFQMR, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation("BiCGstab (ILU0)");
         benchmarkSolver< BICGStab, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< BICGStab, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation("BiCGstab(" + convertToString(ell) + ") (ILU0)");
            benchmarkSolver< BICGStabL, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< BICGStabL, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }
   }


   if( preconditioners.count( "ilut" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation("preconditioner update (ILUT)");
         benchmarkPreconditionerUpdate< ILUT >( benchmark, parameters, matrixPointer );
         #ifdef HAVE_CUDA
         benchmarkPreconditionerUpdate< ILUT >( benchmark, parameters, cudaMatrixPointer );
         #endif
      }

      if( solvers.count( "gmres" ) ) {
         for( auto variant : gmresVariants ) {
            benchmark.setOperation(variant + "-GMRES (ILUT)");
            parameters.template setParameter< String >( "gmres-variant", variant );
            benchmarkSolver< GMRES, ILUT >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< GMRES, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation("TFQMR (ILUT)");
         benchmarkSolver< TFQMR, ILUT >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< TFQMR, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation("BiCGstab (ILUT)");
         benchmarkSolver< BICGStab, ILUT >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< BICGStab, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation("BiCGstab(" + convertToString(ell) + ") (ILUT)");
            benchmarkSolver< BICGStabL, ILUT >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< BICGStabL, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }
   }
}

template< typename MatrixType >
struct LinearSolversBenchmark
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   using Partitioner = Containers::Partitioner< IndexType >;
   using DistributedMatrix = Matrices::DistributedMatrix< MatrixType >;
   using DistributedVector = Containers::DistributedVector< RealType, DeviceType, IndexType >;
   using DistributedRowLengths = typename DistributedMatrix::RowsCapacitiesType;

   static bool
   run( Benchmark<>& benchmark,
        const Config::ParameterContainer& parameters )
   {
      const String file_matrix = parameters.getParameter< String >( "input-matrix" );
      const String file_dof = parameters.getParameter< String >( "input-dof" );
      const String file_rhs = parameters.getParameter< String >( "input-rhs" );

      auto matrixPointer = std::make_shared< MatrixType >();
      VectorType x0, b;

      // load the matrix
      if( file_matrix.endsWith( ".mtx" ) ) {
         Matrices::MatrixReader< MatrixType > reader;
         reader.readMtx( file_matrix, *matrixPointer );
      }
      else {
         matrixPointer->load( file_matrix );
      }

      // load the vectors
      if( file_dof && file_rhs ) {
         File( file_dof, std::ios_base::in ) >> x0;
         File( file_rhs, std::ios_base::in ) >> b;
      }
      else {
         // set x0 := 0
         x0.setSize( matrixPointer->getColumns() );
         x0 = 0;

         // generate random vector x
         VectorType x;
         x.setSize( matrixPointer->getColumns() );
         set_random_vector( x, 1e2, 1e3 );

         // set b := A*x
         b.setSize( matrixPointer->getRows() );
         matrixPointer->vectorProduct( x, b );
      }

      typename MatrixType::RowsCapacitiesType rowLengths;
      matrixPointer->getCompressedRowLengths( rowLengths );
      const IndexType maxRowLength = max( rowLengths );

      const String title = (TNL::MPI::GetSize() > 1) ? "Distributed linear solvers" : "Linear solvers";
      std::cout << "\n== " << title << " ==\n" << std::endl;

      benchmark.setMetadataColumns( Benchmark<>::MetadataColumns({
         { "matrix name", parameters.getParameter< String >( "name" ) },
         // TODO: strip the device
//         { "matrix type", matrixPointer->getType() },
         { "rows", convertToString( matrixPointer->getRows() ) },
         { "columns", convertToString( matrixPointer->getColumns() ) },
         // FIXME: getMaxRowLengths() returns 0 for matrices loaded from file
//         { "max elements per row", matrixPointer->getMaxRowLength() },
         { "max elements per row", convertToString( maxRowLength ) },
      } ));

      const bool reorder = parameters.getParameter< bool >( "reorder-dofs" );
      if( reorder ) {
         using PermutationVector = Containers::Vector< IndexType, DeviceType, IndexType >;
         PermutationVector perm, iperm;
         getTrivialOrdering( *matrixPointer, perm, iperm );
         auto matrix_perm = std::make_shared< MatrixType >();
         VectorType x0_perm, b_perm;
         x0_perm.setLike( x0 );
         b_perm.setLike( b );
         Matrices::reorderSparseMatrix( *matrixPointer, *matrix_perm, perm, iperm );
         Matrices::reorderArray( x0, x0_perm, perm );
         Matrices::reorderArray( b, b_perm, perm );
         if( TNL::MPI::GetSize() > 1 )
            runDistributed( benchmark, parameters, matrix_perm, x0_perm, b_perm );
         else
            runNonDistributed( benchmark, parameters, matrix_perm, x0_perm, b_perm );
      }
      else {
         if( TNL::MPI::GetSize() > 1 )
            runDistributed( benchmark, parameters, matrixPointer, x0, b );
         else
            runNonDistributed( benchmark, parameters, matrixPointer, x0, b );
      }

      return true;
   }

   static void
   runDistributed( Benchmark<>& benchmark,
                   const Config::ParameterContainer& parameters,
                   const std::shared_ptr< MatrixType >& matrixPointer,
                   const VectorType& x0,
                   const VectorType& b )
   {
      // set up the distributed matrix
      const auto communicator = MPI_COMM_WORLD;
      const auto localRange = Partitioner::splitRange( matrixPointer->getRows(), communicator );
      auto distMatrixPointer = std::make_shared< DistributedMatrix >( localRange, matrixPointer->getRows(), matrixPointer->getColumns(), communicator );
      DistributedVector dist_x0( localRange, 0, matrixPointer->getRows(), communicator );
      DistributedVector dist_b( localRange, 0, matrixPointer->getRows(), communicator );

      // copy the row capacities from the global matrix to the distributed matrix
      DistributedRowLengths distributedRowLengths( localRange, 0, matrixPointer->getRows(), communicator );
      for( IndexType i = 0; i < distMatrixPointer->getLocalMatrix().getRows(); i++ ) {
         const auto gi = distMatrixPointer->getLocalRowRange().getGlobalIndex( i );
         distributedRowLengths[ gi ] = matrixPointer->getRowCapacity( gi );
      }
      distMatrixPointer->setRowCapacities( distributedRowLengths );

      // copy data from the global matrix/vector into the distributed matrix/vector
      for( IndexType i = 0; i < distMatrixPointer->getLocalMatrix().getRows(); i++ ) {
         const auto gi = distMatrixPointer->getLocalRowRange().getGlobalIndex( i );
         dist_x0[ gi ] = x0[ gi ];
         dist_b[ gi ] = b[ gi ];

//         const IndexType rowLength = matrixPointer->getRowLength( i );
//         IndexType columns[ rowLength ];
//         RealType values[ rowLength ];
//         matrixPointer->getRowFast( gi, columns, values );
//         distMatrixPointer->setRowFast( gi, columns, values, rowLength );
         const auto global_row = matrixPointer->getRow( gi );
         auto local_row = distMatrixPointer->getRow( gi );
         for( IndexType j = 0; j < global_row.getSize(); j++ )
            local_row.setElement( j, global_row.getColumnIndex( j ), global_row.getValue( j ) );
      }

      std::cout << "Iterative solvers:" << std::endl;
      benchmarkIterativeSolvers( benchmark, parameters, distMatrixPointer, dist_x0, dist_b );
   }

   static void
   runNonDistributed( Benchmark<>& benchmark,
                      const Config::ParameterContainer& parameters,
                      const std::shared_ptr< MatrixType >& matrixPointer,
                      const VectorType& x0,
                      const VectorType& b )
   {
      // direct solvers
      if( parameters.getParameter< bool >( "with-direct" ) ) {
         using CSR = TNL::Matrices::SparseMatrix< RealType,
                                                  DeviceType,
                                                  IndexType,
                                                  TNL::Matrices::GeneralMatrix,
                                                  Algorithms::Segments::CSRDefault
                                                >;
         auto matrixCopy = std::make_shared< CSR >();
         Matrices::copySparseMatrix( *matrixCopy, *matrixPointer );

#ifdef HAVE_UMFPACK
         std::cout << "UMFPACK wrapper:" << std::endl;
         using UmfpackSolver = Solvers::Linear::UmfpackWrapper< CSR >;
         using Preconditioner = Solvers::Linear::Preconditioners::Preconditioner< CSR >;
         benchmarkSolver< UmfpackSolver, Preconditioner >( parameters, matrixCopy, x0, b );
#endif

#ifdef HAVE_ARMADILLO
         std::cout << "Armadillo wrapper (which wraps SuperLU):" << std::endl;
         benchmarkArmadillo( parameters, matrixCopy, x0, b );
#endif
      }

      std::cout << "Iterative solvers:" << std::endl;
      benchmarkIterativeSolvers( benchmark, parameters, matrixPointer, x0, b );

#ifdef HAVE_CUSOLVER
      std::cout << "CuSOLVER:" << std::endl;
      {
         using CSR = TNL::Matrices::SparseMatrix< RealType,
                                                  DeviceType,
                                                  IndexType,
                                                  TNL::Matrices::GeneralMatrix,
                                                  Algorithms::Segments::CSR
                                                >;
         auto matrixCopy = std::make_shared< CSR >();
         Matrices::copySparseMatrix( *matrixCopy, *matrixPointer );

         using CudaCSR = TNL::Matrices::SparseMatrix< RealType,
                                                      Devices::Cuda,
                                                      IndexType,
                                                      TNL::Matrices::GeneralMatrix,
                                                      Algorithms::Segments::CSR
                                                    >;
         using CudaVector = typename VectorType::template Self< RealType, Devices::Cuda >;
         auto cuda_matrixCopy = std::make_shared< CudaCSR >();
         *cuda_matrixCopy = *matrixCopy;
         CudaVector cuda_x0, cuda_b;
         cuda_x0.setLike( x0 );
         cuda_b.setLike( b );
         cuda_x0 = x0;
         cuda_b = b;

         using namespace Solvers::Linear;
         using namespace Solvers::Linear::Preconditioners;
         benchmarkSolver< CuSolverWrapper, Preconditioner >( benchmark, parameters, cuda_matrixCopy, cuda_x0, cuda_b );
      }
#endif
   }
};

void
configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-linear-solvers.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addRequiredEntry< String >( "input-matrix", "File name of the input matrix (in binary TNL format or textual MTX format)." );
   config.addEntry< String >( "input-dof", "File name of the input DOF vector (in binary TNL format).", "" );
   config.addEntry< String >( "input-rhs", "File name of the input right-hand-side vector (in binary TNL format).", "" );
   config.addEntry< String >( "name", "Name of the matrix in the benchmark.", "" );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< bool >( "reorder-dofs", "Reorder matrix entries corresponding to the same DOF together.", false );
   config.addEntry< bool >( "with-direct", "Includes the 3rd party direct solvers in the benchmark.", false );
   config.addEntry< String >( "solvers", "Comma-separated list of solvers to run benchmarks for. Options: gmres, tfqmr, bicgstab, bicgstab-ell.", "all" );
   config.addEntry< String >( "gmres-variants", "Comma-separated list of GMRES variants to run benchmarks for. Options: CGS, CGSR, MGS, MGSR, CWY.", "all" );
   config.addEntry< String >( "preconditioners", "Comma-separated list of preconditioners to run benchmarks for. Options: jacobi, ilu0, ilut.", "all" );
   config.addEntry< bool >( "with-preconditioner-update", "Run benchmark for the preconditioner update.", true );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
   #ifdef HAVE_CUDA
   config.addEntryEnum( "cuda" );
   #endif

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );

   config.addDelimiter( "Linear solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   using Matrix = Matrices::SparseMatrix< double >;
   using GMRES = Solvers::Linear::GMRES< Matrix >;
   GMRES::configSetup( config );
   using BiCGstabL = Solvers::Linear::BICGStabL< Matrix >;
   BiCGstabL::configSetup( config );
   using ILUT = Solvers::Linear::Preconditioners::ILUT< Matrix >;
   ILUT::configSetup( config );
}

int
main( int argc, char* argv[] )
{
#ifndef NDEBUG
   Debugging::trackFloatingPointExceptions();
#endif

   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   TNL::MPI::ScopedInitializer mpi(argc, argv);
   const int rank = TNL::MPI::GetRank();

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) ||
       ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = (rank == 0) ? parameters.getParameter< int >( "verbose" ) : 0;

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile;
   if( rank == 0 )
      logFile.open( logFileName, mode );

   // init benchmark and set parameters
   Benchmark<> benchmark( logFile, loops, verbose );

   // write global metadata into a separate file
   std::map< std::string, std::string > metadata = getHardwareMetadata();
   writeMapAsJson( metadata, logFileName, ".metadata.json" );

   // TODO: implement resolveMatrixType
//   return ! Matrices::resolveMatrixType< MainConfig,
//                                         Devices::Host,
//                                         LinearSolversBenchmark >( benchmark, parameters );
   using MatrixType = TNL::Matrices::SparseMatrix< double,
                                                   Devices::Host,
                                                   int,
                                                   TNL::Matrices::GeneralMatrix,
                                                   SegmentsType
                                                 >;
   return ! LinearSolversBenchmark< MatrixType >::run( benchmark, parameters );
}
