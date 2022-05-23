#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Matrices/DistributedMatrix.h>

#include <TNL/Benchmarks/Benchmarks.h>

#ifdef HAVE_ARMADILLO
#include <armadillo>
#include <TNL/Matrices/CSR.h>
#endif

#include <stdexcept>  // std::runtime_error

using namespace TNL;
using namespace TNL::Benchmarks;


template< typename Device >
const char*
getPerformer()
{
   if( std::is_same< Device, Devices::Cuda >::value )
      return "GPU";
   return "CPU";
}

template< typename Matrix >
void barrier( const Matrix& matrix )
{
}

template< typename Matrix >
void barrier( const Matrices::DistributedMatrix< Matrix >& matrix )
{
   TNL::MPI::Barrier( matrix.getCommunicator() );
}

template< typename Device >
bool checkDevice( const Config::ParameterContainer& parameters )
{
   const String device = parameters.getParameter< String >( "devices" );
   if( device == "all" )
      return true;
   if( std::is_same< Device, Devices::Host >::value && device == "host" )
      return true;
   if( std::is_same< Device, Devices::Cuda >::value && device == "cuda" )
      return true;
   return false;
}

template< template<typename> class Preconditioner, typename Matrix >
void
benchmarkPreconditionerUpdate( Benchmark<>& benchmark,
                               const Config::ParameterContainer& parameters,
                               const std::shared_ptr< Matrix >& matrix )
{
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< typename Matrix::DeviceType >( parameters ) )
      return;

   barrier( matrix );
   const char* performer = getPerformer< typename Matrix::DeviceType >();
   Preconditioner< Matrix > preconditioner;
   preconditioner.setup( parameters );

   auto reset = []() {};
   auto compute = [&]() {
      preconditioner.update( matrix );
      barrier( matrix );
   };

   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute );
}

template< template<typename> class Solver, template<typename> class Preconditioner, typename Matrix, typename Vector >
void
benchmarkSolver( Benchmark<>& benchmark,
                 const Config::ParameterContainer& parameters,
                 const std::shared_ptr< Matrix >& matrix,
                 const Vector& x0,
                 const Vector& b )
{
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< typename Matrix::DeviceType >( parameters ) )
      return;

   barrier( matrix );
   const char* performer = getPerformer< typename Matrix::DeviceType >();

   // setup
   Solver< Matrix > solver;
   solver.setup( parameters );
   solver.setMatrix( matrix );

   // FIXME: getMonitor returns solver monitor specialized for double and int
   solver.setSolverMonitor( benchmark.getMonitor() );

   auto pre = std::make_shared< Preconditioner< Matrix > >();
   pre->setup( parameters );
   solver.setPreconditioner( pre );
   // preconditioner update may throw if it's not implemented for CUDA
   try {
      pre->update( matrix );
   }
   catch ( const std::runtime_error& ) {}

   Vector x;
   x.setLike( x0 );

   // reset function
   auto reset = [&]() {
      x = x0;
   };

   // benchmark function
   auto compute = [&]() {
      const bool converged = solver.solve( b, x );
      barrier( matrix );
      if( ! converged )
         throw std::runtime_error("solver did not converge");
   };

   // subclass BenchmarkResult to add extra columns to the benchmark
   // (iterations, preconditioned residue, true residue)
   struct MyBenchmarkResult : public BenchmarkResult
   {
      using HeaderElements = BenchmarkResult::HeaderElements;
      using RowElements = BenchmarkResult::RowElements;

      Solver< Matrix >& solver;
      const std::shared_ptr< Matrix >& matrix;
      const Vector& x;
      const Vector& b;

      MyBenchmarkResult( Solver< Matrix >& solver,
                         const std::shared_ptr< Matrix >& matrix,
                         const Vector& x,
                         const Vector& b )
      : solver(solver), matrix(matrix), x(x), b(b)
      {}

      virtual HeaderElements getTableHeader() const override
      {
         return HeaderElements({ "time", "stddev", "stddev/time", "speedup", "converged", "iterations", "residue_precond", "residue_true" });
      }

      virtual RowElements getRowElements() const override
      {
         const bool converged = ! std::isnan(solver.getResidue()) && solver.getResidue() < solver.getConvergenceResidue();
         const long iterations = solver.getIterations();
         const double residue_precond = solver.getResidue();

         Vector r;
         r.setLike( x );
         matrix->vectorProduct( x, r );
         r = b - r;
         const double residue_true = lpNorm( r, 2.0 ) / lpNorm( b, 2.0 );

         RowElements elements;
         elements << time << stddev << stddev/time;
         if( speedup != 0  )
            elements << speedup;
         else
            elements <<  "N/A";
         elements << ( converged ? "yes" : "no" ) << iterations << residue_precond << residue_true;
         return elements;
      }
   };
   MyBenchmarkResult benchmarkResult( solver, matrix, x, b );

   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute, benchmarkResult );
}

#ifdef HAVE_ARMADILLO
// TODO: make a TNL solver like UmfpackWrapper
template< typename Vector >
void
benchmarkArmadillo( const Config::ParameterContainer& parameters,
                    const std::shared_ptr< Matrices::CSR< double, Devices::Host, int > >& matrix,
                    const Vector& x0,
                    const Vector& b )
{
    // copy matrix into Armadillo's class
    // sp_mat is in CSC format
    arma::uvec _colptr( matrix->getRowPointers().getSize() );
    for( int i = 0; i < matrix->getRowPointers().getSize(); i++ )
        _colptr[ i ] = matrix->getRowPointers()[ i ];
    arma::uvec _rowind( matrix->getColumnIndexes().getSize() );
    for( int i = 0; i < matrix->getColumnIndexes().getSize(); i++ )
        _rowind[ i ] = matrix->getColumnIndexes()[ i ];
    arma::vec _values( matrix->getValues().getData(), matrix->getValues().getSize() );
    arma::sp_mat AT( _rowind,
                     _colptr,
                     _values,
                     matrix->getColumns(),
                     matrix->getRows() );
    arma::sp_mat A = AT.t();

    Vector x;
    x.setLike( x0 );

    // Armadillo vector using the same memory as x (with copy_aux_mem=false, strict=true)
    arma::vec arma_x( x.getData(), x.getSize(), false, true );
    arma::vec arma_b( b.getData(), b.getSize() );

    arma::superlu_opts settings;
//    settings.equilibrate = false;
    settings.equilibrate = true;
    settings.pivot_thresh = 1.0;
//    settings.permutation = arma::superlu_opts::COLAMD;
    settings.permutation = arma::superlu_opts::MMD_AT_PLUS_A;
//    settings.refine = arma::superlu_opts::REF_DOUBLE;
    settings.refine = arma::superlu_opts::REF_NONE;

    // reset function
    auto reset = [&]() {
        x = x0;
    };

    // benchmark function
    auto compute = [&]() {
        const bool converged = arma::spsolve( arma_x, A, arma_b, "superlu", settings );
        if( ! converged )
            throw std::runtime_error("solver did not converge");
    };

    const int loops = parameters.getParameter< int >( "loops" );
    double time = timeFunction( compute, reset, loops );

    arma::vec r = A * arma_x - arma_b;
//    std::cout << "Converged: " << (time > 0) << ", residue = " << arma::norm( r ) / arma::norm( arma_b ) << "                                                 " << std::endl;
//    std::cout << "Mean time: " << time / loops << " seconds." << std::endl;
    std::cout << "Converged: "   << std::setw( 5) << std::boolalpha << (time > 0) << "   "
              << "iterations = " << std::setw( 4) << "N/A" << "   "
              << "residue = "    << std::setw(10) << arma::norm( r ) / arma::norm( arma_b ) << "   "
              << "mean time = "  << std::setw( 9) << time / loops << " seconds." << std::endl;
}
#endif
