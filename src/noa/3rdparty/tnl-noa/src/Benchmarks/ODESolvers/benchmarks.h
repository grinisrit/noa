// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Config/ParameterContainer.h>

#include <TNL/Benchmarks/Benchmarks.h>

#include <stdexcept>  // std::runtime_error

using namespace TNL;
using namespace TNL::Pointers;
using namespace TNL::Benchmarks;


template< typename Device >
const char*
getPerformer()
{
   if( std::is_same< Device, Devices::Cuda >::value )
      return "GPU";
   return "CPU";
}

template< typename Solver, typename VectorPointer >
void
benchmarkSolver( Benchmark<>& benchmark,
                 const Config::ParameterContainer& parameters,
                 VectorPointer& u )
{
   using VectorType = typename VectorPointer::ObjectType;
   using RealType = typename VectorType::RealType;
   using DeviceType = typename VectorType::DeviceType;
   using IndexType = typename VectorType::IndexType;
   //using ProblemType = typename Solver::ProblemType;

   // setup
   //ProblemType problem;
   Solver solver;
   solver.setup( parameters );
   //solver.setProblem( problem );
   //solver.setSolverMonitor( benchmark.getMonitor() );
   //solver.setStopTime( parameters.getParameter< double >( "final-time" ) );
   //solver.setTau( parameters.getParameter< double >( "time-step" ) );

   // reset function
   auto reset = [&]() {
      *u = 0.0;
   };

   // benchmark function
   auto compute = [&]() {
      //solver.solve( u );
   };

   // subclass BenchmarkResult to add extra columns to the benchmark
   // (iterations, preconditioned residue, true residue)
   /*struct MyBenchmarkResult : public BenchmarkResult
   {
      using HeaderElements = BenchmarkResult::HeaderElements;
      using RowElements = BenchmarkResult::RowElements;

      Solver< Matrix >& solver;
      const SharedPointer< Matrix >& matrix;
      const Vector& x;
      const Vector& b;

      MyBenchmarkResult( Solver< Matrix >& solver,
                         const SharedPointer< Matrix >& matrix,
                         const Vector& x,
                         const Vector& b )
      : solver(solver), matrix(matrix), x(x), b(b)
      {}

      virtual HeaderElements getTableHeader() const override
      {
         return HeaderElements({"time", "speedup", "converged", "iterations", "residue_precond", "residue_true"});
      }

      virtual RowElements getRowElements() const override
      {
         const bool converged = ! std::isnan(solver.getResidue()) && solver.getResidue() < solver.getConvergenceResidue();
         const long iterations = solver.getIterations();
         const double residue_precond = solver.getResidue();

         Vector r;
         r.setLike( x );
         matrix->vectorProduct( x, r );
         r.addVector( b, 1.0, -1.0 );
         const double residue_true = lpNorm( r.getView(), 2.0 ) / lpNorm( b.getView(), 2.0 );

         return RowElements({ time, speedup, (double) converged, (double) iterations,
                              residue_precond, residue_true });
      }
   };
   MyBenchmarkResult benchmarkResult( solver, matrix, x, b );*/

   benchmark.time< Devices::Host >( reset, getPerformer< DeviceType >(), compute ); //, benchmarkResult );
}

