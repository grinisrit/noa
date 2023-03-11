#include <TNL/Timer.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/GinkgoVector.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/GinkgoOperator.h>

#include "PoissonEquation2D.h"
#include "utils.h"

int
main( int argc, char* argv[] )
{
   using ValueType = double;
#ifdef __CUDACC__
   using DeviceType = TNL::Devices::Cuda;
#else
   using DeviceType = TNL::Devices::Host;
#endif
   using IndexType = int;
   using MatrixType = TNL::Matrices::SparseMatrix< ValueType, DeviceType, IndexType >;
   using VectorType = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;

   // Parse the command line arguments
   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription config;
   config.addDelimiter( "General settings:" );
   config.addEntry< std::string >( "executor", "Ginkgo executor.", "omp" );
   config.addEntry< std::string >( "preconditioner", "Preconditioning algorithm.", "Jacobi" );
   config.addEntryEnum( "Jacobi" );
   config.addEntryEnum( "ParILU" );
   config.addEntryEnum( "AMGX" );
   config.addEntry< int >( "grid-size", "Number of grid points along the cube edge.", 33 );

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   const int n = parameters.getParameter< int >( "grid-size" );;
   const std::string executor_string = parameters.getParameter< std::string >( "executor" );
   const std::string preconditioner = parameters.getParameter< std::string >( "preconditioner" );

   // Create the linear system in TNL
   MatrixType A;
   VectorType x;
   VectorType b;
   generateStencilMatrix( A, b, n );
   // Set initial guess
   x.setLike( b );
   x.setValue( 0 );

   // Create Ginkgo executor
   std::shared_ptr< gko::Executor > exec = nullptr;
   if( executor_string == "omp" )
      exec = gko::OmpExecutor::create();
   else if( executor_string == "cuda" )
      // NOTE: false here disables device reset in the executor's destructor
      exec = gko::CudaExecutor::create( 0, gko::OmpExecutor::create(), false );
   else if( executor_string == "hip" )
      // NOTE: false here disables device reset in the executor's destructor
      exec = gko::HipExecutor::create( 0, gko::OmpExecutor::create(), false );
   else if( executor_string == "dpcpp" )
      exec = gko::DpcppExecutor::create( 0, gko::OmpExecutor::create() );
   else if( executor_string == "reference" )
      exec = gko::ReferenceExecutor::create();
   else
      throw std::logic_error( "invalid executor string: " + executor_string );

   // Create a Ginkgo Csr view
   auto gko_A = gko::share( TNL::Matrices::getGinkgoMatrixCsrView( exec, A ) );

   // Wrap the vectors
   auto gko_b = TNL::Containers::GinkgoVector< ValueType, DeviceType >::create( exec, b.getView() );
   auto gko_x = TNL::Containers::GinkgoVector< ValueType, DeviceType >::create( exec, x.getView() );

   // Create the logger
   auto logger = gko::share( gko::log::Convergence< ValueType >::create() );

   // Create stopping criteria so we can add the logger
   // https://github.com/ginkgo-project/ginkgo/discussions/1099#discussioncomment-3439954
   auto iter_stop = gko::share(
               gko::stop::Iteration::build()
                  .with_max_iters( 2000 )
                  .on( exec ) );
   auto tol_stop = gko::share(
               gko::stop::ResidualNorm< ValueType >::build()
                  .with_baseline( gko::stop::mode::rhs_norm )
                  .with_reduction_factor( 1e-7 )
                  .on( exec ) );
   iter_stop->add_logger( logger );
   tol_stop->add_logger( logger );

   std::shared_ptr< gko::LinOp > solver = nullptr;

   if( preconditioner == "Jacobi" ) {
      // Create the solver
      auto solver_factory =
            gko::solver::Cg< ValueType >::build()
               .with_preconditioner(
                  gko::preconditioner::Jacobi< ValueType, IndexType >::build()
                     .on( exec ) )
               .with_criteria( iter_stop, tol_stop )
               .on( exec );
      solver = solver_factory->generate( gko_A );
   }
   else if( preconditioner == "ParILU" ) {
      // Generate incomplete factors using ParILU
      auto fact_factory = gko::share(
            gko::factorization::ParIlu< ValueType, IndexType >::build()
               .on( exec )
      );

      // Generate an ILU preconditioner factory by setting lower and upper
      // triangular solver - in this case incomplete sparse approximate inverse
      const int sparsity_power = 2; // TODO: parametrize
      auto ilu_pre_factory = gko::share(
            gko::preconditioner::Ilu< gko::preconditioner::LowerIsai< ValueType, IndexType >,
                                      gko::preconditioner::UpperIsai< ValueType, IndexType > >::build()
               .with_factorization_factory( fact_factory )
               .with_l_solver_factory(
                  gko::preconditioner::LowerIsai< ValueType, IndexType >::build()
                     .with_sparsity_power( sparsity_power )
                     .on( exec )
                  )
               .with_u_solver_factory(
                  gko::preconditioner::UpperIsai< ValueType, IndexType >::build()
                     .with_sparsity_power( sparsity_power )
                     .on( exec )
                  )
               .on( exec )
      );

      // Create the solver
      auto solver_factory =
            gko::solver::Cg< ValueType >::build()
               .with_preconditioner( ilu_pre_factory )
               .with_criteria( iter_stop, tol_stop )
               .on( exec );
      solver = solver_factory->generate( gko_A );
   }
   else if( preconditioner == "AMGX" ) {
      // Create smoother factory (ir with bj)
      auto inner_solver_gen = gko::share(
            gko::preconditioner::Jacobi< ValueType, IndexType >::build()
               .with_max_block_size( 1 )
               .on( exec )
      );
      auto smoother_gen = gko::share(
            gko::solver::Ir< ValueType >::build()
               .with_solver( inner_solver_gen )
               .with_relaxation_factor( 0.9 )
               .with_criteria(
                   gko::stop::Iteration::build().with_max_iters( 2 ).on( exec ) )
               .on( exec )
      );
      // Create MultigridLevel factory
      auto mg_level_gen = gko::share(
            gko::multigrid::Pgm< ValueType, IndexType >::build()
               .with_deterministic( true )
               .on( exec )
      );
      // Create CoarsestSolver factory
      auto coarsest_gen = gko::share(
            gko::solver::Ir< ValueType >::build()
               .with_solver( inner_solver_gen )
               .with_relaxation_factor( 0.9 )
               .with_criteria(
                   gko::stop::Iteration::build().with_max_iters( 4 ).on( exec ) )
               .on( exec )
      );
      // Create multigrid factory
      auto multigrid_gen = gko::share(
            gko::solver::Multigrid::build()
               .with_max_levels( 9 )
               .with_min_coarse_rows( 10 )
               .with_pre_smoother( smoother_gen )
               .with_post_uses_pre( true )
               .with_mg_level( mg_level_gen )
               .with_coarsest_solver( coarsest_gen )
               .with_default_initial_guess(
                   gko::solver::initial_guess_mode::zero )
               .with_criteria(
                   gko::stop::Iteration::build().with_max_iters( 1 ).on( exec ) )
               .on( exec )
      );

      // Create the solver
      auto solver_factory =
            gko::solver::Cg< ValueType >::build()
               .with_preconditioner( multigrid_gen )
               .with_criteria( iter_stop, tol_stop )
               .on( exec );
      solver = solver_factory->generate( gko_A );
   }

   // Solve the system
   TNL::Timer t;
   t.start();
   solver->apply( lend( gko_b ), lend( gko_x ) );
   t.stop();

   // Write the result
   std::cout << "converged: " << logger->has_converged() << "\n"
             << "number of iterations: " << logger->get_num_iterations() << "\n"
             << "residual norm: " << get_scalar_value< ValueType >( logger->get_residual_norm() ) << "\n"
             << "implicit residual norm: " << std::sqrt( get_scalar_value< ValueType >( logger->get_implicit_sq_resnorm() ) ) << "\n"
             << "time to solution: " << t.getRealTime() << " seconds\n"
             << std::flush;
}
