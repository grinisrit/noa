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
   using VectorType = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;

   // Parse the command line arguments
   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription config;
   config.addDelimiter( "General settings:" );
   config.addEntry< std::string >( "executor", "Ginkgo executor.", "omp" );
   config.addEntry< int >( "grid-size", "Number of grid points along the cube edge.", 33 );

   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   const int n = parameters.getParameter< int >( "grid-size" );;
   const std::string executor_string = parameters.getParameter< std::string >( "executor" );

   // Create the linear system in TNL
   VectorType x;
   VectorType b;
   auto A = getLambdaMatrix( b, n );
   using MatrixType = decltype( A );
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

   // Wrap the matrix for usage in Ginkgo - note that it is not convertible to Csr,
   // so it cannot be used with preconditioners
   auto gko_A = gko::share( TNL::Matrices::GinkgoOperator< MatrixType >::create( exec, A ) );

   // Wrap the vectors
   auto gko_b = TNL::Containers::GinkgoVector< ValueType, DeviceType >::create( exec, b.getView() );
   auto gko_x = TNL::Containers::GinkgoVector< ValueType, DeviceType >::create( exec, x.getView() );

   // Create the logger
   auto logger = gko::share( gko::log::Convergence< ValueType >::create() );

   // Create stopping criteria so we can add the logger
   // https://github.com/ginkgo-project/ginkgo/discussions/1099#discussioncomment-3439954
   auto iter_stop = gko::share(
               gko::stop::Iteration::build()
                  .with_max_iters( 1000 )
                  .on( exec ) );
   auto tol_stop = gko::share(
               gko::stop::ResidualNorm< ValueType >::build()
                  .with_baseline( gko::stop::mode::rhs_norm )
                  .with_reduction_factor( 1e-11 )
                  .on( exec ) );
   iter_stop->add_logger( logger );
   tol_stop->add_logger( logger );

   // Create the solver
   auto solver_factory =
         gko::solver::Cg< ValueType >::build()
            .with_criteria( iter_stop, tol_stop )
            .on( exec );
   auto solver = solver_factory->generate( gko_A );

   // Solve system
   solver->apply( lend( gko_b ), lend( gko_x ) );

   // Write the result
   std::cout << "converged: " << logger->has_converged() << "\n"
             << "number of iterations: " << logger->get_num_iterations() << "\n"
             << "residual norm: " << get_scalar_value< ValueType >( logger->get_residual_norm() ) << "\n"
             << "implicit residual norm: " << std::sqrt( get_scalar_value< ValueType >( logger->get_implicit_sq_resnorm() ) ) << "\n"
             << std::flush;
}
