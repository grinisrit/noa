#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include "HeatEquationBenchmarkProblem.h"
#include "BenchmarkLaplace.h"
#include "DirichletBoundaryConditions.h"
#include "HeatEquationBenchmarkRhs.h"
#include "HeatEquationBenchmarkBuildConfigTag.h"

using BuildConfig = HeatEquationBenchmarkBuildConfigTag;

/****
 * Uncoment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * esppecially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class HeatEquationBenchmarkConfig
{
   public:
      static void configSetup( Config::ConfigDescription & config )
      {
         config.addDelimiter( "Heat Equation Benchmark settings:" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< String >( "cuda-kernel-type", "CUDA kernel type.", "pure-c" );
            config.addEntryEnum< String >( "pure-c" );
            config.addEntryEnum< String >( "templated" );
            config.addEntryEnum< String >( "templated-compact" );
            config.addEntryEnum< String >( "tunning" );

         /****
          * Add definition of your solver command line arguments.
          */

      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class HeatEquationBenchmarkSetter
{
   public:
      using RealType = Real;
      using DeviceType = Device;
      using IndexType = Index;

      static bool run( const Config::ParameterContainer & parameters )
      {
          enum { Dimension = MeshType::getMeshDimension() };
          using ApproximateOperator = BenchmarkLaplace< MeshType, Real, Index >;
          using RightHandSide = HeatEquationBenchmarkRhs< MeshType, Real >;
          using Point = Containers::StaticVector< MeshType::getMeshDimension(), Real >;

          /****
           * Resolve the template arguments of your solver here.
           * The following code is for the Dirichlet and the Neumann boundary conditions.
           * Both can be constant or defined as descrete values of Vector.
           */
          String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
          if( parameters.checkParameter( "boundary-conditions-constant" ) )
          {
             using Constant = Functions::Analytic::Constant< Dimension, Real >;
             if( boundaryConditionsType == "dirichlet" )
             {
                using BoundaryConditions =
                   Operators::DirichletBoundaryConditions< MeshType, Constant, MeshType::getMeshDimension(), Real, Index >;
                using Problem =
                   HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator >;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
             /*typedef Operators::NeumannBoundaryConditions< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );*/
          }
          /*typedef Functions::MeshFunction< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef Operators::DirichletBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimension(), Real, Index > BoundaryConditions;
             typedef HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          typedef Operators::NeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
          typedef HeatEquationBenchmarkProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
          SolverStarter solverStarter;
          return solverStarter.template run< Problem >( parameters );*/
          return false;
      }

};

int main( int argc, char* argv[] )
{
   Solvers::Solver< HeatEquationBenchmarkSetter, HeatEquationBenchmarkConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

