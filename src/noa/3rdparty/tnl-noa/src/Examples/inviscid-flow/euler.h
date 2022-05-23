#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include "eulerProblem.h"
#include "LaxFridrichs.h"
#include "eulerRhs.h"
#include "eulerBuildConfigTag.h"

#include "RiemannProblemInitialCondition.h"

using namespace TNL;

typedef eulerBuildConfigTag BuildConfig;

/****
 * Uncomment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * especially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class eulerConfig
{
   public:
      static void configSetup( Config::ConfigDescription & config )
      {
         config.addDelimiter( "Inviscid flow settings:" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         typedef Meshes::Grid< 3 > Mesh;
         LaxFridrichs< Mesh >::configSetup( config, "inviscid-operators-" );
         RiemannProblemInitialCondition< Mesh >::configSetup( config );

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
class eulerSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool run( const Config::ParameterContainer & parameters )
      {
          enum { Dimension = MeshType::getMeshDimension() };
          typedef LaxFridrichs< MeshType, Real, Index > ApproximateOperator;
          typedef eulerRhs< MeshType, Real > RightHandSide;
          typedef Containers::StaticVector < MeshType::getMeshDimension(), Real > Point;

         /****
          * Resolve the template arguments of your solver here.
          * The following code is for the Dirichlet and the Neumann boundary conditions.
          * Both can be constant or defined as descrete values of Vector.
          */
          String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
          if( parameters.checkParameter( "boundary-conditions-constant" ) )
          {
             typedef Functions::Analytic::Constant< Dimension, Real > Constant;
             if( boundaryConditionsType == "dirichlet" )
             {
                typedef Operators::DirichletBoundaryConditions< MeshType, Constant, MeshType::getMeshDimension(), Real, Index > BoundaryConditions;
                typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
             if( boundaryConditionsType == "neumann" )
             {
               typedef Operators::NeumannBoundaryConditions< MeshType, Constant, Real, Index > BoundaryConditions;
               typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
               SolverStarter solverStarter;
               return solverStarter.template run< Problem >( parameters );
             }
          }
          typedef Functions::MeshFunctionView< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef Operators::DirichletBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimension(), Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator> Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "neumann" )
          {
             typedef Operators::NeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }

      return true;}

};

int main( int argc, char* argv[] )
{
   Solvers::Solver< eulerSetter, eulerConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
