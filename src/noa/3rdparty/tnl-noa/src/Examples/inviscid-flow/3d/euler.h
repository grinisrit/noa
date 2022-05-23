#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include "eulerProblem.h"
#include "LaxFridrichs3D.h"
#include "eulerRhs.h"
#include "eulerBuildConfigTag.h"
#include "MyMixedBoundaryConditions.h"
#include "MyNeumannBoundaryConditions.h"

using namespace TNL;

typedef eulerBuildConfigTag BuildConfig;

/****
 * Uncoment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * esppecially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class eulerConfig
{
   public:
      static void configSetup( Config::ConfigDescription & config )
      {
         config.addDelimiter( "euler2D settings:" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );
            config.addEntryEnum< String >( "mymixed" );
            config.addEntryEnum< String >( "myneumann" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "NWU-density", "This sets a value of northwest up density." );
         config.addEntry< double >( "NWU-velocityX", "This sets a value of northwest up x velocity." );
         config.addEntry< double >( "NWU-velocityY", "This sets a value of northwest up y velocity." );
         config.addEntry< double >( "NWU-velocityZ", "This sets a value of northwest up z velocity." );
         config.addEntry< double >( "NWU-pressure", "This sets a value of northwest up pressure." );
         config.addEntry< double >( "SWU-density", "This sets a value of southwest up density." );
         config.addEntry< double >( "SWU-velocityX", "This sets a value of southwest up x velocity." );
         config.addEntry< double >( "SWU-velocityY", "This sets a value of southwest up y velocity." );
         config.addEntry< double >( "SWU-velocityZ", "This sets a value of southwest up z velocity." );
         config.addEntry< double >( "SWU-pressure", "This sets a value of southwest up pressure." );
         config.addEntry< double >( "NWD-density", "This sets a value of northwest down density." );
         config.addEntry< double >( "NWD-velocityX", "This sets a value of northwest down x velocity." );
         config.addEntry< double >( "NWD-velocityY", "This sets a value of northwest down y velocity." );
         config.addEntry< double >( "NWD-velocityZ", "This sets a value of northwest down z velocity." );
         config.addEntry< double >( "NWD-pressure", "This sets a value of northwest down pressure." );
         config.addEntry< double >( "SWD-density", "This sets a value of southwest down density." );
         config.addEntry< double >( "SWD-velocityX", "This sets a value of southwest down x velocity." );
         config.addEntry< double >( "SWD-velocityY", "This sets a value of southwest down y velocity." );
         config.addEntry< double >( "SWF-velocityZ", "This sets a value of southwest down z velocity." );
         config.addEntry< double >( "SWD-pressure", "This sets a value of southwest down pressure." );
         config.addEntry< double >( "riemann-border", "This sets a position of discontinuity cross." );
         config.addEntry< double >( "NEU-density", "This sets a value of northeast up density." );
         config.addEntry< double >( "NEU-velocityX", "This sets a value of northeast up x velocity." );
         config.addEntry< double >( "NEU-velocityY", "This sets a value of northeast up y velocity." );
         config.addEntry< double >( "NEU-velocityZ", "This sets a value of northeast up z velocity." );
         config.addEntry< double >( "NEU-pressure", "This sets a value of northeast up pressure." );
         config.addEntry< double >( "SEU-density", "This sets a value of southeast up density." );
         config.addEntry< double >( "SEU-velocityX", "This sets a value of southeast up x velocity." );
         config.addEntry< double >( "SEU-velocityY", "This sets a value of southeast up y velocity." );
         config.addEntry< double >( "SEU-velocityZ", "This sets a value of southeast up z velocity." );
         config.addEntry< double >( "SEU-pressure", "This sets a value of southeast up pressure." );
         config.addEntry< double >( "NED-density", "This sets a value of northeast down density." );
         config.addEntry< double >( "NED-velocityX", "This sets a value of northeast down x velocity." );
         config.addEntry< double >( "NED-velocityY", "This sets a value of northeast down y velocity." );
         config.addEntry< double >( "NED-velocityZ", "This sets a value of northeast down z velocity." );
         config.addEntry< double >( "NED-pressure", "This sets a value of northeast down pressure." );
         config.addEntry< double >( "SED-density", "This sets a value of southeast down density." );
         config.addEntry< double >( "SED-velocityX", "This sets a value of southeast down x velocity." );
         config.addEntry< double >( "SED-velocityY", "This sets a value of southeast down y velocity." );
         config.addEntry< double >( "SED-velocityZ", "This sets a value of southeast down z velocity." );
         config.addEntry< double >( "SED-pressure", "This sets a value of southeast down pressure." );
         config.addEntry< double >( "gamma", "This sets a value of gamma constant." );

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
          enum { Dimensions = MeshType::getMeshDimensions() };
          typedef LaxFridrichs3D< MeshType, Real, Index > ApproximateOperator;
          typedef eulerRhs< MeshType, Real > RightHandSide;
          typedef Containers::StaticVector < MeshType::getMeshDimensions(), Real > Vertex;

         /****
          * Resolve the template arguments of your solver here.
          * The following code is for the Dirichlet and the Neumann boundary conditions.
          * Both can be constant or defined as descrete values of Vector.
          */
          String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
          if( parameters.checkParameter( "boundary-conditions-constant" ) )
          {
             typedef Functions::Analytic::Constant< Dimensions, Real > Constant;
             if( boundaryConditionsType == "dirichlet" )
             {
                typedef Operators::DirichletBoundaryConditions< MeshType, Constant, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
                typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
             typedef Operators::NeumannBoundaryConditions< MeshType, Constant, Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          typedef Functions::MeshFunctionView< MeshType > MeshFunction;
          if( boundaryConditionsType == "dirichlet" )
          {
             typedef Operators::DirichletBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "mymixed" )
          {
             typedef Operators::MyMixedBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
             SolverStarter solverStarter;
             return solverStarter.template run< Problem >( parameters );
          }
          if( boundaryConditionsType == "myneumann" )
          {
             typedef Operators::MyNeumannBoundaryConditions< MeshType, MeshFunction, MeshType::getMeshDimensions(), Real, Index > BoundaryConditions;
             typedef eulerProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
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
