#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include "navierStokesProblem.h"
#include "Upwind.h"
#include "navierStokesRhs.h"
#include "navierStokesBuildConfigTag.h"

#include "RiemannProblemInitialCondition.h"
#include "BoundaryConditionsCavity.h"
#include "BoundaryConditionsBoiler.h"

using namespace TNL;

typedef navierStokesBuildConfigTag BuildConfig;

/****
 * Uncomment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * especially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class navierStokesConfig
{
   public:
      static void configSetup( Config::ConfigDescription & config )
      {
         config.addDelimiter( "Inviscid flow settings:" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "cavity");
            config.addEntryEnum< String >( "boiler" );
            config.addEntryEnum< String >( "cavity" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< double >( "speed-increment", "This sets increment of input speed.", 0.0 );
         config.addEntry< double >( "speed-increment-until", "This sets time until input speed will rose", -0.1 );
         config.addEntry< double >( "cavity-speed", "This sets speed parameter of cavity", 0.0 );
         typedef Meshes::Grid< 3 > Mesh;
         Upwind< Mesh >::configSetup( config, "inviscid-operators-" );
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
class navierStokesSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static bool run( const Config::ParameterContainer & parameters )
      {
          enum { Dimension = MeshType::getMeshDimension() };
          typedef Upwind< MeshType, Real, Index > ApproximateOperator;
          typedef navierStokesRhs< MeshType, Real > RightHandSide;
          typedef Containers::StaticVector < MeshType::getMeshDimension(), Real > Point;

         /****
          * Resolve the template arguments of your solver here.
          * The following code is for the Dirichlet and the Neumann boundary conditions.
          * Both can be constant or defined as descrete values of Vector.
          */
          typedef Functions::Analytic::Constant< Dimension, Real > Constant;
          String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
          if( boundaryConditionsType == "cavity" )
             {
                typedef BoundaryConditionsCavity< MeshType, Constant, Real, Index > BoundaryConditions;
                typedef navierStokesProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }
           if( boundaryConditionsType == "boiler" )
             {
                typedef BoundaryConditionsBoiler< MeshType, Constant, Real, Index > BoundaryConditions;
                typedef navierStokesProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Problem;
                SolverStarter solverStarter;
                return solverStarter.template run< Problem >( parameters );
             }

      return true;}

};

int main( int argc, char* argv[] )
{
   Solvers::Solver< navierStokesSetter, navierStokesConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
