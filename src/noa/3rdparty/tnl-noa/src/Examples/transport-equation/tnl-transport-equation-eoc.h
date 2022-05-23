#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Operators/Advection/LaxFridrichs.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Functions/Analytic/VectorNorm.h>
#include <TNL/Operators/Analytic/Heaviside.h>
#include <TNL/Functions/VectorField.h>
#include <TNL/Meshes/Grid.h>
#include "transportEquationProblemEoc.h"
#include "transportEquationBuildConfigTag.h"

using namespace TNL;

typedef transportEquationBuildConfigTag BuildConfig;

/****
 * Uncomment the following (and comment the previous line) for the complete build.
 * This will include support for all floating point precisions, all indexing types
 * and more solvers. You may then choose between them from the command line.
 * The compile time may, however, take tens of minutes or even several hours,
 * especially if CUDA is enabled. Use this, if you want, only for the final build,
 * not in the development phase.
 */
//typedef tnlDefaultConfigTag BuildConfig;

template< typename ConfigTag >class advectionConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
      {
         config.addDelimiter( "Transport equation settings:" );
         config.addDelimiter( "Initial condition" );
         config.addEntry< String >( "initial-condition", "Set type of initial condition.", "heaviside-vector-norm" );
            config.addEntryEnum< String >( "heaviside-vector-norm" );
         Functions::Analytic::VectorNorm< 3, double >::configSetup( config, "vector-norm-" );
         Operators::Analytic::Heaviside< 3, double >::configSetup( config, "heaviside-" );
         Operators::Analytic::Shift< 3, double >::configSetup( config, "heaviside-" );

         config.addDelimiter( "Velocity field" );
         config.addEntry< String >( "velocity-field", "Type of velocity field.", "constant" );
            config.addEntryEnum< String >( "constant" );
         Functions::VectorField< 3, Functions::Analytic::Constant< 3 > >::configSetup( config, "velocity-field-" );

         config.addDelimiter( "Numerical scheme" );
         typedef Meshes::Grid< 3 > MeshType;
         Operators::Advection::LaxFridrichs< MeshType >::configSetup( config );

         config.addDelimiter( "Boundary conditions" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions.", 0.0 );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class advectionSetter
{
   public:

      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      static const int Dimensions = MeshType::getMeshDimension();

      template< typename Problem >
      static bool callSolverStarter( const Config::ParameterContainer& parameters )
      {
         SolverStarter solverStarter;
         return solverStarter.template run< Problem >( parameters );
      }

      template< typename DifferentialOperatorType >
      static bool setBoundaryConditionsType( const Config::ParameterContainer& parameters )
      {
         typedef Functions::Analytic::Constant< Dimensions, Real > ConstantFunctionType;
         String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
         if( boundaryConditionsType == "dirichlet" )
         {
            typedef Operators::DirichletBoundaryConditions< MeshType, ConstantFunctionType, MeshType::getMeshDimension(), Real, Index > BoundaryConditions;
            typedef transportEquationProblemEoc< MeshType, BoundaryConditions, ConstantFunctionType, DifferentialOperatorType > Problem;
            return callSolverStarter< Problem >( parameters );
         }
         if( boundaryConditionsType == "neumann" )
         {
            typedef Operators::DirichletBoundaryConditions< MeshType, ConstantFunctionType, MeshType::getMeshDimension(), Real, Index > BoundaryConditions;
            typedef transportEquationProblemEoc< MeshType, BoundaryConditions, ConstantFunctionType, DifferentialOperatorType > Problem;
            return callSolverStarter< Problem >( parameters );
         }
         std::cerr << "Unknown boundary conditions type: " << boundaryConditionsType << "." << std::endl;
         return false;
      }

      template< typename VelocityFieldType >
      static bool setDifferentialOperatorType( const Config::ParameterContainer& parameters )
      {
         typedef Operators::Advection::LaxFridrichs< MeshType, Real, Index, VelocityFieldType > DifferentialOperatorType;
         return setBoundaryConditionsType< DifferentialOperatorType >( parameters );
      }

      static bool setVelocityFieldType( const Config::ParameterContainer& parameters )
      {
         String velocityFieldType = parameters.getParameter< String >( "velocity-field" );
         if( velocityFieldType == "constant" )
         {
            typedef Functions::Analytic::Constant< Dimensions, RealType > VelocityFieldType;
            return setDifferentialOperatorType< VelocityFieldType >( parameters );
         }
         return false;
      }

      static bool run( const Config::ParameterContainer& parameters )
      {
         return setVelocityFieldType( parameters );
      }
};

int main( int argc, char* argv[] )
{
   Solvers::Solver< advectionSetter, advectionConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

