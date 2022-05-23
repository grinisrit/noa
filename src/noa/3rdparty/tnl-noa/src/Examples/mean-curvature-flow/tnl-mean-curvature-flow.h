#ifndef TNL_MEAN_CURVATIVE_FLOW_H_
#define TNL_MEAN_CURVATIVE_FLOW_H_

#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/FastBuildConfigTag.h>
#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Problems/MeanCurvatureFlowProblem.h>
#include <TNL/Operators/diffusion/OneSidedNonlinearDiffusion.h>
#include <TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <TNL/Operators/operator-Q/tnlFiniteVolumeOperatorQ.h>
#include <TNL/Operators/diffusion/nonlinear-diffusion-operators/FiniteVolumeNonlinearOperator.h>
#include <TNL/Functions/MeshFunctionView.h>

//typedef tnlDefaultConfigTag BuildConfig;
typedef FastBuildConfig BuildConfig;

template< typename ConfigTag >
class meanCurvatureFlowConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
      {
         config.addDelimiter( "Mean Curvature Flow settings:" );
         config.addEntry< String >( "numerical-scheme", "Numerical scheme for the solution approximation.", "fvm" );
            config.addEntryEnum< String >( "fdm" );
            config.addEntryEnum< String >( "fvm" );
         config.addEntry< String >( "boundary-conditions-type", "Choose the boundary conditions type.", "dirichlet");
            config.addEntryEnum< String >( "dirichlet" );
            config.addEntryEnum< String >( "neumann" );

         config.addEntry< String >( "boundary-conditions-file", "File with the values of the boundary conditions.", "boundary.tnl" );
         config.addEntry< double >( "boundary-conditions-constant", "This sets a value in case of the constant boundary conditions." );
         config.addEntry< String >( "initial-condition", "File with the initial condition.", "initial.tnl");
	      config.addEntry< double >( "right-hand-side-constant", "This sets a value in case of the constant right hand side.", 0.0 );
	      config.addEntry< double >( "eps", "This sets a eps in operator Q.", 1.0 );
      };
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class meanCurvatureFlowSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef typename MeshType::PointType Point;
   enum { Dimension = MeshType::getMeshDimension() };

   static bool run( const Config::ParameterContainer& parameters )
   {
      return setNumericalScheme( parameters );
   }
   
   static bool setNumericalScheme( const Config::ParameterContainer& parameters )
   {
      const String& numericalScheme = parameters.getParameter< String >( "numerical-scheme" );
      if( numericalScheme == "fdm" )
      {
         typedef tnlOneSideDiffOperatorQ<MeshType, Real, Index > QOperator;
         typedef tnlOneSideNonlinearDiffusion<MeshType, QOperator, Real, Index > NonlinearOperator;         
         return setBoundaryConditions< NonlinearOperator, QOperator >( parameters );
      }
      if( numericalScheme == "fvm" )
      {
         typedef tnlFiniteVolumeOperatorQ<MeshType, Real, Index, 0> QOperator;
         typedef FiniteVolumeNonlinearOperator<MeshType, QOperator, Real, Index > NonlinearOperator;         
         return setBoundaryConditions< NonlinearOperator, QOperator >( parameters );
      }
      return false;
   }
   
   template< typename NonlinearOperator,
             typename QOperator >
   static bool setBoundaryConditions( const Config::ParameterContainer& parameters )
   {
      typedef OneSidedNonlinearDiffusion< MeshType, NonlinearOperator, Real, Index > ApproximateOperator;
      typedef Constant< Dimension, Real > RightHandSide;
      typedef StaticVector< MeshType::getMeshDimension(), Real > Point;

      String boundaryConditionsType = parameters.getParameter< String >( "boundary-conditions-type" );
      if( parameters.checkParameter( "boundary-conditions-constant" ) )
      {
         typedef Constant< Dimension, Real > Constant;
         if( boundaryConditionsType == "dirichlet" )
         {
            typedef DirichletBoundaryConditions< MeshType, Constant, Dimension, Real, Index > BoundaryConditions;
            typedef MeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
            SolverStarter solverStarter;
            return solverStarter.template run< Solver >( parameters );
         }
         typedef NeumannBoundaryConditions< MeshType, Constant, Real, Index > BoundaryConditions;
         typedef MeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
         SolverStarter solverStarter;
         return solverStarter.template run< Solver >( parameters );
      }
      //typedef Vector< Real, Device, Index > VectorType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunction;
      if( boundaryConditionsType == "dirichlet" )
      {
         typedef DirichletBoundaryConditions< MeshType, MeshFunction, Dimension, Real, Index > BoundaryConditions;
         typedef MeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
         SolverStarter solverStarter;
         return solverStarter.template run< Solver >( parameters );
      }
      typedef NeumannBoundaryConditions< MeshType, MeshFunction, Real, Index > BoundaryConditions;
      typedef MeanCurvatureFlowProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   Solver< meanCurvatureFlowSetter, meanCurvatureFlowConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_MEAN_CURVATIVE_FLOW_H_ */
