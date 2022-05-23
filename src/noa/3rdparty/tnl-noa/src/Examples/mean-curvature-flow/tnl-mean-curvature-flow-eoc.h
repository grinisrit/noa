#ifndef TNL_MEAN_CURVATURE_FLOW_EOC_H_
#define TNL_MEAN_CURVATURE_FLOW_EOC_H_

#include <TNL/Solvers/Solver.h>
#include <TNL/Solvers/FastBuildConfigTag.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Functions/TestFunction.h>
#include <TNL/Operators/DirichletBoundaryConditions.h>
#include <TNL/Operators/NeumannBoundaryConditions.h>
#include <TNL/Problems/MeanCurvatureFlowEocRhs.h>
#include <TNL/Problems/MeanCurvatureFlowEocProblem.h>
#include <TNL/Operators/diffusion/ExactNonlinearDiffusion.h>
#include <TNL/Operators/diffusion/NonlinearDiffusion.h>
#include <TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ.h>
#include <TNL/Operators/operator-Q/tnlFiniteVolumeOperatorQ.h>
#include <TNL/Operators/diffusion/ExactNonlinearDiffusion.h>
#include <TNL/Operators/diffusion/nonlinear-diffusion-operators/tnlOneSideDiffNonlinearOperator.h>
#include <TNL/Operators/diffusion/nonlinear-diffusion-operators/FiniteVolumeNonlinearOperator.h>
#include <TNL/Operators/geometric/ExactGradientNorm.h>

//typedef tnlDefaultConfigTag BuildConfig;
typedef FastBuildConfig BuildConfig;

template< typename ConfigTag >
class meanCurvatureFlowEocConfig
{
   public:
      static void configSetup( Config::ConfigDescription& config )
      {
         config.addDelimiter( "Mean Curvature Flow EOC settings:" );         
         config.addEntry< String >( "numerical-scheme", "Numerical scheme for the solution approximation.", "fvm" );
            config.addEntryEnum< String >( "fdm" );
            config.addEntryEnum< String >( "fvm" );

         config.addEntry< double >( "eps", "This sets a eps in operator Q.", 1.0 );
         config.addDelimiter( "Tests setting::" );         
         TestFunction< 3, double >::configSetup( config );
      }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
class meanCurvatureFlowEocSetter
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef typename MeshType::PointType Point;
   enum { Dimension = MeshType::getMeshDimension() };

   static bool run( const Config::ParameterContainer& parameters )
   {

      typedef tnlFiniteVolumeOperatorQ<MeshType, Real, Index, 0> OperatorQ;
      typedef FiniteVolumeNonlinearOperator<MeshType, OperatorQ, Real, Index > NonlinearOperator;
      typedef NonlinearDiffusion< MeshType, NonlinearOperator, Real, Index > ApproximateOperator;
      typedef ExactNonlinearDiffusion< ExactGradientNorm< Dimension >, Dimension > ExactOperator;
      typedef TestFunction< MeshType::getMeshDimension(), Real, Device > TestFunction;
      typedef MeanCurvatureFlowEocRhs< ExactOperator, TestFunction, Dimension > RightHandSide;
      typedef StaticVector < MeshType::getMeshDimension(), Real > Point;
      typedef DirichletBoundaryConditions< MeshType, TestFunction, Dimension, Real, Index > BoundaryConditions;
      typedef MeanCurvatureFlowEocProblem< MeshType, BoundaryConditions, RightHandSide, ApproximateOperator > Solver;
      SolverStarter solverStarter;
      return solverStarter.template run< Solver >( parameters );
   };
};

int main( int argc, char* argv[] )
{
   Solver< meanCurvatureFlowEocSetter, meanCurvatureFlowEocConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}

#endif /* TNL_MEAN_CURVATURE_FLOW_EOC_H_ */
