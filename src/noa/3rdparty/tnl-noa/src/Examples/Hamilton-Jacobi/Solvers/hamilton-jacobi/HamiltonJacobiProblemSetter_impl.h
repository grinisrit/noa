#pragma once

#include <mesh/tnlGrid.h>
#include <functions/tnlConstantFunction.h>
#include <operators/tnlNeumannBoundaryConditions.h>
#include <operators/tnlDirichletBoundaryConditions.h>
#include <operators/hamilton-jacobi/upwindEikonal.h>
#include <operators/hamilton-jacobi/godunovEikonal.h>
#include <operators/hamilton-jacobi/tnlEikonalOperator.h>


template< typename RealType,
          typename DeviceType,
          typename IndexType,
          typename MeshType,
          typename ConfigTag,
          typename SolverStarter >
bool HamiltonJacobiProblemSetter< RealType, DeviceType, IndexType, MeshType, ConfigTag, SolverStarter > :: run( const Config::ParameterContainer& parameters )
{
   static const int Dimensions = MeshType::getMeshDimension();

   if( Dimensions <= 0 || Dimensions > 3 )
   {
     std::cerr << "The problem is not defined for " << Dimensions << "dimensions." <<std::endl;
      return false;
   }
   else
   {
      typedef Containers::StaticVector < Dimensions, RealType > Point;
      typedef tnlConstantFunction< Dimensions, RealType > ConstantFunctionType;
      typedef tnlNeumannBoundaryConditions< MeshType, ConstantFunctionType, RealType, IndexType > BoundaryConditions;

      SolverStarter solverStarter;

      const String& schemeName = parameters.getParameter< String >( "scheme" );

      if( schemeName == "upwind" )
      {
           typedef upwindEikonalScheme< MeshType, RealType, IndexType > GradientNormOperator;
           typedef tnlConstantFunction< Dimensions, RealType > RightHandSide;
           typedef tnlEikonalOperator< GradientNormOperator, RightHandSide > Operator;
           typedef HamiltonJacobiProblem< MeshType, Operator, BoundaryConditions, RightHandSide > Solver;
           return solverStarter.template run< Solver >( parameters );
      }
      if( schemeName == "godunov" )
      {
           typedef godunovEikonalScheme< MeshType, RealType, IndexType > GradientNormOperator;
           typedef tnlConstantFunction< Dimensions, RealType > RightHandSide;
           typedef tnlEikonalOperator< GradientNormOperator, RightHandSide > Operator;
           typedef HamiltonJacobiProblem< MeshType, Operator, BoundaryConditions, RightHandSide > Solver;
           return solverStarter.template run< Solver >( parameters );
      }      
      else
        std::cerr << "Unknown scheme '" << schemeName << "'." <<std::endl;


      return false;
   }
}
