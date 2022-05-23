#ifndef HeatEquationBenchmarkPROBLEM_H_
#define HeatEquationBenchmarkPROBLEM_H_

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include "Tuning/ExplicitUpdater.h"

using namespace TNL;
using namespace TNL::Problems;

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
class HeatEquationBenchmarkProblem:
   public PDEProblem< Mesh,
                      typename DifferentialOperator::RealType,
                      typename Mesh::DeviceType,
                      typename DifferentialOperator::IndexType >
{
   public:
      using RealType = typename DifferentialOperator::RealType;
      using DeviceType = typename Mesh::DeviceType;
      using IndexType = typename DifferentialOperator::IndexType;
      using MeshFunctionViewType = Functions::MeshFunctionView< Mesh >;
      using MeshFunctionViewPointer = Pointers::SharedPointer< MeshFunctionViewType, DeviceType >;
      using BaseType = PDEProblem< Mesh, RealType, DeviceType, IndexType >;
      using DifferentialOperatorPointer = Pointers::SharedPointer< DifferentialOperator >;
      using BoundaryConditionPointer = Pointers::SharedPointer< BoundaryCondition >;
      using RightHandSidePointer = Pointers::SharedPointer< RightHandSide, DeviceType >;

      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorPointer;

      HeatEquationBenchmarkProblem();

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                DofVectorPointer& dofsPointer );

      template< typename Matrix >
      bool setupLinearSystem( Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         DofVectorPointer& dofsPointer );

      IndexType getDofs() const;

      void bindDofs( DofVectorPointer& dofsPointer );

      void getExplicitUpdate( const RealType& time,
                              const RealType& tau,
                              DofVectorPointer& _uPointer,
                              DofVectorPointer& _fuPointer );

      void applyBoundaryConditions( const RealType& time,
                                       DofVectorPointer& dofs );

      template< typename MatrixPointer >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 DofVectorPointer& dofs,
                                 MatrixPointer& matrix,
                                 DofVectorPointer& rightHandSide );

      ~HeatEquationBenchmarkProblem();

   protected:

      DifferentialOperatorPointer differentialOperatorPointer;
      BoundaryConditionPointer boundaryConditionPointer;
      RightHandSidePointer rightHandSidePointer;

      MeshFunctionViewPointer fu, u;

      String cudaKernelType;

      MeshType* cudaMesh;
      BoundaryCondition* cudaBoundaryConditions;
      RightHandSide* cudaRightHandSide;
      DifferentialOperator* cudaDifferentialOperator;

      TNL::ExplicitUpdater< Mesh, MeshFunctionViewType, DifferentialOperator, BoundaryCondition, RightHandSide > tuningExplicitUpdater;
      TNL::Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionViewType, DifferentialOperator, BoundaryCondition, RightHandSide > explicitUpdater;

};

#include "HeatEquationBenchmarkProblem_impl.h"

#endif /* HeatEquationBenchmarkPROBLEM_H_ */
