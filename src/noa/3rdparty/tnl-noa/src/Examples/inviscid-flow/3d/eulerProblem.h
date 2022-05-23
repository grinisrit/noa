#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunctionView.h>

using namespace TNL::Problems;

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
           typename DifferentialOperator >
class eulerProblem:
   public PDEProblem< Mesh,
                         typename DifferentialOperator::RealType,
                         typename Mesh::DeviceType,
                         typename DifferentialOperator::IndexType >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::MeshFunctionView< Mesh > MeshFunctionType;
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef Pointers::SharedPointer< DifferentialOperator > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer< BoundaryCondition > BoundaryConditionPointer;
      typedef Pointers::SharedPointer< RightHandSide, DeviceType > RightHandSidePointer;
      typedef PDEProblem< Mesh, RealType, DeviceType, IndexType > BaseType;      
      
      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;
      using typename BaseType::MeshDependentDataType;
      using typename BaseType::MeshDependentDataPointer;

      typedef typename DifferentialOperator::Continuity Continuity;
      typedef typename DifferentialOperator::MomentumX MomentumX;
      typedef typename DifferentialOperator::MomentumY MomentumY;
      typedef typename DifferentialOperator::MomentumZ MomentumZ;
      typedef typename DifferentialOperator::Energy Energy;
      typedef typename DifferentialOperator::Velocity Velocity;
      typedef typename DifferentialOperator::VelocityX VelocityX;
      typedef typename DifferentialOperator::Pressure Pressure;

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                const MeshPointer& mesh,
                                DofVectorPointer& dofs,
                                MeshDependentDataPointer& meshDependentData );

      template< typename Matrix >
      bool setupLinearSystem( const MeshPointer& mesh,
                              Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         const MeshPointer& mesh,
                         DofVectorPointer& dofs,
                         MeshDependentDataPointer& meshDependentData );

      IndexType getDofs( const MeshPointer& mesh ) const;

      void bindDofs( const MeshPointer& mesh,
                     DofVectorPointer& dofs );

      void getExplicitRHS( const RealType& time,
                           const RealType& tau,
                           const MeshPointer& mesh,
                           DofVectorPointer& _u,
                           DofVectorPointer& _fu,
                           MeshDependentDataPointer& meshDependentData );

      template< typename Matrix >
      void assemblyLinearSystem( const RealType& time,
                                 const RealType& tau,
                                 const MeshPointer& mesh,
                                 DofVectorPointer& dofs,
                                 Matrix& matrix,
                                 DofVectorPointer& rightHandSide,
                                 MeshDependentDataPointer& meshDependentData );

      bool postIterate( const RealType& time,
                        const RealType& tau,
                        const MeshPointer& mesh,
                        DofVectorPointer& dofs,
                        MeshDependentDataPointer& meshDependentData );

   protected:

      DifferentialOperatorPointer differentialOperatorPointer;
      BoundaryConditionPointer boundaryConditionPointer;
      RightHandSidePointer rightHandSidePointer;

      MeshFunctionPointer uRho, uRhoVelocityX, uRhoVelocityY, uRhoVelocityZ, uEnergy;
      MeshFunctionPointer fuRho, fuRhoVelocityX, fuRhoVelocityY, fuRhoVelocityZ, fuEnergy;
      
      MeshFunctionPointer pressure, velocity, velocityX, velocityY, velocityZ;
      
      RealType gamma;

};

} // namespace TNL

#include "eulerProblem_impl.h"

