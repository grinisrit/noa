#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Pointers/SharedPointer.h>
#include "transportEquationProblem.h"

using namespace TNL::Problems;

namespace TNL {

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
class transportEquationProblemEoc:
public transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >
{
   public:

      typedef typename DifferentialOperator::RealType RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef typename DifferentialOperator::IndexType IndexType;
      typedef Functions::MeshFunctionView< Mesh > MeshFunctionType;
      typedef transportEquationProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator > BaseType;
      typedef Pointers::SharedPointer<  MeshFunctionType, DeviceType > MeshFunctionPointer;
      typedef Pointers::SharedPointer<  DifferentialOperator > DifferentialOperatorPointer;
      typedef Pointers::SharedPointer<  BoundaryCondition > BoundaryConditionPointer;
      typedef Pointers::SharedPointer<  RightHandSide, DeviceType > RightHandSidePointer;
      typedef typename DifferentialOperator::VelocityFieldType VelocityFieldType;
      typedef Pointers::SharedPointer<  VelocityFieldType, DeviceType > VelocityFieldPointer;


      using typename BaseType::MeshType;
      using typename BaseType::MeshPointer;
      using typename BaseType::DofVectorType;
      using typename BaseType::DofVectorPointer;

      //using BaseType::getExplicitUpdate;

      String getPrologHeader() const;

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      bool setInitialCondition( const Config::ParameterContainer& parameters,
                                DofVectorPointer& dofs );

      /*template< typename Matrix >
      bool setupLinearSystem( Matrix& matrix );

      bool makeSnapshot( const RealType& time,
                         const IndexType& step,
                         DofVectorPointer& dofs );

      IndexType getDofs() const;

      void bindDofs( DofVectorPointer& dofs );

      void getExplicitUpdate( const RealType& time,
                              const RealType& tau,
                              DofVectorPointer& _u,
                              DofVectorPointer& _fu );*/


   protected:
};

} // namespace TNL

#include "transportEquationProblemEoc_impl.h"

