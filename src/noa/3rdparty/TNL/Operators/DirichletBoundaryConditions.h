// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Operators/Operator.h>
#include <noa/3rdparty/TNL/Functions/Analytic/Constant.h>
#include <noa/3rdparty/TNL/Functions/FunctionAdapter.h>
#include <noa/3rdparty/TNL/Functions/MeshFunction.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh,
          typename Function = Functions::Analytic::Constant< Mesh::getMeshDimension(), typename Mesh::RealType >,
          int MeshEntitiesDimension = Mesh::getMeshDimension(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class DirichletBoundaryConditions
: public Operator< Mesh,
                   Functions::MeshBoundaryDomain,
                   MeshEntitiesDimension,
                   MeshEntitiesDimension,
                   Real,
                   Index >
{
   public:

      typedef Mesh MeshType;
      typedef Function FunctionType;
      typedef Real RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Index IndexType;
      
      typedef Pointers::SharedPointer<  Mesh > MeshPointer;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::PointType PointType;

      static constexpr int getMeshDimension() { return MeshType::getMeshDimension(); }

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         Function::configSetup( config, prefix );
      }
 
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return Functions::FunctionAdapter< MeshType, FunctionType >::template setup< MeshPointer >( this->function, meshPointer, parameters, prefix );
      }

      void setFunction( const Function& function )
      {
         this->function = function;
      }

      Function& getFunction()
      {
         return this->function;
      }
 
      const Function& getFunction() const
      {
         return this->function;
      }

      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,
                                 const RealType& time = 0 ) const
      {
         //static_assert( EntityType::getDimension() == MeshEntitiesDimension, "Wrong mesh entity dimension." );
         return Functions::FunctionAdapter< MeshType, Function >::template getValue( this->function, entity, time );
      }

      template< typename EntityType >
      __cuda_callable__
      IndexType getLinearSystemRowLength( const MeshType& mesh,
                                          const IndexType& index,
                                          const EntityType& entity ) const
      {
         return 1;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      void setMatrixElements( const PreimageFunction& u,
                              const MeshEntity& entity,
                              const RealType& time,
                              const RealType& tau,
                              Matrix& matrix,
                              Vector& b ) const
      {
         auto matrixRow = matrix.getRow( entity.getIndex() );
         const IndexType& index = entity.getIndex();
         matrixRow.setElement( 0, index, 1.0 );
         b[ index ] = Functions::FunctionAdapter< MeshType, Function >::getValue( this->function, entity, time );
      }
 

   protected:

      Function function;
 
   //static_assert( Device::DeviceType == Function::Device::DeviceType );
};


template< typename Mesh,
          typename Function >
std::ostream& operator << ( std::ostream& str, const DirichletBoundaryConditions< Mesh, Function >& bc )
{
   str << "Dirichlet boundary conditions: vector = " << bc.getVector();
   return str;
}

} // namespace Operators
} // namespace noa::TNL
