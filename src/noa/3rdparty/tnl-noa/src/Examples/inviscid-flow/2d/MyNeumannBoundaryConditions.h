/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef MYNEUMANNBOUNDARYCONDITIONS_H_
#define MYNEUMANNBOUNDARYCONDITIONS_H_

#pragma once

#include <TNL/Operators/Operator.h>
#include <TNL/Functions/Analytic/Constant.h>
#include <TNL/Functions/FunctionAdapter.h>
#include <TNL/Functions/MeshFunctionView.h>

namespace TNL {
namespace Operators {

template< typename Mesh,
          typename Function = Functions::Analytic::Constant< Mesh::getMeshDimensions(), typename Mesh::RealType >,
          int MeshEntitiesDimensions = Mesh::getMeshDimensions(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class MyNeumannBoundaryConditions
: public Operator< Mesh,
                   Functions::MeshBoundaryDomain,
                   MeshEntitiesDimensions,
                   MeshEntitiesDimensions,
                   Real,
                   Index >
{
   public:

      typedef Mesh MeshType;
      typedef Function FunctionType;
      typedef Real RealType;
      typedef typename MeshType::DeviceType DeviceType;
      typedef Index IndexType;
      
      typedef Pointers::SharedPointer< Mesh > MeshPointer;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef typename MeshType::VertexType VertexType;

      static constexpr int getMeshDimensions() { return MeshType::meshDimensions; }

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         Function::configSetup( config, prefix );
      }
 
      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return true; //Functions::FunctionAdapter< MeshType, FunctionType >::template setup< MeshPointer >( this->function, meshPointer, parameters, prefix );
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
      const MeshType& mesh = entity.getMesh();
      const auto& neighbourEntities = entity.getNeighbourEntities();
      typedef typename MeshType::Cell Cell;
      int count = mesh.template getEntitiesCount< Cell >();
      count = std::sqrt(count);
      if( entity.getCoordinates().x() == 0 )
         return u[ neighbourEntities.template getEntityIndex< 1, 0 >() ];
         else if( entity.getCoordinates().x() == count-1 )
            return u[ neighbourEntities.template getEntityIndex< -1, 0 >() ];
            else if( entity.getCoordinates().y() == 0 )
               return u[ neighbourEntities.template getEntityIndex< 0, 1 >() ];
               else return u[ neighbourEntities.template getEntityIndex< 0, -1 >() ];
         //tady se asi delaji okrajove podminky
         //static_assert( EntityType::getDimensions() == MeshEntitiesDimensions, "Wrong mesh entity dimensions." );
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
         typename Matrix::MatrixRow matrixRow = matrix.getRow( entity.getIndex() );
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
std::ostream& operator << ( std::ostream& str, const MyNeumannBoundaryConditions< Mesh, Function >& bc )
{
   str << "MyNeumann boundary conditions: vector = " << bc.getVector();
   return str;
}

} // namespace Operators
} // namespace TNL

#endif /* MYNEUMANNBOUNDARYCONDITIONS_H_ */
