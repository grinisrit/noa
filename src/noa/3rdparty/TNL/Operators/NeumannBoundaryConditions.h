// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT


#pragma once

#include <noa/3rdparty/TNL/Functions/FunctionAdapter.h>

namespace noaTNL {
namespace Operators {

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class NeumannBoundaryConditions
{

};

/****
 * Base
 */
template< typename Function >
class NeumannBoundaryConditionsBase
{
   public:
      
      typedef Function FunctionType;
      
      static void configSetup( const Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         Function::configSetup( config, prefix );
      }
      
      template< typename MeshPointer >
      bool setup( const MeshPointer& meshPointer, 
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return Functions::FunctionAdapter< typename MeshPointer::ObjectType, FunctionType >::setup( this->function, meshPointer, parameters, prefix );
      }

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         Function::configSetup( config, prefix );
      };

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
          return this->function.setup( parameters, prefix );
      };

      void setFunction( const FunctionType& function )
      {
         this->function = function;
      };
      
      FunctionType& getFunction()
      {
         return this->function;
      }

      const FunctionType& getFunction() const
      {
         return this->function;
      };

   protected:

      FunctionType function;

};

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class NeumannBoundaryConditions< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public NeumannBoundaryConditionsBase< Function >,
     public Operator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                         Functions::MeshBoundaryDomain,
                         1, 1,
                         Real,
                         Index >
{
   public:

   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Function FunctionType;
   typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
   typedef Containers::StaticVector< 1, RealType > PointType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef NeumannBoundaryConditionsBase< Function > BaseType;

   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType operator()( const MeshFunction& u,
                              const EntityType& entity,   
                              const RealType& time = 0 ) const
   {
      //const MeshType& mesh = entity.getMesh();
      const auto& neighborEntities = entity.getNeighborEntities();
      //const IndexType& index = entity.getIndex();
      if( entity.getCoordinates().x() == 0 )
         return u[ neighborEntities.template getEntityIndex< 1 >() ] + entity.getMesh().getSpaceSteps().x() * 
            Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
      else
         return u[ neighborEntities.template getEntityIndex< -1 >() ] + entity.getMesh().getSpaceSteps().x() * 
            Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );   

   }


   template< typename EntityType >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const EntityType& entity ) const
   {
      return 2;
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
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         auto matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index, 1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() * 
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         else
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< -1 >(), -1.0 );
            matrixRow.setElement( 1, index, 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }
};

/****
 * 2D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class NeumannBoundaryConditions< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public NeumannBoundaryConditionsBase< Function >,
     public Operator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >,
                         Functions::MeshBoundaryDomain,
                         2, 2,
                         Real,
                         Index >

{
   public:

      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      typedef Function FunctionType;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef Containers::StaticVector< 2, RealType > PointType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef NeumannBoundaryConditionsBase< Function > BaseType;


      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,                            
                                 const RealType& time = 0 ) const
      {
         //const MeshType& mesh = entity.getMesh();
         const auto& neighborEntities = entity.getNeighborEntities();
         //const IndexType& index = entity.getIndex();
         if( entity.getCoordinates().x() == 0 )
         {
            return u[ neighborEntities.template getEntityIndex< 1, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            return u[ neighborEntities.template getEntityIndex< -1, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, 1 >() ] + entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         // The following line is commented to avoid compiler warning
         //if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, -1 >() ] + entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }

      template< typename EntityType >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity ) const
      {
         return 2;
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
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         auto matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index,                                                1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 1, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< -1, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            matrixRow.setElement( 0, index,                                                1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 0, 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, -1 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }
};

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class NeumannBoundaryConditions< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public NeumannBoundaryConditionsBase< Function >,
     public Operator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >,
                         Functions::MeshBoundaryDomain,
                         3, 3,
                         Real,
                         Index >
{
   public:

      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      typedef Function FunctionType;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef Containers::StaticVector< 3, RealType > PointType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef NeumannBoundaryConditionsBase< Function > BaseType;   

      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,
                                 const RealType& time = 0 ) const
      {
         //const MeshType& mesh = entity.getMesh();
         const auto& neighborEntities = entity.getNeighborEntities();
         //const IndexType& index = entity.getIndex();
         if( entity.getCoordinates().x() == 0 )
         {
            return u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            return u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ] + entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] + entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ] + entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == 0 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] + entity.getMesh().getSpaceSteps().z() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         // The following line is commented to avoid compiler warning
         //if( entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ] + entity.getMesh().getSpaceSteps().z() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }   
      }


      template< typename EntityType >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity ) const
      {
         return 2;
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
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         auto matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 1, 0, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< -1, 0, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 0, 1, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() * 
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, -1, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 0, 0, 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().z() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, 0, -1 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().z() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
      }
};

template< typename Mesh,
          typename Function,
          typename Real,
          typename Index >
std::ostream& operator << ( std::ostream& str, const NeumannBoundaryConditions< Mesh, Function, Real, Index >& bc )
{
   str << "Neumann boundary conditions: function = " << bc.getFunction();
   return str;
}

} // namespace Operators
} // namespace noaTNL

