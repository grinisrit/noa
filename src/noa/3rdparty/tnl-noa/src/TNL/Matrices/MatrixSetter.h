// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SharedPointer.h>

namespace noa::TNL {
namespace Matrices {

template< typename DifferentialOperator, typename BoundaryConditions, typename RowsCapacitiesType >
class MatrixSetterTraverserUserData
{
public:
   using DeviceType = typename RowsCapacitiesType::DeviceType;

   const DifferentialOperator* differentialOperator;

   const BoundaryConditions* boundaryConditions;

   RowsCapacitiesType* rowLengths;

   MatrixSetterTraverserUserData( const DifferentialOperator* differentialOperator,
                                  const BoundaryConditions* boundaryConditions,
                                  RowsCapacitiesType* rowLengths )
   : differentialOperator( differentialOperator ), boundaryConditions( boundaryConditions ), rowLengths( rowLengths )
   {}
};

template< typename Mesh, typename DifferentialOperator, typename BoundaryConditions, typename RowsCapacitiesType >
class MatrixSetter
{
public:
   using MeshType = Mesh;
   using MeshPointer = Pointers::SharedPointer< MeshType >;
   using DeviceType = typename MeshType::DeviceType;
   using IndexType = typename RowsCapacitiesType::RealType;
   using TraverserUserData = MatrixSetterTraverserUserData< DifferentialOperator, BoundaryConditions, RowsCapacitiesType >;
   using DifferentialOperatorPointer = Pointers::SharedPointer< DifferentialOperator, DeviceType >;
   using BoundaryConditionsPointer = Pointers::SharedPointer< BoundaryConditions, DeviceType >;
   using RowsCapacitiesTypePointer = Pointers::SharedPointer< RowsCapacitiesType, DeviceType >;

   template< typename EntityType >
   void
   getCompressedRowLengths( const MeshPointer& meshPointer,
                            const DifferentialOperatorPointer& differentialOperatorPointer,
                            const BoundaryConditionsPointer& boundaryConditionsPointer,
                            RowsCapacitiesTypePointer& rowLengthsPointer ) const;

   class TraverserBoundaryEntitiesProcessor
   {
   public:
      template< typename EntityType >
      __cuda_callable__
      static void
      processEntity( const MeshType& mesh, TraverserUserData& userData, const EntityType& entity )
      {
         ( *userData.rowLengths )[ entity.getIndex() ] =
            userData.boundaryConditions->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
      }
   };

   class TraverserInteriorEntitiesProcessor
   {
   public:
      template< typename EntityType >
      __cuda_callable__
      static void
      processEntity( const MeshType& mesh, TraverserUserData& userData, const EntityType& entity )
      {
         ( *userData.rowLengths )[ entity.getIndex() ] =
            userData.differentialOperator->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
      }
   };
};

/*
template< int Dimension,
          typename Real,
          typename Device,
          typename Index,
          typename DifferentialOperator,
          typename BoundaryConditions,
          typename RowsCapacitiesType >
class MatrixSetter< Meshes::Grid< Dimension, Real, Device, Index >,
                       DifferentialOperator,
                       BoundaryConditions,
                       RowsCapacitiesType >
{
   public:
   typedef Meshes::Grid< Dimension, Real, Device, Index > MeshType;
   typedef typename MeshType::DeviceType DeviceType;
   typedef typename RowsCapacitiesType::RealType IndexType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef MatrixSetterTraverserUserData< DifferentialOperator,
                                             BoundaryConditions,
                                             RowsCapacitiesType > TraverserUserData;

   template< typename EntityType >
   void getCompressedRowLengths( const MeshType& mesh,
                       const DifferentialOperator& differentialOperator,
                       const BoundaryConditions& boundaryConditions,
                       RowsCapacitiesType& rowLengths ) const;

   class TraverserBoundaryEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.rowLengths )[ entity.getIndex() ] =
                     userData.boundaryConditions->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }

   };

   class TraverserInteriorEntitiesProcessor
   {
      public:

         template< typename EntityType >
         __cuda_callable__
         static void processEntity( const MeshType& mesh,
                                    TraverserUserData& userData,
                                    const EntityType& entity )
         {
            ( *userData.rowLengths )[ entity.getIndex() ] =
                     userData.differentialOperator->getLinearSystemRowLength( mesh, entity.getIndex(), entity );
         }
   };

};
*/

}  // namespace Matrices
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixSetter_impl.h>
