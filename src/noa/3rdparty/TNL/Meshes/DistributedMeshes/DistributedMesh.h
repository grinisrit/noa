// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/Meshes/DistributedMeshes/GlobalIndexStorage.h>
#include <TNL/Meshes/MeshDetails/IndexPermutationApplier.h>

namespace TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename Mesh >
class DistributedMesh
: protected GlobalIndexStorageFamily< Mesh >
{
public:
   using MeshType           = Mesh;
   using Config             = typename Mesh::Config;
   using DeviceType         = typename Mesh::DeviceType;
   using GlobalIndexType    = typename Mesh::GlobalIndexType;
   using LocalIndexType     = typename Mesh::LocalIndexType;
   using PointType          = typename Mesh::PointType;
   using RealType           = typename PointType::RealType;
   using GlobalIndexArray   = typename Mesh::GlobalIndexArray;
   using VTKTypesArrayType  = Containers::Array< std::uint8_t, Devices::Sequential, GlobalIndexType >;

   DistributedMesh() = default;

   DistributedMesh( MeshType&& localMesh )
   : localMesh( std::move(localMesh) )
   {}

   DistributedMesh( const DistributedMesh& ) = default;

   DistributedMesh( DistributedMesh&& ) = default;

   DistributedMesh& operator=( const DistributedMesh& ) = default;

   DistributedMesh& operator=( DistributedMesh&& ) = default;

   template< typename Mesh_ >
   DistributedMesh& operator=( const Mesh_& other )
   {
      GlobalIndexStorageFamily< Mesh >::operator=( other );
      localMesh = other.getLocalMesh();
      communicator = other.getCommunicator();
      ghostLevels = other.getGhostLevels();
      vtkPointGhostTypesArray = other.vtkPointGhostTypes();
      vtkCellGhostTypesArray = other.vtkCellGhostTypes();
      return *this;
   }

   bool operator==( const DistributedMesh& other ) const
   {
      return ( GlobalIndexStorageFamily< Mesh, DeviceType >::operator==( other ) &&
               localMesh == other.getLocalMesh() &&
               communicator == other.getCommunicator() &&
               ghostLevels == other.getGhostLevels() &&
               vtkPointGhostTypesArray == other.vtkPointGhostTypes() &&
               vtkCellGhostTypesArray == other.vtkCellGhostTypes() );
   }

   bool operator!=( const DistributedMesh& other ) const
   {
      return ! operator==( other );
   }

   /**
    * Common methods redirected to the local mesh
    */
   static constexpr int getMeshDimension()
   {
      return MeshType::getMeshDimension();
   }

   // types of common entities
   using Cell = typename MeshType::template EntityType< getMeshDimension() >;
   using Face = typename MeshType::template EntityType< getMeshDimension() - 1 >;
   using Vertex = typename MeshType::template EntityType< 0 >;

   static_assert( Mesh::Config::entityTagsStorage( getMeshDimension() ),
                  "DistributedMesh must store entity tags on cells" );
   static_assert( Mesh::Config::entityTagsStorage( 0 ),
                  "DistributedMesh must store entity tags on vertices" );


   /**
    * Methods specific to the distributed mesh
    */
   void setCommunicator( MPI_Comm communicator )
   {
      this->communicator = communicator;
   }

   MPI_Comm getCommunicator() const
   {
      return communicator;
   }

   const MeshType& getLocalMesh() const
   {
      return localMesh;
   }

   MeshType& getLocalMesh()
   {
      return localMesh;
   }

   void setGhostLevels( int levels )
   {
      ghostLevels = levels;
   }

   int getGhostLevels() const
   {
      return ghostLevels;
   }

   template< int Dimension >
   const GlobalIndexArray&
   getGlobalIndices() const
   {
      return GlobalIndexStorage< MeshType, DeviceType, Dimension >::getGlobalIndices();
   }

   template< int Dimension >
   GlobalIndexArray&
   getGlobalIndices()
   {
      return GlobalIndexStorage< MeshType, DeviceType, Dimension >::getGlobalIndices();
   }

   VTKTypesArrayType&
   vtkCellGhostTypes()
   {
      return vtkCellGhostTypesArray;
   }

   const VTKTypesArrayType&
   vtkCellGhostTypes() const
   {
      return vtkCellGhostTypesArray;
   }

   VTKTypesArrayType&
   vtkPointGhostTypes()
   {
      return vtkPointGhostTypesArray;
   }

   const VTKTypesArrayType&
   vtkPointGhostTypes() const
   {
      return vtkPointGhostTypesArray;
   }

   // wrapper for MeshType::reorderEntities - reorders the local mesh, global indices and,
   // if applicable, the vtkCellGhostTypes/vtkPointGhostTypes arrays
   template< int Dimension >
   void reorderEntities( const GlobalIndexArray& perm,
                         const GlobalIndexArray& iperm )
   {
      localMesh.template reorderEntities< Dimension >( perm, iperm );
      if( getGlobalIndices< Dimension >().getSize() > 0 )
         IndexPermutationApplier< MeshType, Dimension >::permuteArray( getGlobalIndices< Dimension >(), perm );
      if( Dimension == 0 && vtkPointGhostTypes().getSize() > 0 )
         IndexPermutationApplier< MeshType, Dimension >::permuteArray( vtkPointGhostTypes(), perm );
      if( Dimension == getMeshDimension() && vtkCellGhostTypes().getSize() > 0 )
         IndexPermutationApplier< MeshType, Dimension >::permuteArray( vtkCellGhostTypes(), perm );
   }

   void
   printInfo( std::ostream& str ) const
   {
      const GlobalIndexType verticesCount = localMesh.template getEntitiesCount< 0 >();
      const GlobalIndexType cellsCount = localMesh.template getEntitiesCount< Mesh::getMeshDimension() >();

      MPI::Barrier();
      for( int i = 0; i < MPI::GetSize(); i++ ) {
         if( i == MPI::GetRank() ) {
            str << "MPI rank:\t" << MPI::GetRank() << "\n"
                << "\tMesh dimension:\t" << getMeshDimension() << "\n"
                << "\tCell topology:\t" << getType( typename Cell::EntityTopology{} ) << "\n"
                << "\tCells count:\t" << cellsCount << "\n"
                << "\tFaces count:\t" << localMesh.template getEntitiesCount< Mesh::getMeshDimension() - 1 >() << "\n"
                << "\tvertices count:\t" << verticesCount << "\n"
                << "\tGhost levels:\t" << getGhostLevels() << "\n"
                << "\tGhost cells count:\t" << localMesh.template getGhostEntitiesCount< Mesh::getMeshDimension() >() << "\n"
                << "\tGhost faces count:\t" << localMesh.template getGhostEntitiesCount< Mesh::getMeshDimension() - 1 >() << "\n"
                << "\tGhost vertices count:\t" << localMesh.template getGhostEntitiesCount< 0 >() << "\n"
                << "\tBoundary cells count:\t" << localMesh.template getBoundaryIndices< Mesh::getMeshDimension() >().getSize() << "\n"
                << "\tBoundary faces count:\t" << localMesh.template getBoundaryIndices< Mesh::getMeshDimension() - 1 >().getSize() << "\n"
                << "\tBoundary vertices count:\t" << localMesh.template getBoundaryIndices< 0 >().getSize() << "\n";
            const GlobalIndexType globalPointIndices = getGlobalIndices< 0 >().getSize();
            const GlobalIndexType globalCellIndices = getGlobalIndices< Mesh::getMeshDimension() >().getSize();
            if( getGhostLevels() > 0 ) {
               if( globalPointIndices != verticesCount )
                  str << "ERROR: array of global point indices has wrong size: " << globalPointIndices << "\n";
               if( globalCellIndices != cellsCount )
                  str << "ERROR: array of global cell indices has wrong size: " << globalCellIndices << "\n";
               if( vtkPointGhostTypesArray.getSize() != verticesCount )
                  str << "ERROR: array of VTK point ghost types has wrong size: " << vtkPointGhostTypesArray.getSize() << "\n";
               if( vtkCellGhostTypesArray.getSize() != cellsCount )
                  str << "ERROR: array of VTK cell ghost types has wrong size: " << vtkCellGhostTypesArray.getSize() << "\n";
            }
            else {
               if( globalPointIndices > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of global point indices has non-zero size: " << globalPointIndices << "\n";
               if( globalCellIndices > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of global cell indices has non-zero size: " << globalCellIndices << "\n";
               if( vtkPointGhostTypesArray.getSize() > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of VTK point ghost types has non-zero size: " << vtkPointGhostTypesArray.getSize() << "\n";
               if( vtkCellGhostTypesArray.getSize() > 0 )
                  str << "WARNING: mesh has 0 ghost levels, but array of VTK cell ghost types has non-zero size: " << vtkCellGhostTypesArray.getSize() << "\n";
            }
            str.flush();
         }
         MPI::Barrier();
      }
   }

protected:
   MeshType localMesh;
   MPI_Comm communicator = MPI_COMM_NULL;
   int ghostLevels = 0;

   // vtkGhostType arrays for points and cells (cached for output into VTK formats)
   VTKTypesArrayType vtkPointGhostTypesArray, vtkCellGhostTypesArray;
};

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>
