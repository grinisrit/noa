// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Problems/Problem.h>
#include <noa/3rdparty/TNL/Problems/CommonData.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>
#include <noa/3rdparty/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/SlicedEllpack.h>
#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace noa::TNL {
namespace Problems {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Device = typename Mesh::DeviceType,
          typename Index = typename Mesh::GlobalIndexType >
class PDEProblem : public Problem< Real, Device, Index >
{
   public:

      using BaseType = Problem< Real, Device, Index >;
      using typename BaseType::RealType;
      using typename BaseType::DeviceType;
      using typename BaseType::IndexType;

      using MeshType = Mesh;
      using MeshPointer = Pointers::SharedPointer< MeshType, DeviceType >;
      using DistributedMeshType = Meshes::DistributedMeshes::DistributedMesh< MeshType >;
      using DistributedMeshPointer = Pointers::SharedPointer< DistributedMeshType, DeviceType >;
      using SubdomainOverlapsType = typename DistributedMeshType::SubdomainOverlapsType;
      using DofVectorType = Containers::Vector< RealType, DeviceType, IndexType>;
      using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
      template< typename _Device, typename _Index, typename _IndexAlocator >
      using SegmentsType = Algorithms::Segments::SlicedEllpack< _Device, _Index, _IndexAlocator >;
      using MatrixType = noa::TNL::Matrices::SparseMatrix< Real,
                                                      Device,
                                                      Index,
                                                      noa::TNL::Matrices::GeneralMatrix,
                                                      SegmentsType
                                                    >;
      using CommonDataType = CommonData;
      using CommonDataPointer = Pointers::SharedPointer< CommonDataType, DeviceType >;

      static constexpr bool isTimeDependent() { return true; };

      /****
       * This means that the time stepper will be set from the command line arguments.
       */
      typedef void TimeStepper;

      String getPrologHeader() const;

      void writeProlog( Logger& logger,
                        const Config::ParameterContainer& parameters ) const;

      bool writeEpilog( Logger& logger ) const;

      void setMesh( MeshPointer& meshPointer );

      void setMesh( DistributedMeshPointer& distributedMeshPointer );

      const MeshPointer& getMesh() const;

      MeshPointer& getMesh();

      const DistributedMeshPointer& getDistributedMesh() const;

      DistributedMeshPointer& getDistributedMesh();

      void setCommonData( CommonDataPointer& commonData );

      const CommonDataPointer& getCommonData() const;

      CommonDataPointer& getCommonData();

      // Width of the subdomain overlaps in case when all of them are the same
      virtual IndexType subdomainOverlapSize();

      // Returns default subdomain overlaps i.e. no overlaps on the boundaries, only
      // in the domain interior.
      void getSubdomainOverlaps( const Config::ParameterContainer& parameters,
                                 const String& prefix,
                                 const MeshType& mesh,
                                 SubdomainOverlapsType& lower,
                                 SubdomainOverlapsType& upper );

      bool preIterate( const RealType& time,
                       const RealType& tau,
                       DofVectorPointer& dofs );

      void applyBoundaryConditions( const RealType& time,
                                       DofVectorPointer& dofs );

      template< typename Matrix >
      void saveFailedLinearSystem( const Matrix& matrix,
                                   const DofVectorType& dofs,
                                   const DofVectorType& rightHandSide ) const;

      bool postIterate( const RealType& time,
                        const RealType& tau,
                        DofVectorPointer& dofs );

      Solvers::SolverMonitor* getSolverMonitor();

      MeshPointer meshPointer;

      DistributedMeshPointer distributedMeshPointer;

      CommonDataPointer commonDataPointer;
};

} // namespace Problems
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Problems/PDEProblem_impl.h>
