// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/File.h>
#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Functions/Domain.h>
#include <noa/3rdparty/TNL/Pointers/SharedPointer.h>

namespace noaTNL {
namespace Functions {

template< typename Mesh,
          int MeshEntityDimension = Mesh::getMeshDimension(),
          typename Real = typename Mesh::RealType >
class MeshFunction :
   public Domain< Mesh::getMeshDimension(), MeshDomain >
{
   //static_assert( Mesh::DeviceType::DeviceType == Vector::DeviceType::DeviceType,
   //               "Both mesh and vector of a mesh function must reside on the same device.");
   public:

      using MeshType = Mesh;
      using DeviceType = typename MeshType::DeviceType;
      using IndexType = typename MeshType::GlobalIndexType;
      using MeshPointer = Pointers::SharedPointer< MeshType >;
      using RealType = Real;
      using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

      static constexpr int getEntitiesDimension() { return MeshEntityDimension; }

      static constexpr int getMeshDimension() { return MeshType::getMeshDimension(); }

      MeshFunction();

      MeshFunction( const MeshPointer& meshPointer );

      MeshFunction( const MeshFunction& meshFunction );

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" );

      bool setup( const MeshPointer& meshPointer,
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" );

      void setMesh( const MeshPointer& meshPointer );

      template< typename Device = Devices::Host >
      __cuda_callable__
      const MeshType& getMesh() const;

      const MeshPointer& getMeshPointer() const;

      MeshPointer& getMeshPointer();

      static IndexType getDofs( const MeshPointer& meshPointer );

      __cuda_callable__ const VectorType& getData() const;

      __cuda_callable__ VectorType& getData();

      bool refresh( const RealType& time = 0.0 ) const;

      bool deepRefresh( const RealType& time = 0.0 ) const;

      template< typename EntityType >
      RealType getValue( const EntityType& meshEntity ) const;

      template< typename EntityType >
      void setValue( const EntityType& meshEntity,
                     const RealType& value );

      template< typename EntityType >
      __cuda_callable__
      RealType& operator()( const EntityType& meshEntity,
                            const RealType& time = 0 );

      template< typename EntityType >
      __cuda_callable__
      const RealType& operator()( const EntityType& meshEntity,
                                  const RealType& time = 0 ) const;

      __cuda_callable__
      RealType& operator[]( const IndexType& meshEntityIndex );

      __cuda_callable__
      const RealType& operator[]( const IndexType& meshEntityIndex ) const;

      MeshFunction& operator = ( const MeshFunction& f );

      template< typename Function >
      MeshFunction& operator = ( const Function& f );

      template< typename Function >
      MeshFunction& operator -= ( const Function& f );

      template< typename Function >
      MeshFunction& operator += ( const Function& f );

      RealType getLpNorm( const RealType& p ) const;

      RealType getMaxNorm() const;

      bool write( const std::string& functionName,
                  const std::string& fileName,
                  const std::string& fileFormat = "auto" ) const;

   protected:

      MeshPointer meshPointer;

      VectorType data;

      template< typename, typename > friend class MeshFunctionEvaluator;
};

template< typename Mesh,
          int MeshEntityDimension,
          typename Real >
std::ostream& operator << ( std::ostream& str, const MeshFunction< Mesh, MeshEntityDimension, Real >& f );

} // namespace Functions
} // namespace noaTNL

#include <noa/3rdparty/TNL/Functions/MeshFunction.hpp>
