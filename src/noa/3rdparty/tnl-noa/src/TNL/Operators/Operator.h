// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh,
          Functions::DomainType DomainType = Functions::MeshInteriorDomain,
          int PreimageEntitiesDimension = Mesh::getMeshDimension(),
          int ImageEntitiesDimension = Mesh::getMeshDimension(),
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class Operator : public Functions::Domain< Mesh::getMeshDimension(), DomainType >
{
public:
   using MeshType = Mesh;
   using MeshRealType = typename MeshType::RealType;
   using DeviceType = typename MeshType::DeviceType;
   using MeshIndexType = typename MeshType::GlobalIndexType;
   using RealType = Real;
   using IndexType = Index;
   using ExactOperatorType = void;

   constexpr static int
   getMeshDimension()
   {
      return MeshType::getMeshDimension();
   }
   constexpr static int
   getPreimageEntitiesDimension()
   {
      return PreimageEntitiesDimension;
   }
   constexpr static int
   getImageEntitiesDimension()
   {
      return ImageEntitiesDimension;
   }

   bool
   refresh( const RealType& time = 0.0 )
   {
      return true;
   }

   bool
   deepRefresh( const RealType& time = 0.0 )
   {
      return true;
   }

   template< typename MeshFunction >
   void
   setPreimageFunction( const MeshFunction& f )
   {}
};

}  // namespace Operators
}  // namespace noa::TNL
