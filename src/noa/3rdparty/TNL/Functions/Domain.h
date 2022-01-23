// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT


#pragma once

namespace noa::TNL {
namespace Functions {   

enum DomainType { NonspaceDomain, SpaceDomain, MeshDomain, MeshInteriorDomain, MeshBoundaryDomain };

template< int Dimension,
          DomainType DomainType_ = SpaceDomain >
class Domain
{
   public:
 
      typedef void DeviceType;
 
      static constexpr int getDomainDimension() { return Dimension; }
 
      static constexpr DomainType getDomainType() { return DomainType_; }
};

} // namespace Functions
} // namespace noa::TNL

