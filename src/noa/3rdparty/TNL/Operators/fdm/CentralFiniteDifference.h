// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Operators/fdm/FiniteDifferences.h>
#include <TNL/Operators/fdm/ExactDifference.h>
#include <TNL/Operators/Operator.h>

namespace TNL {
namespace Operators {   

template< typename Mesh,
          int Xdifference = 0,
          int YDifference = 0,
          int ZDifference = 0,
          typename RealType = typename Mesh::RealType,
          typename IndexType = typename Mesh::GlobalIndexType >
class CentralFiniteDifference
{
};

template< int Dimension,
          typename MeshReal,
          typename MeshDevice,
          typename MeshIndex,
          int XDifference,
          int YDifference,
          int ZDifference,
          typename Real,
          typename Index >
class CentralFiniteDifference< Meshes::Grid< Dimension, MeshReal, MeshDevice, MeshIndex >, XDifference, YDifference, ZDifference, Real, Index >
: public Operator< Meshes::Grid< Dimension, MeshReal, MeshDevice, MeshIndex >,
                      Functions::MeshInteriorDomain, Dimension, Dimension, Real, Index >
{
   public:
 
      typedef Meshes::Grid< Dimension, MeshReal, MeshDevice, MeshIndex > MeshType;
      typedef Real RealType;
      typedef MeshDevice DeviceType;
      typedef Index IndexType;
      typedef ExactDifference< Dimension, XDifference, YDifference, ZDifference > ExactOperatorType;
 
      //static constexpr int getMeshDimension() { return Dimension; }
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      inline Real operator()( const MeshFunction& u,
                              const MeshEntity& entity,
                              const RealType& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntitiesDimension() == Dimension,
            "Finite differences can be evaluate only on mesh cells, i.e. the dimensions count of the mesh entities of mesh function must be the same as mesh dimensions count." );
         return FiniteDifferences< MeshType, Real, Index, XDifference, YDifference, ZDifference, 0, 0, 0 >::getValue( u, entity );
      };
};

} // namespace Operators
} // namespace TNL

