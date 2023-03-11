// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functions/MeshFunction.h>

namespace noa::TNL {
namespace Functions {

// BoundaryMeshFunction is supposed to store values of a mesh functions only
// at boundary mesh entities. It is just a small memory optimization.
// Currently, it is only a wrap around common MeshFunction so that we can introduce
// boundary mesh functions in the rest of the code.
// TODO: Implement it.
template< typename Mesh, int MeshEntityDimension = Mesh::getMeshDimension(), typename Real = typename Mesh::RealType >
class BoundaryMeshFunction : public MeshFunction< Mesh, MeshEntityDimension, Real >
{
public:
   using BaseType = MeshFunction< Mesh, MeshEntityDimension, Real >;
   using typename BaseType::DeviceType;
   using typename BaseType::IndexType;
   using typename BaseType::MeshPointer;
   using typename BaseType::MeshType;
   using typename BaseType::RealType;
   using typename BaseType::VectorType;
};

}  // namespace Functions
}  // namespace noa::TNL
