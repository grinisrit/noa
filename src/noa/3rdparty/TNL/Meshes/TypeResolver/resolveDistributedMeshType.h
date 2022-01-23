// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Mesh.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace noa::TNL {
namespace Meshes {

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveDistributedMeshType( Functor&& functor,
                            const std::string& fileName,
                            const std::string& fileFormat = "auto" );

template< typename ConfigTag,
          typename Device,
          typename Functor >
bool
resolveAndLoadDistributedMesh( Functor&& functor,
                               const std::string& fileName,
                               const std::string& fileFormat = "auto" );

template< typename Mesh >
bool
loadDistributedMesh( DistributedMeshes::DistributedMesh< Mesh >& distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat = "auto" );

} // namespace Meshes
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Meshes/TypeResolver/resolveDistributedMeshType.hpp>
