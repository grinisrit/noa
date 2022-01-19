// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Meshes {

template< typename T >
class isGrid
: public std::false_type
{};

template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
class isGrid< Grid< Dimension, Real, Device, Index > >
: public std::true_type
{};

template< typename T >
class isDistributedGrid
: public std::false_type
{};

template< int Dimension,
          typename Real,
          typename Device,
          typename Index >
class isDistributedGrid< DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > > >
: public std::true_type
{};

} // namespace Meshes
} // namespace TNL
