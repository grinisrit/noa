// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Operators {

template< typename Grid >
class FiniteDifferences
{};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< Meshes::Grid< 1, Real, Device, Index > >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using GridType = Meshes::Grid< 1, Real, Device, Index >;
   // typedef typename GridType::CoordinatesType CoordinatesType;
   using CellType = typename GridType::Cell;

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType
   getDifference( const GridType& grid, const GridFunction& inFunction, GridFunction& outFunction );

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType
   getDifference( const GridType& grid, const CellType& cell, const GridFunction& function );
};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< Meshes::Grid< 2, Real, Device, Index > >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using GridType = Meshes::Grid< 2, Real, Device, Index >;
   // typedef typename GridType::CoordinatesType CoordinatesType;
   using CellType = typename GridType::Cell;

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType
   getDifference( const GridType& grid, const GridFunction& inFunction, GridFunction& outFunction );

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType
   getDifference( const GridType& grid, const CellType& cell, const GridFunction& function );
};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< Meshes::Grid< 3, Real, Device, Index > >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using GridType = Meshes::Grid< 3, Real, Device, Index >;
   // typedef typename GridType::CoordinatesType CoordinatesType;
   using CellType = typename GridType::Cell;

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType
   getDifference( const GridType& grid, const GridFunction& inFunction, GridFunction& outFunction );

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType
   getDifference( const GridType& grid, const CellType& cell, const GridFunction& function );
};

}  // namespace Operators
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Operators/FiniteDifferences_impl.h>
