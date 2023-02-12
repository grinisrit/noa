// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/VTKTraits.h>

namespace noa::TNL {
namespace Meshes {
namespace VTK {

template< EntityShape GeneralShape >
struct EntityShapeGroup
{};

template< EntityShape GeneralShape, int index >
struct EntityShapeGroupElement
{};

template<>
struct EntityShapeGroup< EntityShape::Polygon >
{
   static constexpr int size = 2;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polygon, 0 >
{
   static constexpr EntityShape shape = EntityShape::Triangle;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polygon, 1 >
{
   static constexpr EntityShape shape = EntityShape::Quad;
};

template<>
struct EntityShapeGroup< EntityShape::Polyhedron >
{
   static constexpr int size = 6;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polyhedron, 0 >
{
   static constexpr EntityShape shape = EntityShape::Tetra;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polyhedron, 1 >
{
   static constexpr EntityShape shape = EntityShape::Hexahedron;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polyhedron, 2 >
{
   static constexpr EntityShape shape = EntityShape::Wedge;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polyhedron, 3 >
{
   static constexpr EntityShape shape = EntityShape::Pyramid;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polyhedron, 4 >
{
   static constexpr EntityShape shape = EntityShape::PentagonalPrism;
};

template<>
struct EntityShapeGroupElement< EntityShape::Polyhedron, 5 >
{
   static constexpr EntityShape shape = EntityShape::HexagonalPrism;
};

}  // namespace VTK
}  // namespace Meshes
}  // namespace noa::TNL
