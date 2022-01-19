// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Meshes {

/**
 * \brief Minimal class usable as \e Mesh in the \ref VTIWriter.
 *
 * It does not have a \e Device template argument, because it does not describe
 * a data structure - it contains only the information describing the domain of
 * the grid (i.e., the metadata of the \e ImageData tag in the VTI file
 * format). Note that VTK supports only 1, 2, or 3-dimensional grids.
 */
template< int Dimension, typename Real, typename Index >
class NDMetaGrid
{
public:
   using RealType = Real;
   using GlobalIndexType = Index;
   using PointType = Containers::StaticVector< Dimension, Real >;
   using CoordinatesType = Containers::StaticVector< Dimension, Index >;

   //! \brief Returns the spatial dimension of the grid.
   static constexpr int getMeshDimension()
   {
      return Dimension;
   }

   //! \brief Sets the grid dimensions/size.
   void setDimensions( const CoordinatesType& dimensions )
   {
      this->dimensions = dimensions;
   }

   //! \brief Returns the grid dimensions/size.
   const CoordinatesType& getDimensions() const
   {
      return dimensions;
   }

   //! \brief Sets the origin of the grid (coordinates of the left bottom front
   //! corner).
   void setOrigin( const PointType& origin )
   {
      this->origin = origin;
   }

   //! \brief Returns the origin of the grid (coordinates of the left bottom
   //! front corner).
   const PointType& getOrigin() const
   {
      return origin;
   }

   //! \brief Sets the domain of the grid (i.e., the origin and
   //! proportions/length). Note that space steps are computed using the
   //! current grid dimensions.
   void setDomain( const PointType& origin,
                   const PointType& proportions )
   {
      this->origin = origin;
      this->spaceSteps = proportions / dimensions;
   }

   //! \brief Sets the space steps of the grid, i.e. the parameters usually
   //! denoted as \e hx, \e hy, \e hz.
   void setSpaceSteps( const PointType& spaceSteps )
   {
      this->spaceSteps = spaceSteps;
   }

   //! \brief Returns the space steps of the grid.
   const PointType& getSpaceSteps() const
   {
      return spaceSteps;
   }

protected:
   CoordinatesType dimensions = 0;

   PointType origin = 0;

   PointType spaceSteps = 0;
};

} // namespace Meshes
} // namespace TNL
