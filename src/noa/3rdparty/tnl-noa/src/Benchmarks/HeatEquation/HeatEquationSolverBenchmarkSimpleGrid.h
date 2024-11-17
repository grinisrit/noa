// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Containers/StaticVector.h>
#include "HeatEquationSolverBenchmark.h"

template<int Size, typename Index, typename Real>
class Entity;

template <size_t Value, size_t Power>
constexpr size_t pow()
{
   size_t result = 1;

   for (size_t i = 0; i < Power; i++) {
      result *= Value;
   }

   return result;
}

constexpr int spaceStepsPowers = 5;

template<int Size, typename Index, typename Real>
class Grid
{
   public:
      Grid() = default;

      Grid( const TNL::Containers::StaticVector<Size, Index>& dim )
      : dimensions( dim ),
        entitiesCountAlongNormals( 0 ),
        cumulativeEntitiesCountAlongNormals( 0 ),
        origin( 0 ),
        proportions( 0 ),
        spaceSteps( 0 ),
        spaceProducts( 0 )
      {}

      __cuda_callable__ inline
      Entity<Size, Index, Real> getEntity(Index i, Index j) const {
         Entity<Size, Index, Real> entity(*this);

         entity.coordinates.x() = i;
         entity.coordinates.y() = j;
         entity.index = j * dimensions.x() + i;

         return entity;
      }

      TNL::Containers::StaticVector<Size, Index> dimensions;
      TNL::Containers::StaticVector<1 << Size, Index> entitiesCountAlongNormals;
      TNL::Containers::StaticVector<Size + 1, Index> cumulativeEntitiesCountAlongNormals;

      TNL::Containers::StaticVector<Size, Real> origin, proportions, spaceSteps;

      TNL::Containers::StaticVector<std::integral_constant<Index, pow<spaceStepsPowers, Size>()>::value, Real> spaceProducts;
};

template<int Size, typename Index, typename Real>
class Entity
{
   public:
      __cuda_callable__ inline
      Entity( const Grid<Size, Index, Real>& grid )
      : grid( grid )
      {}

      const Grid<Size, Index, Real>& grid;

      Index index;
      //Index orientation;
      TNL::Containers::StaticVector<Size, Index> coordinates;
      //TNL::Containers::StaticVector<Size, Index> normals;
};

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkSimpleGrid : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);

      Grid<2, int, Real> grid( {xSize, ySize} );

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min(hx * hx, hy * hy);
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [=] __cuda_callable__(int i, int j) mutable {
            auto entity = grid.getEntity(i, j);

            auto index = entity.index;
            auto element = uxView[index];
            auto center = 2 * element;

            auxView[index] = element + ((uxView[index - 1] - center + uxView[index + 1]) * hx_inv +
                                        (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv) * timestep;
         };
         TNL::Algorithms::ParallelFor2D<Device>::exec( 1, 1, xSize - 1, ySize - 1, next);
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }
};
