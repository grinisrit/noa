// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include "HeatEquationSolverBenchmark.h"

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmarkParallelFor : public HeatEquationSolverBenchmark< Real, Device, Index >
{
   void exec( const Index xSize, const Index ySize )
   {
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      const Real hx_inv = 1.0 / (hx * hx);
      const Real hy_inv = 1.0 / (hy * hy);

      Real start = 0;
      Index iterations = 0;
      auto timestep = this->timeStep ? this->timeStep : 0.1 * std::min(hx * hx, hy * hy);
      while( start < this->finalTime && ( ! this->maxIterations || iterations < this->maxIterations ) )
      {
         auto uxView = this->ux.getView();
         auto auxView = this->aux.getView();
         auto next = [=] __cuda_callable__( Index i, Index j ) mutable
         {
            auto index = j * xSize + i;
            auto element = uxView[index];
            auto center = 2 * element;

            auxView[index] = element + ( (uxView[index - 1] -     center + uxView[index + 1]    ) * hx_inv +
                                         (uxView[index - xSize] - center + uxView[index + xSize]) * hy_inv   ) * timestep;
         };

         TNL::Algorithms::ParallelFor2D< Device >::exec( 1, 1, xSize - 1, ySize - 1, next );
         this->ux.swap( this->aux );
         start += timestep;
         iterations++;
      }
   }
};
