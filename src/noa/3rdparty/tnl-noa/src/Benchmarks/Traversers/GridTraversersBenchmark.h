// Implemented by: Tomas Oberhuber

#pragma once

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< int Dimension,
          typename Device,
          typename Real,
          typename Index >
class GridTraversersBenchmark{};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL

#include "GridTraversersBenchmark_1D.h"
#include "GridTraversersBenchmark_2D.h"
#include "GridTraversersBenchmark_3D.h"
