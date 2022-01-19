// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/Sorting/BubbleSort.h>
#include <TNL/Algorithms/Sorting/Quicksort.h>
#include <TNL/Algorithms/Sorting/STLSort.h>

namespace TNL {
   namespace Algorithms {
      namespace Sorting {

template< typename Device >
struct DefaultSorter;

template<>
struct DefaultSorter< Devices::Sequential >
{
   using SorterType = Algorithms::Sorting::STLSort;
};

template<>
struct DefaultSorter< Devices::Host >
{
   using SorterType = Algorithms::Sorting::STLSort;
};

template<>
struct DefaultSorter< Devices::Cuda >
{
   using SorterType = Algorithms::Sorting::Quicksort;
};

template< typename Device >
struct DefaultInplaceSorter;

template<>
struct DefaultInplaceSorter< Devices::Sequential >
{
   using SorterType = Algorithms::Sorting::BubbleSort;
};

template<>
struct DefaultInplaceSorter< Devices::Host >
{
   using SorterType = Algorithms::Sorting::BubbleSort;
};

template<>
struct DefaultInplaceSorter< Devices::Cuda >
{
   using SorterType = Algorithms::Sorting::BitonicSort;
};

      } // namespace Sorting
   } // namespace Algorithms
} // namespace TNL
