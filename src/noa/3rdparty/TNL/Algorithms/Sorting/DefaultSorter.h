// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber

#pragma once

#include <noa/3rdparty/TNL/Devices/Sequential.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Algorithms/Sorting/BitonicSort.h>
#include <noa/3rdparty/TNL/Algorithms/Sorting/BubbleSort.h>
#include <noa/3rdparty/TNL/Algorithms/Sorting/Quicksort.h>
#include <noa/3rdparty/TNL/Algorithms/Sorting/STLSort.h>

namespace noaTNL {
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
} // namespace noaTNL
