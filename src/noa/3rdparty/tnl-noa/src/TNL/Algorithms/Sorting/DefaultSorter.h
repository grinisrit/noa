// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Sequential.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/BitonicSort.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/BubbleSort.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/Quicksort.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/STLSort.h>

namespace noa::TNL {
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

}  // namespace Sorting
}  // namespace Algorithms
}  // namespace noa::TNL
