// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Xuan Thang Nguyen, Tomas Oberhuber

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Array.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/detail/task.h>

namespace noa::TNL {
namespace Algorithms {
namespace Sorting {

template< typename Value, typename Device >
class Quicksorter;

template< typename Value >
class Quicksorter< Value, Devices::Cuda >
{
public:
   using ValueType = Value;
   using DeviceType = Devices::Cuda;

   template< typename Array, typename Compare >
   void
   sort( Array& arr, const Compare& cmp );

   template< typename Array >
   void
   sort( Array& arr );

protected:
   void
   init( Containers::ArrayView< Value, Devices::Cuda > arr,
         int gridDim,
         int blockDim,
         int desiredElemPerBlock,
         int maxSharable );

   template< typename CMP >
   void
   performSort( const CMP& Cmp );

   /**
    * returns how many blocks are needed to start sort phase 1 if @param elemPerBlock were to be used
    * */
   int
   getSetsNeeded( int elemPerBlock ) const;

   /**
    * returns the optimal amount of elements per thread needed for phase
    * */
   int
   getElemPerBlock() const;

   /**
    * returns the amount of blocks needed to start phase 1 while also initializing all tasks
    * */
   template< typename CMP >
   int
   initTasks( int elemPerBlock, const CMP& Cmp );

   /**
    * does the 1st phase of Quicksort until out of task memory or each task is small enough
    * for correctness, secondphase method needs to be called to sort each subsequences
    * */
   template< typename CMP >
   void
   firstPhase( const CMP& Cmp );

   /**
    * update necessary variables after 1 phase1 sort
    * */
   void
   processNewTasks();

   /**
    * sorts all leftover tasks
    * */
   template< typename CMP >
   void
   secondPhase( const CMP& Cmp );

   int maxBlocks, threadsPerBlock, desiredElemPerBlock, maxSharable;  // kernel config

   Containers::Array< Value, Devices::Cuda > auxMem;
   Containers::ArrayView< Value, Devices::Cuda > arr, aux;

   int desired_2ndPhasElemPerBlock;
   const int g_maxTasks = 1 << 14;
   int maxTasks;

   Containers::Array< TASK, Devices::Cuda > cuda_tasks, cuda_newTasks,
      cuda_2ndPhaseTasks;  // 1 set of 2 rotating tasks and 2nd phase
   Containers::Array< int, Devices::Cuda > cuda_newTasksAmount, cuda_2ndPhaseTasksAmount;  // is in reality 1 integer each

   Containers::Array< int, Devices::Cuda > cuda_blockToTaskMapping;
   Containers::Array< int, Devices::Cuda > cuda_reductionTaskInitMem;

   int host_1stPhaseTasksAmount = 0, host_2ndPhaseTasksAmount = 0;
   int iteration = 0;

   template< typename T >
   friend int
   getSetsNeededFunction( int elemPerBlock, const Quicksorter< T, Devices::Cuda >& quicksort );
};

}  // namespace Sorting
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Sorting/detail/Quicksorter.hpp>
