// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/ParallelFor.h>


namespace TNL {
   namespace Algorithms {

/**
 * \brief Wrapper to ParallelFor which makes it run sequentially.
 *
 *  It is helpfull for debuging or just sequential for loops on GPUs.
 */
template< typename Device = Devices::Sequential >
struct SequentialFor
{
   /**
    * \brief Static method for execution of the loop.
    *
    * \tparam Index defines the type of indexes over which the loop iterates.
    * \tparam Function is the type of function to be called in each iteration.
    *
    * \param start the for-loop iterates over index interval [start, end).
    * \param end the for-loop iterates over index interval [start, end).
    * \param f is the function to be called in each iteration
    *
    * \par Example
    * \include Algorithms/SequentialForExample.cpp
    * \par Output
    * \include SequentialForExample.out
    *
    */
   template< typename Index,
             typename Function >
   static void exec( Index start, Index end, Function f )
   {
      for( Index i = start; i < end; i++ )
         ParallelFor< Device >::exec( i, i + 1, f );
   }
};


   } // namespace Algorithms
} // namespace TNL