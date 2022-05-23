// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <iostream>

#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrix.h>

#include "Comm.h"
#include "Utils.h"

namespace noa::TNL {
namespace MPI {

/**
 * \brief Returns a matrix of communication costs between each pair of ranks as
 * measured by a microbenchmark using a given message size and number of
 * iterations.
 */
template< typename Device >
auto
measureAlltoallCommunicationCost( const MPI::Comm& communicator, int messageSize = 1e7, int iterations = 10 )
{
   using ValueType = int;
   using Vector = TNL::Containers::Vector< ValueType, Device, int >;
   using Matrix = TNL::Matrices::DenseMatrix< double, TNL::Devices::Sequential, int >;

   // initialize communication buffers
   Vector send_buf( messageSize );
   Vector recv_buf( messageSize );

   const int rank = communicator.rank();
   const int nproc = communicator.size();

   // cost matrix: symmetric, normalized, proportional to timings
   Matrix cost;
   cost.setDimensions( nproc, nproc );

   double min_time = 1e9;

   for( int src = 0; src < nproc; src++ ) {
      for( int dest = src; dest < nproc; dest++ ) {
         // skip deadlock
         if( src == dest )
            continue;

         // warm up
         if( src == rank )
            MPI::send( send_buf, dest, 0, communicator );
         else if( dest == rank )
            MPI::recv( recv_buf, src, 0, communicator );

         Timer t;
         t.start();
         for( int i = 1; i <= iterations; i++ ) {
            if( src == rank )
               MPI::send( send_buf, dest, 0, communicator );
            else if( dest == rank )
               MPI::recv( recv_buf, src, 0, communicator );
         }
         t.stop();

         // make sure all ranks get the same time
         const double time = MPI::bcast( t.getRealTime(), src, communicator );

         // set matrix elements
         cost( src, dest ) = cost( dest, src ) = time;

         // keep the minimal time for normalization
         min_time = TNL::min( min_time, time );
      }
   }

   // normalize
   for( int src = 0; src < nproc; src++ )
      for( int dest = 0; dest < nproc; dest++ )
         cost( src, dest ) = std::round( cost( src, dest ) / min_time );

   return cost;
}

/**
 * \brief Returns a vector of communication costs per each rank for given cost
 * matrix, communication pattern and permutation of the ranks.
 */
template< typename CostMatrix, typename CommPattern, typename Permutation >
TNL::Containers::Vector< double, TNL::Devices::Sequential, int >
getCommunicationCosts( const CostMatrix& cost, const CommPattern& pattern, const Permutation& perm )
{
   using Vector = TNL::Containers::Vector< double, TNL::Devices::Sequential, int >;

   const int size = perm.getSize();
   if( cost.getRows() != size || cost.getColumns() != size || pattern.getRows() != size || pattern.getColumns() != size )
      throw std::invalid_argument( "invalid size of the input matrix" );

   Vector rank_costs( size );
   rank_costs = 0;
   for( int j = 0; j < size; j++ )
      for( int i = 0; i < size; i++ )
         rank_costs[ perm[ j ] ] += cost( j, i ) * pattern( perm[ j ], perm[ i ] );

   return rank_costs;
}

/**
 * \brief Solves the quadratic assignment problem heuristically using the
 * improvement method.
 *
 * [Quadratic assignment problem](https://en.wikipedia.org/wiki/Quadratic_assignment_problem)
 * is an NP-hard problem, so there is no known algorithm for solving this
 * problem in polynomial time. This function finds a sub-optimal solution in
 * polynomial time ($O(n^3)$) using a heuristic algorithm.
 *
 * The algorithm does not check all permutations, only all transpositions are
 * checked and those that improve the cost are considered. This allows to skip
 * many permutations and guarantees improvement of the cost (in the worst case,
 * the same cost is obtained).
 *
 * See the following paper for more details about the algorithm:
 * - B. Brandfass, T. Alrutz, T. Gerhold - Rank reordering for MPI communication
 *   optimization, Computers & Fluids 80 (2013) 372–380.
 *   https://doi.org/10.1016/j.compfluid.2012.01.019
 *
 * Notes:
 * - Transpositions can be represented by a pair of integers $(i, j)$ such that
 *   $i < j$.
 * - Any permutation can be expressed uniquely as a product of transpositions
 *   that commute with each other.
 * - Reference: https://en.wikipedia.org/wiki/Transposition_(mathematics)
 */
template< typename CostMatrix, typename CommPattern >
Containers::Vector< int, TNL::Devices::Sequential, int >
solveQuadraticAssignmentProblem( int nproc, const CostMatrix& costMatrix, const CommPattern& pattern )
{
   using Vector = Containers::Vector< int, TNL::Devices::Sequential, int >;

   // start with identity permutation
   Vector perm( nproc );
   for( int i = 0; i < nproc; i++ )
      perm[ i ] = i;

   double current_cost = TNL::sum( getCommunicationCosts( costMatrix, pattern, perm ) );

   // generate transpositions
   for( int i = 0; i < nproc; i++ ) {
      for( int j = i + 1; j < nproc; j++ ) {
         // check if the transposition is better
         std::swap( perm[ i ], perm[ j ] );
         const double cost = TNL::sum( getCommunicationCosts( costMatrix, pattern, perm ) );
         if( cost < current_cost ) {
            // keep the permutation
            current_cost = cost;
            if( TNL::MPI::GetRank( MPI_COMM_WORLD ) == 0 ) {
               std::cout << "permutation " << perm << " cost " << cost << std::endl;
            }
         }
         else {
            // restore previous permutation
            std::swap( perm[ i ], perm[ j ] );
         }
      }
   }

   return perm;
}

/**
 * \brief Returns a communicator comprising the same group of processes as the
 * given communicator, but ranks are reordered to minimize the communication
 * cost for the given communication pattern.
 *
 * The communication costs are determined by \ref
 * measureAlltoallCommunicationCost and the sub-optimal permutation of ranks
 * is found heuristically using \ref solveQuadraticAssignmentProblem.
 */
template< typename Device, typename CommPattern >
MPI::Comm
optimizeRanks( const MPI::Comm& communicator, const CommPattern& communicationPattern )
{
   const int rank = communicator.rank();
   const int nproc = communicator.size();

   if( communicationPattern.getRows() != nproc || communicationPattern.getColumns() != nproc )
      throw std::invalid_argument( "the size of the communicationPattern matrix does not match the given communicator" );

   // determine communication cost between each pair of processes
   const auto costMatrix = measureAlltoallCommunicationCost< Device >( communicator );

   if( rank == 0 ) {
      using Vector = TNL::Containers::Vector< double, TNL::Devices::Sequential, int >;
      Vector identity( nproc );
      for( int i = 0; i < nproc; i++ )
         identity[ i ] = i;

      std::cout << "cost matrix:\n" << costMatrix << std::endl;
      const auto cost = getCommunicationCosts( costMatrix, communicationPattern, identity );
      std::cout << "initial cost vector: " << cost << " sum " << TNL::sum( cost ) << std::endl;
   }

   // get permutation by solving the quadratic assignment problem
   const auto perm = solveQuadraticAssignmentProblem( nproc, costMatrix, communicationPattern );

   if( rank == 0 ) {
      const auto rank_costs = getCommunicationCosts( costMatrix, communicationPattern, perm );
      std::cout << "restored best permutation " << perm << " with cost vector " << rank_costs << " sum "
                << TNL::sum( rank_costs ) << std::endl;
   }

   // create a communicator comprising all processes, but with permuted ranks
   return communicator.split( 0, perm[ rank ] );
}

}  // namespace MPI
}  // namespace noa::TNL
