// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/MPI/Wrappers.h>
#include <noa/3rdparty/TNL/Algorithms/reduce.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

template< typename Expression >
auto DistributedExpressionMin( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::Min{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_MIN, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionArgMin( const Expression& expression )
-> std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType >
{
   using RealType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = std::pair< RealType, IndexType >;

   static_assert( std::numeric_limits< RealType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's real type" );
   ResultType result( -1, std::numeric_limits< RealType >::max() );
   const MPI_Comm communicator = expression.getCommunicator();
   if( communicator != MPI_COMM_NULL ) {
      // compute local argMin
      ResultType localResult = Algorithms::reduceWithArgument( expression.getConstLocalView(), noa::TNL::MinWithArg{} );
      // transform local index to global index
      localResult.second += expression.getLocalRange().getBegin();

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( communicator );
      ResultType dataForScatter[ nproc ];
      for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = localResult;
      ResultType gatheredResults[ nproc ];
      // NOTE: exchanging general data types does not work with MPI
      //MPI::Alltoall( dataForScatter, 1, gatheredResults, 1, communicator );
      MPI::Alltoall( (char*) dataForScatter, sizeof(ResultType), (char*) gatheredResults, sizeof(ResultType), communicator );

      // reduce the gathered data
      const auto* _data = gatheredResults;  // workaround for nvcc which does not allow to capture variable-length arrays (even in pure host code!)
      auto fetch = [_data] ( IndexType i ) { return _data[ i ].first; };
      result = Algorithms::reduceWithArgument< Devices::Host >( (IndexType) 0, (IndexType) nproc, fetch, noa::TNL::MinWithArg{} );
      result.second = gatheredResults[ result.second ].second;
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionMax( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::lowest();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::Max{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_MAX, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionArgMax( const Expression& expression )
-> std::pair< std::decay_t< decltype( expression[0] ) >, typename Expression::IndexType >
{
   using RealType = std::decay_t< decltype( expression[0] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = std::pair< RealType, IndexType >;

   static_assert( std::numeric_limits< RealType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's real type" );
   ResultType result( -1, std::numeric_limits< RealType >::lowest() );
   const MPI_Comm communicator = expression.getCommunicator();
   if( communicator != MPI_COMM_NULL ) {
      // compute local argMax
      ResultType localResult = Algorithms::reduceWithArgument( expression.getConstLocalView(), noa::TNL::MaxWithArg{} );
      // transform local index to global index
      localResult.second += expression.getLocalRange().getBegin();

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( communicator );
      ResultType dataForScatter[ nproc ];
      for( int i = 0; i < nproc; i++ ) dataForScatter[ i ] = localResult;
      ResultType gatheredResults[ nproc ];
      // NOTE: exchanging general data types does not work with MPI
      //MPI::Alltoall( dataForScatter, 1, gatheredResults, 1, communicator );
      MPI::Alltoall( (char*) dataForScatter, sizeof(ResultType), (char*) gatheredResults, sizeof(ResultType), communicator );

      // reduce the gathered data
      const auto* _data = gatheredResults;  // workaround for nvcc which does not allow to capture variable-length arrays (even in pure host code!)
      auto fetch = [_data] ( IndexType i ) { return _data[ i ].first; };
      result = Algorithms::reduceWithArgument< Devices::Host >( ( IndexType ) 0, (IndexType) nproc, fetch, noa::TNL::MaxWithArg{} );
      result.second = gatheredResults[ result.second ].second;
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionSum( const Expression& expression ) -> std::decay_t< decltype( expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::Plus{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_SUM, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionProduct( const Expression& expression ) -> std::decay_t< decltype( expression[0] * expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ) >;

   ResultType result = 1;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::Multiplies{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_PROD, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionLogicalAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] && expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] && expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::LogicalAnd{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_LAND, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionLogicalOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] || expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] || expression[0] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::LogicalOr{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_LOR, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionBinaryAnd( const Expression& expression ) -> std::decay_t< decltype( expression[0] & expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] & expression[0] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::BitAnd{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BAND, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionBinaryOr( const Expression& expression ) -> std::decay_t< decltype( expression[0] | expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] | expression[0] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::BitOr{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BOR, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto DistributedExpressionBinaryXor( const Expression& expression ) -> std::decay_t< decltype( expression[0] ^ expression[0] ) >
{
   using ResultType = std::decay_t< decltype( expression[0] ^ expression[0] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), noa::TNL::BitXor{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BXOR, expression.getCommunicator() );
   }
   return result;
}

} // namespace Expressions
} // namespace Containers
} // namespace noa::TNL
