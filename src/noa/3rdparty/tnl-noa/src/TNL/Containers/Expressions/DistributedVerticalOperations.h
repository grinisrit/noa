// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>
#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Wrappers.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/reduce.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

template< typename Expression >
auto
DistributedExpressionMin( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Min{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_MIN, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionArgMin( const Expression& expression )
   -> std::pair< std::decay_t< decltype( expression[ 0 ] ) >, typename Expression::IndexType >
{
   using RealType = std::decay_t< decltype( expression[ 0 ] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = std::pair< RealType, IndexType >;

   static_assert( std::numeric_limits< RealType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's real type" );
   ResultType result( -1, std::numeric_limits< RealType >::max() );
   const MPI::Comm& communicator = expression.getCommunicator();
   if( communicator != MPI_COMM_NULL ) {
      // compute local argMin
      ResultType localResult = Algorithms::reduceWithArgument( expression.getConstLocalView(), TNL::MinWithArg{} );
      // transform local index to global index
      localResult.second += expression.getLocalRange().getBegin();

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( communicator );
      std::unique_ptr< ResultType[] > dataForScatter{ new ResultType[ nproc ] };
      for( int i = 0; i < nproc; i++ )
         dataForScatter[ i ] = localResult;
      std::unique_ptr< ResultType[] > gatheredResults{ new ResultType[ nproc ] };
      // NOTE: exchanging general data types does not work with MPI
      // MPI::Alltoall( dataForScatter.get(), 1, gatheredResults.get(), 1, communicator );
      MPI::Alltoall( (char*) dataForScatter.get(),
                     sizeof( ResultType ),
                     (char*) gatheredResults.get(),
                     sizeof( ResultType ),
                     communicator );

      auto fetch = [ &gatheredResults ]( IndexType i )
      {
         return gatheredResults[ i ].first;
      };
      result = Algorithms::reduceWithArgument< Devices::Host >( (IndexType) 0, (IndexType) nproc, fetch, TNL::MinWithArg{} );
      result.second = gatheredResults[ result.second ].second;
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionMax( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::lowest();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Max{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_MAX, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionArgMax( const Expression& expression )
   -> std::pair< std::decay_t< decltype( expression[ 0 ] ) >, typename Expression::IndexType >
{
   using RealType = std::decay_t< decltype( expression[ 0 ] ) >;
   using IndexType = typename Expression::IndexType;
   using ResultType = std::pair< RealType, IndexType >;

   static_assert( std::numeric_limits< RealType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's real type" );
   ResultType result( -1, std::numeric_limits< RealType >::lowest() );
   const MPI::Comm& communicator = expression.getCommunicator();
   if( communicator != MPI_COMM_NULL ) {
      // compute local argMax
      ResultType localResult = Algorithms::reduceWithArgument( expression.getConstLocalView(), TNL::MaxWithArg{} );
      // transform local index to global index
      localResult.second += expression.getLocalRange().getBegin();

      // scatter local result to all processes and gather their results
      const int nproc = MPI::GetSize( communicator );
      std::unique_ptr< ResultType[] > dataForScatter{ new ResultType[ nproc ] };
      for( int i = 0; i < nproc; i++ )
         dataForScatter[ i ] = localResult;
      std::unique_ptr< ResultType[] > gatheredResults{ new ResultType[ nproc ] };
      // NOTE: exchanging general data types does not work with MPI
      // MPI::Alltoall( dataForScatter.get(), 1, gatheredResults.get(), 1, communicator );
      MPI::Alltoall( (char*) dataForScatter.get(),
                     sizeof( ResultType ),
                     (char*) gatheredResults.get(),
                     sizeof( ResultType ),
                     communicator );

      auto fetch = [ &gatheredResults ]( IndexType i )
      {
         return gatheredResults[ i ].first;
      };
      result = Algorithms::reduceWithArgument< Devices::Host >( (IndexType) 0, (IndexType) nproc, fetch, TNL::MaxWithArg{} );
      result.second = gatheredResults[ result.second ].second;
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionSum( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Plus{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_SUM, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionProduct( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] * expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] ) >;

   ResultType result = 1;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::Multiplies{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_PROD, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionLogicalAnd( const Expression& expression )
   -> std::decay_t< decltype( expression[ 0 ] && expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] && expression[ 0 ] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::LogicalAnd{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_LAND, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionLogicalOr( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] || expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] || expression[ 0 ] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::LogicalOr{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_LOR, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionBinaryAnd( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] & expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] & expression[ 0 ] ) >;

   static_assert( std::numeric_limits< ResultType >::is_specialized,
                  "std::numeric_limits is not specialized for the reduction's result type" );
   ResultType result = std::numeric_limits< ResultType >::max();
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::BitAnd{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BAND, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionBinaryOr( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] | expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] | expression[ 0 ] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::BitOr{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BOR, expression.getCommunicator() );
   }
   return result;
}

template< typename Expression >
auto
DistributedExpressionBinaryXor( const Expression& expression ) -> std::decay_t< decltype( expression[ 0 ] ^ expression[ 0 ] ) >
{
   using ResultType = std::decay_t< decltype( expression[ 0 ] ^ expression[ 0 ] ) >;

   ResultType result = 0;
   if( expression.getCommunicator() != MPI_COMM_NULL ) {
      const ResultType localResult = Algorithms::reduce( expression.getConstLocalView(), TNL::BitXor{} );
      MPI::Allreduce( &localResult, &result, 1, MPI_BXOR, expression.getCommunicator() );
   }
   return result;
}

}  // namespace Expressions
}  // namespace Containers
}  // namespace noa::TNL
