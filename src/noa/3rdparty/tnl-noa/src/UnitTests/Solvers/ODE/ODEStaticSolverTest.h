#pragma once

#include <TNL/Solvers/ODE/Euler.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Algorithms/ParallelFor.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_GTEST

// test fixture for typed tests
template< typename DofContainer >
class ODENumericSolverTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
};

template< typename DofContainer >
class ODEStaticSolverTest : public ::testing::Test
{
protected:
   using DofContainerType = DofContainer;
};

// types for which DofContainerTest is instantiated
using DofNumericTypes = ::testing::Types<
   float,
   double
>;

// types for which DofContainerTest is instantiated
using DofStaticVectorTypes = ::testing::Types<
   StaticVector< 1, float >,
   StaticVector< 2, float >,
   StaticVector< 3, float >,
   StaticVector< 1, double >,
   StaticVector< 2, double >,
   StaticVector< 3, double >
>;

TYPED_TEST_SUITE( ODENumericSolverTest, DofNumericTypes );

TYPED_TEST_SUITE( ODEStaticSolverTest, DofStaticVectorTypes );

template< typename RealType, typename SolverType >
void ODENumericSolverTest_LinearFunctionTest()
{
   const RealType final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   RealType u( 0.0 );
   solver.solve( u, [] __cuda_callable__ ( const RealType& time, const RealType& tau, const RealType& u, RealType& fu ) {
      fu = time;
   } );

   RealType exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::abs( u - exact_solution ), ( RealType ) 0.0, 0.1 );
}

TYPED_TEST( ODENumericSolverTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = ODETestSolver< DofContainerType >;
   using Real = DofContainerType;

   ODENumericSolverTest_LinearFunctionTest< Real, SolverType >();
}

template< typename RealType, typename SolverType, typename Device >
void ODENumericSolverTest_ParallelLinearFunctionTest()
{
   const int size = 10;
   const RealType final_time = 10.0;
   TNL::Containers::Vector< RealType, Device > u( size, 0.0 );
   auto u_view = u.getView();
   auto inner_f = [] __cuda_callable__ ( const RealType& time, const RealType& tau, const RealType& u, RealType& fu ) {
         fu = time;
   };
   auto f = [=] __cuda_callable__ ( int idx ) mutable {
      SolverType solver;
      solver.setTime( 0.0 );
      solver.setStopTime( final_time );
      solver.setTau( 0.005 );
      solver.setConvergenceResidue( 0.0 );
      solver.solve( u_view[ idx ], inner_f );
      // The following is not accepted by nvcc
      //solver.solve( u_view[ idx ], [] __cuda_callable__ ( const RealType& time, const RealType& tau, const RealType& u, RealType& fu ) {
      //   fu = time;
      //} );
   };
   TNL::Algorithms::ParallelFor< Device >::exec( 0, size, f );


   RealType exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::max( TNL::abs( u - exact_solution ) ), ( RealType ) 0.0, 0.1 );
}

TYPED_TEST( ODENumericSolverTest, ParallelLinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = ODETestSolver< DofContainerType >;

#ifndef __CUDACC__
   ODENumericSolverTest_ParallelLinearFunctionTest< DofContainerType, SolverType, TNL::Devices::Host >();
#else
   ODENumericSolverTest_ParallelLinearFunctionTest< DofContainerType, SolverType, TNL::Devices::Cuda >();
#endif
}

template< typename DofContainerType, typename SolverType >
void ODEStaticSolverTest_LinearFunctionTest()
{
   using StaticVectorType = DofContainerType;
   using RealType = typename DofContainerType::RealType;

   const RealType final_time = 10.0;
   SolverType solver;
   solver.setTime( 0.0 );
   solver.setStopTime( final_time );
   solver.setTau( 0.005 );
   solver.setConvergenceResidue( 0.0 );

   DofContainerType u( 0.0 );
   solver.solve( u, [] __cuda_callable__ ( const RealType& time, const RealType& tau, const StaticVectorType& u, StaticVectorType& fu ) {
      fu = time;
   } );

   RealType exact_solution = 0.5 * final_time * final_time;
   EXPECT_NEAR( TNL::max( TNL::abs( u - exact_solution ) ), ( RealType ) 0.0, 0.1 );
}

TYPED_TEST( ODEStaticSolverTest, LinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = ODETestSolver< DofContainerType >;

   ODEStaticSolverTest_LinearFunctionTest< DofContainerType, SolverType >();
}

template< typename DofContainerType, typename SolverType, typename Device >
void ODEStaticSolverTest_ParallelLinearFunctionTest()
{
   using RealType = typename DofContainerType::RealType;

   const int size = 10;
   const RealType final_time = 10.0;
   TNL::Containers::Vector< DofContainerType, Device > u( size, 0.0 );
   auto u_view = u.getView();
   auto inner_f = [=] __cuda_callable__ ( const RealType& time, const RealType& tau, const DofContainerType& u, DofContainerType& fu ) {
      fu = time;
   };
   auto f = [=] __cuda_callable__ ( int idx ) mutable {
      SolverType solver;
      solver.setTime( 0.0 );
      solver.setStopTime( final_time );
      solver.setTau( 0.005 );
      solver.setConvergenceResidue( 0.0 );
      solver.solve( u_view[ idx ], inner_f );

      // The following is not accepted by nvcc compiler
      //solver.solve( u_view[ idx ], [] __cuda_callable__ ( const RealType& time, const RealType& tau, const DofContainerType& u, DofContainerType& fu ) {
      //   fu = time;
      //} );
   };
   TNL::Algorithms::ParallelFor< Device >::exec( 0, size, f );

   RealType exact_solution( 0.5 * final_time * final_time );
   auto error = TNL::Algorithms::reduce< Device >( 0, size,
      [=] __cuda_callable__ ( int idx ) -> RealType { return TNL::max( u_view[ idx ] - exact_solution ); },
      TNL::Max() );
   EXPECT_NEAR( error, ( RealType ) 0.0, 0.1 );
}

TYPED_TEST( ODEStaticSolverTest, ParallelLinearFunctionTest )
{
   using DofContainerType = typename TestFixture::DofContainerType;
   using SolverType = ODETestSolver< DofContainerType >;

#ifndef __CUDACC__
   ODEStaticSolverTest_ParallelLinearFunctionTest< DofContainerType, SolverType, TNL::Devices::Host >();
#else
   ODEStaticSolverTest_ParallelLinearFunctionTest< DofContainerType, SolverType, TNL::Devices::Cuda >();
#endif
}


#endif

#include "../../main.h"
