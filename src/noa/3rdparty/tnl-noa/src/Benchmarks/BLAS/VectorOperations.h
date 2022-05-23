#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Algorithms/ParallelFor.h>

namespace TNL {
namespace Benchmarks {

template< typename Device >
struct VectorOperations;

template<>
struct VectorOperations< Devices::Host >
{
   static constexpr int OpenMPVectorOperationsThreshold = 512;
   static constexpr int PrefetchDistance = 128;

   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2 >
   static void addVector( Vector1& y,
                          const Vector2& x,
                          const Scalar1 alpha,
                          const Scalar2 thisMultiplicator = 1.0 )
   {
      using Index = typename Vector1::IndexType;

      TNL_ASSERT_GT( x.getSize(), 0, "Vector size must be positive." );
      TNL_ASSERT_EQ( x.getSize(), y.getSize(), "The vector sizes must be the same." );

      const Index n = y.getSize();

      if( thisMultiplicator == 1.0 )
         #ifdef HAVE_OPENMP
         #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold )
         #endif
         for( Index i = 0; i < n; i ++ )
            y[ i ] += alpha * x[ i ];
      else
         #ifdef HAVE_OPENMP
         #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold )
         #endif
         for( Index i = 0; i < n; i ++ )
            y[ i ] = thisMultiplicator * y[ i ] + alpha * x[ i ];
}

   template< typename Vector1, typename Vector2, typename Vector3, typename Scalar1, typename Scalar2, typename Scalar3 >
   static void addVectors( Vector1& v,
                           const Vector2& v1,
                           const Scalar1 multiplicator1,
                           const Vector3& v2,
                           const Scalar2 multiplicator2,
                           const Scalar3 thisMultiplicator = 1.0 )
   {
      using Index = typename Vector1::IndexType;

      TNL_ASSERT_GT( v.getSize(), 0, "Vector size must be positive." );
      TNL_ASSERT_EQ( v.getSize(), v1.getSize(), "The vector sizes must be the same." );
      TNL_ASSERT_EQ( v.getSize(), v2.getSize(), "The vector sizes must be the same." );

      const Index n = v.getSize();
      if( thisMultiplicator == 1.0 )
         #ifdef HAVE_OPENMP
         #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold )
         #endif
         for( Index i = 0; i < n; i ++ )
            v[ i ] += multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
      else
         #ifdef HAVE_OPENMP
         #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() && n > OpenMPVectorOperationsThreshold )
         #endif
         for( Index i = 0; i < n; i ++ )
            v[ i ] = thisMultiplicator * v[ i ] + multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ];
   }
};

template<>
struct VectorOperations< Devices::Cuda >
{
   template< typename Vector1, typename Vector2, typename Scalar1, typename Scalar2 >
   static void addVector( Vector1& _y,
                          const Vector2& _x,
                          const Scalar1 alpha,
                          const Scalar2 thisMultiplicator = 1.0 )
   {
      TNL_ASSERT_GT( _x.getSize(), 0, "Vector size must be positive." );
      TNL_ASSERT_EQ( _x.getSize(), _y.getSize(), "The vector sizes must be the same." );

      using IndexType = typename Vector1::IndexType;
      using RealType = typename Vector1::RealType;

      RealType* y = _y.getData();
      const RealType* x = _x.getData();
      auto add1 = [=] __cuda_callable__ ( IndexType i ) { y[ i ] += alpha * x[ i ]; };
      auto add2 = [=] __cuda_callable__ ( IndexType i ) { y[ i ] = thisMultiplicator * y[ i ] + alpha * x[ i ]; };

      if( thisMultiplicator == 1.0 )
         Algorithms::ParallelFor< Devices::Cuda >::exec( (IndexType) 0, _y.getSize(), add1 );
      else
         Algorithms::ParallelFor< Devices::Cuda >::exec( (IndexType) 0, _y.getSize(), add2 );
   }

   template< typename Vector1, typename Vector2, typename Vector3, typename Scalar1, typename Scalar2, typename Scalar3 >
   static void addVectors( Vector1& _v,
                           const Vector2& _v1,
                           const Scalar1 multiplicator1,
                           const Vector3& _v2,
                           const Scalar2 multiplicator2,
                           const Scalar3 thisMultiplicator = 1.0 )
   {
      TNL_ASSERT_GT( _v.getSize(), 0, "Vector size must be positive." );
      TNL_ASSERT_EQ( _v.getSize(), _v1.getSize(), "The vector sizes must be the same." );
      TNL_ASSERT_EQ( _v.getSize(), _v2.getSize(), "The vector sizes must be the same." );

      using IndexType = typename Vector1::IndexType;
      using RealType = typename Vector1::RealType;

      RealType* v = _v.getData();
      const RealType* v1 = _v1.getData();
      const RealType* v2 = _v2.getData();
      auto add1 = [=] __cuda_callable__ ( IndexType i ) { v[ i ] += multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ]; };
      auto add2 = [=] __cuda_callable__ ( IndexType i ) { v[ i ] = thisMultiplicator * v[ i ] + multiplicator1 * v1[ i ] + multiplicator2 * v2[ i ]; };

      if( thisMultiplicator == 1.0 )
         Algorithms::ParallelFor< Devices::Cuda >::exec( (IndexType) 0, _v.getSize(), add1 );
      else
         Algorithms::ParallelFor< Devices::Cuda >::exec( (IndexType) 0, _v.getSize(), add2 );
   }
};

} // namespace Benchmarks
} // namespace TNL
