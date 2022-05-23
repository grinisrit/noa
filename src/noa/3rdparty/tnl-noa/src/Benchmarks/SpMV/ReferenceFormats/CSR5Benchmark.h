/***
 * Wrapper of original CSR5 kernels for TNL benchmarks.
 */

#include <stdexcept>


namespace TNL {
/////
// Currently CSR5 for CUDA cannot be build because of conflict of atomicAdd for `double` type:
//   https://github.com/weifengliu-ssslab/Benchmark_SpMV_using_CSR5/issues/9
// The solution is to insert whole benchmark into separate namespace. In this case, however,
// CSR5 does not work with `float`. So far, this seems to be the best solution.
namespace CSR5Benchmark {

#ifdef HAVE_CSR5
#include <CSR5_cuda/anonymouslib_cuda.h>
#endif

#ifdef HAVE_CSR5
template< typename CsrMatrix,
          typename Real = typename CsrMatrix::RealType >
struct CSR5SpMVCaller
{
   static_assert( std::is_same< typename CsrMatrix::DeviceType, TNL::Devices::Cuda >::value, "Only CUDA device is allowed for CSR matrix for CSR5 benchmark." );
   using RealType = typename CsrMatrix::RealType;
   using DeviceType = TNL::Devices::Cuda;
   using IndexType = typename CsrMatrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using VectorView = typename VectorType::ViewType;
   using CSR5Type = anonymouslibHandle< IndexType, typename std::make_unsigned< IndexType >::type, RealType >;

   static void spmv( CSR5Type& csr5, VectorView& outVector ) {
      csr5.spmv( ( RealType ) 1.0, outVector.getData() );
   };
};

template< typename CsrMatrix >
struct CSR5SpMVCaller< CsrMatrix, float >
{
   static_assert( std::is_same< typename CsrMatrix::DeviceType, TNL::Devices::Cuda >::value, "Only CUDA device is allowed for CSR matrix for CSR5 benchmark." );
   using RealType = typename CsrMatrix::RealType;
   using DeviceType = TNL::Devices::Cuda;
   using IndexType = typename CsrMatrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using VectorView = typename VectorType::ViewType;
   using CSR5Type = anonymouslibHandle< IndexType, typename std::make_unsigned< IndexType >::type, RealType >;

   static void spmv( CSR5Type& csr5, VectorView& outVector )
   {
      //csr5.spmv( ( RealType ) 1.0, outVector.getData() );
   };
};
#endif


template< typename CsrMatrix >
struct CSR5Benchmark
{
   static_assert( std::is_same< typename CsrMatrix::DeviceType, TNL::Devices::Cuda >::value, "Only CUDA device is allowed for CSR matrix for CSR5 benchmark." );
   using RealType = typename CsrMatrix::RealType;
   using DeviceType = TNL::Devices::Cuda;
   using IndexType = typename CsrMatrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using VectorView = typename VectorType::ViewType;
#ifdef HAVE_CSR5
   using CSR5Type = anonymouslibHandle< IndexType, typename std::make_unsigned< IndexType >::type, RealType >;
#endif

   CSR5Benchmark( CsrMatrix& matrix, VectorType& inVector, VectorType& outVector )
   :
#ifdef HAVE_CSR5
   csr5( matrix.getRows(), matrix.getColumns() ),
#endif
     inVectorView( inVector ), outVectorView( outVector )
   {
#ifdef HAVE_CSR5
      // err = A.inputCSR(nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA);
      //cout << "inputCSR err = " << err << endl;
      this->csr5.inputCSR( matrix.getValues().getSize(),
                           matrix.getRowPointers().getData(),
                           matrix.getColumnIndexes().getData(),
                           matrix.getValues().getData() );

      //err = A.setX(d_x); // you only need to do it once!
      //cout << "setX err = " << err << endl;
      this->csr5.setX( inVector.getData() );

      this->csr5.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

      // warmup device
      this->csr5.warmup();

      // conversion ... probably
      this->csr5.asCSR5();
#endif
   }

   void vectorProduct()
   {
#ifdef HAVE_CSR5
      CSR5SpMVCaller< CsrMatrix >::spmv( this->csr5, outVectorView );
#endif
   }

   const VectorView& getCudaOutVector()
   {
      return this->outVectorView;
   }

   ~CSR5Benchmark()
   {
#ifdef HAVE_CSR5
      this->csr5.destroy();
#endif
   }

   protected:
#ifdef HAVE_CSR5
      CSR5Type csr5;
#endif
      VectorView inVectorView, outVectorView;
};

   } // namespace CSR5Benchmark
} // namespace TNL
