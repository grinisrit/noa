#include <TNL/Assert.h>
#include <TNL/Devices/Cuda.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>
#ifdef HAVE_CUDA
#include <cusparse.h>
#endif

namespace TNL {

template< typename Real >
class CusparseCSRBaseLegacy
{
   public:
      using RealType = Real;
      using DeviceType = TNL::Devices::Cuda;
      using MatrixType = Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< Real, Devices::Cuda, int >;

      CusparseCSRBaseLegacy()
      : matrix( 0 )
      {
      };

#ifdef HAVE_CUDA
      void init( const MatrixType& matrix,
                 cusparseHandle_t* cusparseHandle )
      {
         this->matrix = &matrix;
         this->cusparseHandle = cusparseHandle;
         cusparseCreateMatDescr( & this->matrixDescriptor );
      };
#endif

      int getRows() const
      {
         return matrix->getRows();
      }

      int getColumns() const
      {
         return matrix->getColumns();
      }

      int getNumberOfMatrixElements() const
      {
         return matrix->getAllocatedElementsCount();
      }


      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef HAVE_CUDA
#if CUDART_VERSION >= 11000
         throw std::runtime_error("cusparseDcsrmv was removed in CUDA 11.");
#else
         cusparseDcsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->values.getSize(),
                         1.0,
                         this->matrixDescriptor,
                         this->matrix->values.getData(),
                         this->matrix->getRowPointers().getData(),
                         this->matrix->columnIndexes.getData(),
                         inVector.getData(),
                         1.0,
                         outVector.getData() );
#endif
#endif
      }

   protected:

      const MatrixType* matrix;
#ifdef HAVE_CUDA
      cusparseHandle_t* cusparseHandle;

      cusparseMatDescr_t matrixDescriptor;
#endif
};


template< typename Real >
class CusparseCSRLegacy
{};

template<>
class CusparseCSRLegacy< double > : public CusparseCSRBaseLegacy< double >
{
   public:

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef HAVE_CUDA
#if CUDART_VERSION >= 11000
         throw std::runtime_error("cusparseDcsrmv was removed in CUDA 11.");
#else
	 double d = 1.0;
         double* alpha = &d;
         cusparseDcsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->getValues().getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->getValues().getData(),
                         this->matrix->getRowPointers().getData(),
                         this->matrix->getColumnIndexes().getData(),
                         inVector.getData(),
                         alpha,
                         outVector.getData() );
#endif
#endif
      }
};

template<>
class CusparseCSRLegacy< float > : public CusparseCSRBaseLegacy< float >
{
   public:

      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector ) const
      {
         TNL_ASSERT_TRUE( matrix, "matrix was not initialized" );
#ifdef HAVE_CUDA
#if CUDART_VERSION >= 11000
         throw std::runtime_error("cusparseScsrmv was removed in CUDA 11.");
#else
         float d = 1.0;
         float* alpha = &d;
         cusparseScsrmv( *( this->cusparseHandle ),
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         this->matrix->getRows(),
                         this->matrix->getColumns(),
                         this->matrix->getValues().getSize(),
                         alpha,
                         this->matrixDescriptor,
                         this->matrix->getValues().getData(),
                         this->matrix->getRowPointers().getData(),
                         this->matrix->getColumnIndexes().getData(),
                         inVector.getData(),
                         alpha,
                         outVector.getData() );
#endif
#endif
      }
};

} // namespace TNL
