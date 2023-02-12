/***
 * Wrapper of original LightSpMV kernels for TNL benchmarks.
 */

#include <stdexcept>
#ifdef __CUDACC__
#include "LightSpMV-1.0/SpMV.h"
#endif
#include <TNL/Matrices/SparseMatrix.h>

namespace TNL {

enum LightSpMVBenchmarkKernelType { LightSpMVBenchmarkKernelVector, LightSpMVBenchmarkKernelWarp };

template< typename Real1, typename Real2 >
struct LightSpMVVectorsBinder
{
   template< typename Index >
   static void bind( TNL::Containers::VectorView< Real1, TNL::Devices::Cuda, Index >& vectorView, Real2* data, Index size ){}
};

template< typename Real >
struct LightSpMVVectorsBinder< Real, Real >
{
   template< typename Index >
   static void bind( TNL::Containers::VectorView< Real, TNL::Devices::Cuda, Index >& vectorView, Real* data, Index size )
   {
      vectorView.bind( data, size );
   }
};

template< typename Real >
struct LightSpMVBenchmark
{
   using RealType = Real;
   using DeviceType = TNL::Devices::Host;
   using IndexType = uint32_t;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using CudaVectorView = TNL::Containers::VectorView< RealType, TNL::Devices::Cuda, IndexType >;

   template< typename Matrix >
   LightSpMVBenchmark( Matrix& matrix, LightSpMVBenchmarkKernelType kernelType )
   : inVector( matrix.getColumns(), 1.0 ),
     outVector( matrix.getRows(), 0.0 ),
     kernelType( kernelType )
   {
      static_assert( std::is_same< typename Matrix::DeviceType, TNL::Devices::Host >::value, "The only device type accepted here is TNL::Devices::Host." );
#ifdef __CUDACC__
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, 0);
      opt._gpus.push_back(make_pair(0, prop));
      opt._numGPUs = 1;
      opt._numRows = matrix.getRows();
      opt._numCols = matrix.getColumns();
      opt._rowOffsets = matrix.getRowPointers().getData();
      opt._numValues = matrix.getValues().getSize();
      opt._colIndexValues = matrix.getColumnIndexes().getData();
      opt._numericalValues = matrix.getValues().getData();
      opt._alpha = 1.0; // matrix multiplicator
      opt._beta = 0.0;  // output vector multiplicator
      opt._vectorX = inVector.getData();
      opt._vectorY = outVector.getData();
      opt._formula = 0;
      if( std::is_same< Real, float >::value )
      {
         if( kernelType == LightSpMVBenchmarkKernelVector )
            this->spmv = new SpMVFloatVector( &opt );
         else
            this->spmv = new SpMVFloatWarp( &opt );
      }
      else if( std::is_same< Real, double >::value )
      {
         if( kernelType == LightSpMVBenchmarkKernelVector )
            this->spmv = new SpMVDoubleVector( &opt );
         else
            this->spmv = new SpMVDoubleWarp( &opt );
      }
      else throw std::runtime_error( "Unknown real type for LightSpMV." );
      this->spmv->loadData();
      if( std::is_same< Real, float >::value )
      {
         if( kernelType == LightSpMVBenchmarkKernelVector )
         {
            SpMVFloatVector* floatSpMV = dynamic_cast< SpMVFloatVector* >( this->spmv );
            LightSpMVVectorsBinder< Real, float >::bind( this->inVectorView, floatSpMV->_vectorX[ 0 ], matrix.getColumns() );
            LightSpMVVectorsBinder< Real, float >::bind( this->outVectorView, floatSpMV->_vectorY[ 0 ], matrix.getRows() );
         }
         else
         {
            SpMVFloatVector* floatSpMV = dynamic_cast< SpMVFloatWarp* >( this->spmv );
            LightSpMVVectorsBinder< Real, float >::bind( this->inVectorView, floatSpMV->_vectorX[ 0 ], matrix.getColumns() );
            LightSpMVVectorsBinder< Real, float >::bind( this->outVectorView, floatSpMV->_vectorY[ 0 ], matrix.getRows() );
         }
      }
      else if( std::is_same< Real, double >::value )
      {
         if( kernelType == LightSpMVBenchmarkKernelVector )
         {
            SpMVDoubleVector* doubleSpMV = dynamic_cast< SpMVDoubleVector* >( this->spmv );
            LightSpMVVectorsBinder< Real, double >::bind( this->inVectorView, doubleSpMV->_vectorX[ 0 ], matrix.getColumns() );
            LightSpMVVectorsBinder< Real, double >::bind( this->outVectorView, doubleSpMV->_vectorY[ 0 ], matrix.getRows() );
         }
         else
         {
            SpMVDoubleVector* doubleSpMV = dynamic_cast< SpMVDoubleWarp* >( this->spmv );
            LightSpMVVectorsBinder< Real, double >::bind( this->inVectorView, doubleSpMV->_vectorX[ 0 ], matrix.getColumns() );
            LightSpMVVectorsBinder< Real, double >::bind( this->outVectorView, doubleSpMV->_vectorY[ 0 ], matrix.getRows() );
         }
      }
      else std::runtime_error( "Unknown real type for LightSpMV." );
#endif
   }

   void setKernelType( LightSpMVBenchmarkKernelType type )
   {
      this->kernelType = type;
   }

   void resetVectors()
   {
      this->inVectorView = 1.0;
      this->outVectorView = 0.0;
   }

   void vectorProduct()
   {
#ifdef __CUDACC__
      this->spmv->spmvKernel();
      cudaDeviceSynchronize();
#endif

   }

   const CudaVectorView& getCudaOutVector()
   {
      return this->outVectorView;
   }

   ~LightSpMVBenchmark()
   {
#ifdef __CUDACC__
      if( spmv ) delete spmv;
      opt._rowOffsets = nullptr;
      opt._colIndexValues = nullptr;
      opt._numericalValues = nullptr;
      opt._vectorX = nullptr;
      opt._vectorY = nullptr;
#endif
   }

   protected:
#ifdef __CUDACC__
      Options opt;
      SpMV* spmv = nullptr;
#endif
      VectorType  inVector, outVector;
      CudaVectorView inVectorView, outVectorView;
      LightSpMVBenchmarkKernelType kernelType = LightSpMVBenchmarkKernelVector;
};

} // namespace TNL
