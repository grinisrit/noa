#pragma once

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Sparse.h>
#include <TNL/Containers/Vector.h>

#include <TNL/Devices/Cuda.h>
#include <TNL/Exceptions/CudaBadAlloc.h>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

enum class Type {
   /* LONG = 0!!! Non zero value rewrites index[1] */
   LONG = 0,
   STREAM = 1,
   VECTOR = 2
};

template<typename Index>
union Block {
   Block(Index row, Type type = Type::VECTOR, Index index = 0) noexcept {
      this->index[0] = row;
      this->index[1] = index;
      this->byte[sizeof(Index) == 4 ? 7 : 15] = (uint8_t)type;
   }

   Block(Index row, Type type, Index nextRow, Index maxID, Index minID) noexcept {
      this->index[0] = row;
      this->index[1] = 0;
      this->twobytes[sizeof(Index) == 4 ? 2 : 4] = maxID - minID;

      if (type == Type::STREAM)
         this->twobytes[sizeof(Index) == 4 ? 3 : 5] = nextRow - row;

      if (type == Type::STREAM)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b1000000;
      else if (type == Type::VECTOR)
         this->byte[sizeof(Index) == 4 ? 7 : 15] |= 0b10000000;
   }

   Block() = default;

   template< typename Index2 >
   Block& operator=( const Block< Index2 >& source ) {
      index[ 0 ] = source.index[ 0 ];
      index[ 1 ] = source.index[ 1 ];
      for( int i = 0; i < ( sizeof(Index) == 4 ? 8 : 16); i ++ )
         byte[ i ] = source.byte[ i ];
      for( int i = 0; i < (sizeof(Index) == 4 ? 4 : 8); i++ )
         twobytes[ i ] = source.twobytes[ i ];
      return *this;
   }

   Index index[2]; // index[0] is row pointer, index[1] is index in warp
   uint8_t byte[sizeof(Index) == 4 ? 8 : 16]; // byte[7/15] is type specificator
   uint16_t twobytes[sizeof(Index) == 4 ? 4 : 8]; //twobytes[2/4] is maxID - minID
                                                //twobytes[3/5] is nextRow - row
};

#ifdef HAVE_UMFPACK
    template< typename Matrix, typename Preconditioner >
    class UmfpackWrapper;
#endif

template< typename Real >
class CusparseCSR;

template< typename Device >
class CSRDeviceDependentCode;

enum CSRKernel { CSRScalar, CSRVector, CSRHybrid, // Hybrid is not implemented
                 CSRLight, CSRLight2, CSRLight3, CSRLight4, CSRLight5, CSRLight6,
                 CSRAdaptive, CSRMultiVector, CSRLightWithoutAtomic };

template< typename Real, typename Device = Devices::Host, typename Index = int, CSRKernel KernelType = CSRScalar >
class CSR : public Sparse< Real, Device, Index >
{
private:
   // convenient template alias for controlling the selection of copy-assignment operator
   template< typename Device2 >
   using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

   // friend class will be needed for templated assignment operators
   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2 >
   friend class CSR;

public:

   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using RowsCapacitiesType = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesType;
   using RowsCapacitiesTypeView = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesView;
   using ConstRowsCapacitiesTypeView = typename Sparse< RealType, DeviceType, IndexType >::ConstRowsCapacitiesView;
   using BaseType = Sparse< Real, Device, Index >;
   using MatrixRow = typename BaseType::MatrixRow;
   using ConstMatrixRow = typename BaseType::ConstMatrixRow;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = CSR< _Real, _Device, _Index >;

   constexpr CSRKernel getSpMVKernelType() { return KernelType; };

   static constexpr bool isSymmetric() { return false; };

   Containers::Vector< Block<Index>, Device, Index > blocks;

   /* Configuration of CSR SpMV kernels ----------------------------------------- */

   /* Block sizes */
   static constexpr Index THREADS_ADAPTIVE = sizeof(Index) == 8 ? 128 : 256;
   static constexpr Index THREADS_SCALAR = 128;
   static constexpr Index THREADS_VECTOR = 128;
   static constexpr Index THREADS_LIGHT = 128;

   /* Max length of row to process one warp for CSR Light, MultiVector */
   static constexpr Index MAX_ELEMENTS_PER_WARP = 384;

   /* Max length of row to process one warp for CSR Adaptive */
   static constexpr Index MAX_ELEMENTS_PER_WARP_ADAPT = 512;

   /* How many shared memory use per block in CSR Adaptive kernel */
   static constexpr Index SHARED_PER_BLOCK = 24576;

   /* Number of elements in shared memory */
   static constexpr Index SHARED = SHARED_PER_BLOCK/sizeof(Real);

   /* Number of warps in block for CSR Adaptive */
   static constexpr Index WARPS = THREADS_ADAPTIVE / 32;

   /* Number of elements in shared memory per one warp */
   static constexpr Index SHARED_PER_WARP = SHARED / WARPS;
   /* -------------------------------------------------------------------------- */


   using Sparse< Real, Device, Index >::getAllocatedElementsCount;

   CSR();

   static std::string getSerializationType();

   std::string getSerializationTypeVirtual() const override;

   void setDimensions( const IndexType rows,
                       const IndexType columns ) override;

   void setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths );

   void setRowCapacities( ConstRowsCapacitiesTypeView rowLengths );

   void getCompressedRowLengths( RowsCapacitiesTypeView rowLengths ) const;

   IndexType getRowLength( const IndexType row ) const;

   __cuda_callable__
   IndexType getRowLengthFast( const IndexType row ) const;

   IndexType getNonZeroRowLength( const IndexType row ) const;

   __cuda_callable__
   IndexType getNonZeroRowLengthFast( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2 >
   void setLike( const CSR< Real2, Device2, Index2, KernelType2 >& matrix );

   void reset();

   __cuda_callable__
   bool setElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );
   __cuda_callable__
   bool addElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value,
                        const RealType& thisElementMultiplicator = 1.0 );

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );


   __cuda_callable__
   bool setRowFast( const IndexType row,
                    const IndexType* columnIndexes,
                    const RealType* values,
                    const IndexType elements );

   bool setRow( const IndexType row,
                const IndexType* columnIndexes,
                const RealType* values,
                const IndexType elements );


   __cuda_callable__
   bool addRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType numberOfElements,
                    const RealType& thisElementMultiplicator = 1.0 );

   bool addRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType numberOfElements,
                const RealType& thisElementMultiplicator = 1.0 );


   __cuda_callable__
   RealType getElementFast( const IndexType row,
                            const IndexType column ) const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   __cuda_callable__
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   ConstMatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;
   // TODO: add const RealType& multiplicator = 1.0 )

   template< typename Real2, typename Index2, CSRKernel KernelType2 >
   void addMatrix( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2, CSRKernel KernelType2 >
   void getTransposition( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2 >
   bool performSORIteration( const Vector1& b,
                             const IndexType row,
                             Vector2& x,
                             const RealType& omega = 1.0 ) const;

   // copy assignment
   CSR& operator=( const CSR& matrix );

   template< typename IndexType2, CSRKernel KernelType2 >
   CSR& operator=( const CSR< RealType, DeviceType, IndexType2, KernelType2 >& matrix );

   // cross-device copy assignment
   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2,
             typename = typename Enabler< Device2 >::type >
   CSR& operator=( const CSR< Real2, Device2, Index2, KernelType2 >& matrix );

   void save( File& file ) const override;

   void load( File& file ) override;

   void save( const String& fileName ) const;

   void load( const String& fileName );

   void print( std::ostream& str ) const override;

   //void setCudaKernelType( const SPMVCudaKernel kernel );

   //__cuda_callable__
   //SPMVCudaKernel getCudaKernelType() const;

   void setCudaWarpSize( const int warpSize );

   int getCudaWarpSize() const;

   void setHybridModeSplit( const IndexType hybridModeSplit );

   __cuda_callable__
   IndexType getHybridModeSplit() const;

   /* Analyze rowPointers, columnIndecies and values to create block for CSR Adaptive */
   void setBlocks();

   // The following getters allow us to interface TNL with external C-like
   // libraries such as UMFPACK or SuperLU, which need the raw data.
   const Containers::Vector< Index, Device, Index >&
   getRowPointers() const
   {
      return this->rowPointers;
   }

   Containers::Vector< Index, Device, Index >&
   getRowPointers()
   {
      return this->rowPointers;
   }

   const Containers::Vector< Index, Device, Index >&
   getColumnIndexes() const
   {
      return this->columnIndexes;
   }

   Containers::Vector< Index, Device, Index >&
   getColumnIndexes()
   {
      return this->columnIndexes;
   }

protected:

   Containers::Vector< Index, Device, Index > rowPointers;

   //SPMVCudaKernel spmvCudaKernel;

   int cudaWarpSize, hybridModeSplit;

   typedef CSRDeviceDependentCode< DeviceType > DeviceDependentCode;

   friend class CSRDeviceDependentCode< DeviceType >;
   friend class CusparseCSR< RealType >;
};

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR_impl.h>
