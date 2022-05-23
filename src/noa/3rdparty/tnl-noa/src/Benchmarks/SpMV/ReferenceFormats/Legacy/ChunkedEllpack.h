/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Heller Martin
 *
 * The algorithm/method was published in:
 * Heller M., Oberhuber T., Improved Row-grouped CSR Format for Storing of
 * Sparse Matrices on GPU, Proceedings of Algoritmy 2012, 2012, Handlovičová A.,
 * Minarechová Z. and Ševčovič D. (ed.), pages 282-290.
 */


#pragma once

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Sparse.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Device >
class ChunkedEllpackDeviceDependentCode;

template< typename Real, typename Device = Devices::Host, typename Index = int >
class ChunkedEllpack;

#ifdef HAVE_CUDA
#endif

template< typename IndexType >
struct tnlChunkedEllpackSliceInfo
{
   IndexType size;
   IndexType chunkSize;
   IndexType firstRow;
   IndexType pointer;
};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename Vector >
__global__ void ChunkedEllpackVectorProductCudaKernel( const ChunkedEllpack< Real, Devices::Cuda, Index >* matrix,
                                                                const Vector* inVector,
                                                                Vector* outVector,
                                                                int gridIdx );
#endif

template< typename Real, typename Device, typename Index >
class ChunkedEllpack : public Sparse< Real, Device, Index >
{
private:
   // convenient template alias for controlling the selection of copy-assignment operator
   template< typename Device2 >
   using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

   // friend class will be needed for templated assignment operators
   template< typename Real2, typename Device2, typename Index2 >
   friend class ChunkedEllpack;

public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlChunkedEllpackSliceInfo< IndexType > ChunkedEllpackSliceInfo;
   using RowsCapacitiesType = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesType;
   using RowsCapacitiesTypeView = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesView;
   using ConstRowsCapacitiesTypeView = typename Sparse< RealType, DeviceType, IndexType >::ConstRowsCapacitiesView;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
   typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;
   typedef ChunkedEllpack< Real, Device, Index > ThisType;
   typedef ChunkedEllpack< Real, Devices::Host, Index > HostType;
   typedef ChunkedEllpack< Real, Devices::Cuda, Index > CudaType;
   typedef Sparse< Real, Device, Index > BaseType;
   typedef typename BaseType::MatrixRow MatrixRow;
   typedef SparseRow< const RealType, const IndexType > ConstMatrixRow;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = ChunkedEllpack< _Real, _Device, _Index >;

   static constexpr bool isSymmetric() { return false; };

   ChunkedEllpack();

   static std::string getSerializationType();

   std::string getSerializationTypeVirtual() const override;

   void setDimensions( const IndexType rows,
                       const IndexType columns ) override;

   void setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths );

   void setRowCapacities( ConstRowsCapacitiesTypeView rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   __cuda_callable__
   IndexType getRowLengthFast( const IndexType row ) const;

   IndexType getNonZeroRowLength( const IndexType row ) const;

   template< typename Real2, typename Device2, typename Index2 >
   void setLike( const ChunkedEllpack< Real2, Device2, Index2 >& matrix );

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const ChunkedEllpack< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const ChunkedEllpack< Real2, Device2, Index2 >& matrix ) const;

   void setNumberOfChunksInSlice( const IndexType chunksInSlice );

   __cuda_callable__
   IndexType getNumberOfChunksInSlice() const;

   void setDesiredChunkSize( const IndexType desiredChunkSize );

   IndexType getDesiredChunkSize() const;

   __cuda_callable__
   IndexType getNumberOfSlices() const;

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

   /*void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;*/

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   ConstMatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

#ifdef HAVE_CUDA
   template< typename InVector,
             typename OutVector >
   __device__ void computeSliceVectorProduct( const InVector* inVector,
                                              OutVector* outVector,
                                              int gridIdx  ) const;
#endif


   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const ChunkedEllpack< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const ChunkedEllpack< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2 >
   bool performSORIteration( const Vector1& b,
                             const IndexType row,
                             Vector2& x,
                             const RealType& omega = 1.0 ) const;

   // copy assignment
   ChunkedEllpack& operator=( const ChunkedEllpack& matrix );

   // cross-device copy assignment
   template< typename Real2, typename Device2, typename Index2,
             typename = typename Enabler< Device2 >::type >
   ChunkedEllpack& operator=( const ChunkedEllpack< Real2, Device2, Index2 >& matrix );

   void save( File& file ) const override;

   void load( File& file ) override;

   void save( const String& fileName ) const;

   void load( const String& fileName );

   void print( std::ostream& str ) const override;

   void printStructure( std::ostream& str,
                        const String& = "" ) const;

protected:

   void resolveSliceSizes( ConstRowsCapacitiesTypeView rowLengths );

   bool setSlice( ConstRowsCapacitiesTypeView rowLengths,
                  const IndexType sliceIdx,
                  IndexType& elementsToAllocation );

   bool addElementToChunk( const IndexType sliceOffset,
                           const IndexType chunkIndex,
                           const IndexType chunkSize,
                           IndexType& column,
                           RealType& value,
                           RealType& thisElementMultiplicator );

   __cuda_callable__
   bool addElementToChunkFast( const IndexType sliceOffset,
                               const IndexType chunkIndex,
                               const IndexType chunkSize,
                               IndexType& column,
                               RealType& value,
                               RealType& thisElementMultiplicator );

   __cuda_callable__
   void setChunkFast( const IndexType sliceOffset,
                      const IndexType chunkIndex,
                      const IndexType chunkSize,
                      const IndexType* columnIndexes,
                      const RealType* values,
                      const IndexType elements );

   void setChunk( const IndexType sliceOffset,
                  const IndexType chunkIndex,
                  const IndexType chunkSize,
                  const IndexType* columnIndexes,
                  const RealType* values,
                  const IndexType elements );

   bool getElementInChunk( const IndexType sliceOffset,
                           const IndexType chunkIndex,
                           const IndexType chunkSize,
                           const IndexType column,
                           RealType& value ) const;

   __cuda_callable__
   bool getElementInChunkFast( const IndexType sliceOffset,
                               const IndexType chunkIndex,
                               const IndexType chunkSize,
                               const IndexType column,
                               RealType& value ) const;

   void getChunk( const IndexType sliceOffset,
                  const IndexType chunkIndex,
                  const IndexType chunkSize,
                  IndexType* columns,
                  RealType* values ) const;

   __cuda_callable__
   void getChunkFast( const IndexType sliceOffset,
                      const IndexType chunkIndex,
                      const IndexType chunkSize,
                      IndexType* columns,
                      RealType* values ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType chunkVectorProduct( const IndexType sliceOffset,
                                                 const IndexType chunkIndex,
                                                 const IndexType chunkSize,
                                                 const Vector& vector ) const;



   IndexType chunksInSlice, desiredChunkSize;

   Containers::Vector< Index, Device, Index > rowToChunkMapping, rowToSliceMapping, rowPointers;

   Containers::Array< ChunkedEllpackSliceInfo, Device, Index > slices;

   IndexType numberOfSlices;

   typedef ChunkedEllpackDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class ChunkedEllpackDeviceDependentCode< DeviceType >;
   friend class ChunkedEllpack< RealType, Devices::Host, IndexType >;
   friend class ChunkedEllpack< RealType, Devices::Cuda, IndexType >;

#ifdef HAVE_CUDA
   template< typename Vector >
   friend void ChunkedEllpackVectorProductCudaKernel( const ChunkedEllpack< Real, Devices::Cuda, Index >* matrix,
                                                               const Vector* inVector,
                                                               Vector* outVector,
                                                               int gridIdx );
#endif
};

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack_impl.h>

