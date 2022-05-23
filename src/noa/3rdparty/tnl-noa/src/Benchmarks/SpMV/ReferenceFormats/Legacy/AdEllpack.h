/****
 * This class implements AdELL format from:
 *
 * Maggioni M., Berger-Wolf T.,
 * AdELL: An Adaptive Warp-Balancing ELL Format for Efficient Sparse Matrix-Vector Multiplication on GPUs,
 * In proceedings of 42nd International Conference on Parallel Processing, 2013.
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
class AdEllpackDeviceDependentCode;

template< typename MatrixType >
struct warpInfo
{
    using RealType = typename MatrixType::RealType;
    using DeviceType = typename MatrixType::DeviceType;
    using IndexType = typename MatrixType::IndexType;

    IndexType offset;
    IndexType rowOffset;
    IndexType localLoad;
    IndexType reduceMap[ 32 ];

    warpInfo< MatrixType >* next;
    warpInfo< MatrixType >* previous;
};

template< typename MatrixType >
class warpList
{
public:

    using RealType = typename MatrixType::RealType;
    using DeviceType = typename MatrixType::DeviceType;
    using IndexType = typename MatrixType::IndexType;

    warpList();

    bool addWarp( const IndexType offset,
                  const IndexType rowOffset,
                  const IndexType localLoad,
                  const IndexType* reduceMap );

    warpInfo< MatrixType >* splitInHalf( warpInfo< MatrixType >* warp );

    IndexType getNumberOfWarps()
    { return this->numberOfWarps; }

    warpInfo< MatrixType >* getNextWarp( warpInfo< MatrixType >* warp )
    { return warp->next; }

    warpInfo< MatrixType >* getHead()
    { return this->head; }

    warpInfo< MatrixType >* getTail()
    { return this->tail; }

    ~warpList();

    void printList()
    {
        if( this->getHead() == this->getTail() )
            std::cout << "HEAD==TAIL" << std::endl;
        else
        {
            for( warpInfo< MatrixType >* i = this->getHead(); i != this->getTail()->next; i = i->next )
            {
                if( i == this->getHead() )
                    std::cout << "Head:" << "\ti->localLoad = " << i->localLoad << "\ti->offset = " << i->offset << "\ti->rowOffset = " << i->rowOffset << std::endl;
                else if( i == this->getTail() )
                    std::cout << "Tail:" << "\ti->localLoad = " << i->localLoad << "\ti->offset = " << i->offset << "\ti->rowOffset = " << i->rowOffset << std::endl;
                else
                    std::cout << "\ti->localLoad = " << i->localLoad << "\ti->offset = " << i->offset << "\ti->rowOffset = " << i->rowOffset << std::endl;
            }
            std::cout << std::endl;
        }
    }

private:

    IndexType numberOfWarps;

    warpInfo< MatrixType >* head;
    warpInfo< MatrixType >* tail;

};

template< typename Real, typename Device, typename Index >
class AdEllpack : public Sparse< Real, Device, Index >
{
private:
   // convenient template alias for controlling the selection of copy-assignment operator
   template< typename Device2 >
   using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

   // friend class will be needed for templated assignment operators
   template< typename Real2, typename Device2, typename Index2 >
   friend class AdEllpack;

public:

    typedef Real RealType;
    typedef Device DeviceType;
    typedef Index IndexType;
    typedef typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesType RowsCapacitiesType;
    typedef typename Sparse< RealType, DeviceType, IndexType >::ConstRowsCapacitiesTypeView ConstRowsCapacitiesTypeView;
    typedef typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesTypeView RowsCapacitiesTypeView;

    template< typename _Real = Real,
              typename _Device = Device,
              typename _Index = Index >
    using Self = AdEllpack< _Real, _Device, _Index >;

    static constexpr bool isSymmetric() { return false; };

    AdEllpack();

    void setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths );

    void setRowCapacities( ConstRowsCapacitiesTypeView rowLengths );

    void getCompressedRowLengths( RowsCapacitiesTypeView rowLengths ) const;

    IndexType getWarp( const IndexType row ) const;

    IndexType getInWarpOffset( const IndexType row,
                               const IndexType warp ) const;

    IndexType getRowLength( const IndexType row ) const;

    template< typename Real2, typename Device2, typename Index2 >
    void setLike( const AdEllpack< Real2, Device2, Index2 >& matrix );

    void reset();

    template< typename Real2, typename Device2, typename Index2 >
    bool operator == ( const AdEllpack< Real2, Device2, Index2 >& matrix ) const;

    template< typename Real2, typename Device2, typename Index2 >
    bool operator != ( const AdEllpack< Real2, Device2, Index2 >& matrix ) const;

    bool setElement( const IndexType row,
                     const IndexType column,
                     const RealType& value );

    bool addElement( const IndexType row,
                     const IndexType column,
                     const RealType& value,
                     const RealType& thisElementMultiplicator = 1.0 );

    bool setRow( const IndexType row,
                 const IndexType* columnIndexes,
                 const RealType* values,
                 const IndexType elements );

    bool addRow( const IndexType row,
                 const IndexType* columnIndexes,
                 const RealType* values,
                 const IndexType elements,
                 const RealType& thisElementMultiplicator = 1.0 );

    RealType getElement( const IndexType row,
                         const IndexType column ) const;

    //MatrixRow getRow( const IndexType row );

    //const MatrixType getRow( const IndexType row ) const;

    // TODO: Change this to return MatrixRow type like in CSR format, like those above
    void getRow( const IndexType row,
                 IndexType* columns,
                 RealType* values ) const;

    template< typename InVector,
              typename OutVector >
    void vectorProduct( const InVector& inVector,
                        OutVector& outVector ) const;

    // copy assignment
    AdEllpack& operator=( const AdEllpack& matrix );

    // cross-device copy assignment
    template< typename Real2, typename Device2, typename Index2,
             typename = typename Enabler< Device2 >::type >
    AdEllpack& operator=( const AdEllpack< Real2, Device2, Index2 >& matrix );

    void save( File& file ) const override;

    void load( File& file ) override;

    void save( const String& fileName ) const;

    void load( const String& fileName );

    void print( std::ostream& str ) const override;

    bool balanceLoad( const RealType average,
                      ConstRowsCapacitiesTypeView rowLengths,
                      warpList< AdEllpack >* list );

    void computeWarps( const IndexType SMs,
                       const IndexType threadsPerSM,
                       warpList< AdEllpack >* list );

    bool createArrays( warpList< AdEllpack >* list );

    void performRowTest();

    void performRowLengthsTest( ConstRowsCapacitiesTypeView rowLengths );

    IndexType getTotalLoad() const;

#ifdef HAVE_CUDA
    template< typename InVector,
              typename OutVector >
    __device__
    void spmvCuda( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;

    template< typename InVector,
              typename OutVector >
   __device__
   void spmvCuda2( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;

   template< typename InVector,
             typename OutVector >
   __device__
   void spmvCuda4( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;

   template< typename InVector,
          typename OutVector >
   __device__
   void spmvCuda8( const InVector& inVector,
                   OutVector& outVector,
                   const int gridIdx ) const;

   template< typename InVector,
          typename OutVector >
   __device__
   void spmvCuda16( const InVector& inVector,
                    OutVector& outVector,
                    const int gridIdx ) const;

   template< typename InVector,
          typename OutVector >
   __device__
   void spmvCuda32( const InVector& inVector,
                    OutVector& outVector,
                    const int gridIdx ) const;


#endif


    // these arrays must be public
    Containers::Vector< Index, Device, Index > offset;

    Containers::Vector< Index, Device, Index > rowOffset;

    Containers::Vector< Index, Device, Index > localLoad;

    Containers::Vector< Index, Device, Index > reduceMap;

    typedef AdEllpackDeviceDependentCode< DeviceType > DeviceDependentCode;
    friend class AdEllpackDeviceDependentCode< DeviceType >;
    friend class AdEllpack< RealType, Devices::Host, IndexType >;
    friend class AdEllpack< RealType, Devices::Cuda, IndexType >;

protected:

    IndexType totalLoad;

    IndexType warpSize;

};

                } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/AdEllpack_impl.h>
