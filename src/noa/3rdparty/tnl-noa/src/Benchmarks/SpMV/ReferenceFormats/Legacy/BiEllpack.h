/****
 * This class implements BiELL format from:
 *
 * Zheng C., Gu S., Gu T.-X., Yang B., Liu X.-P.,
 * BiELL: A bisection ELLPACK-based storage format for optimizing SpMV on GPUs,
 * Journal of Parallel and Distributed Computing, 74 (7), pp. 2639-2647, 2014.
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
class BiEllpackDeviceDependentCode;

template< typename Real, typename Device, typename Index >
class BiEllpack : public Sparse< Real, Device, Index >
{
private:

    // convenient template alias for controlling the selection of copy-assignment operator
    template< typename Device2 >
    using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

    // friend class will be needed for templated assignment operators
    template< typename Real2, typename Device2, typename Index2 >
    friend class BiEllpack;

public:
	typedef Real RealType;
	typedef Device DeviceType;
	typedef Index IndexType;
   using RowsCapacitiesType = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesType;
   using RowsCapacitiesTypeView = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesView;
   using ConstRowsCapacitiesTypeView = typename Sparse< RealType, DeviceType, IndexType >::ConstRowsCapacitiesView;
	typedef typename Sparse< RealType, DeviceType, IndexType >::ValuesVector ValuesVector;
	typedef typename Sparse< RealType, DeviceType, IndexType >::ColumnIndexesVector ColumnIndexesVector;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = BiEllpack< _Real, _Device, _Index >;

   static constexpr bool isSymmetric() { return false; };

	BiEllpack();

	void setDimensions( const IndexType rows,
	                    const IndexType columns ) override;

   void setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths );

   void setRowCapacities( ConstRowsCapacitiesTypeView rowLengths );

   void getCompressedRowLengths( RowsCapacitiesTypeView rowLengths ) const;

	IndexType getRowLength( const IndexType row ) const;

	template< typename Real2,
			  typename Device2,
			  typename Index2 >
	void setLike( const BiEllpack< Real2, Device2, Index2 >& matrix );

        void reset();

        template< typename Real2, typename Device2, typename Index2 >
        bool operator == ( const BiEllpack< Real2, Device2, Index2 >& matrix ) const;

        template< typename Real2, typename Device2, typename Index2 >
        bool operator != ( const BiEllpack< Real2, Device2, Index2 >& matrix ) const;

	void getRowLengths( RowsCapacitiesType& rowLengths ) const;

	bool setElement( const IndexType row,
					 const IndexType column,
					 const RealType& value );

   __cuda_callable__
	bool setElementFast( const IndexType row,
						 const IndexType column,
						 const RealType& value );

	bool addElement( const IndexType row,
					 const IndexType column,
					 const RealType& value,
					 const RealType& thisElementMultiplicator = 1.0 );

   __cuda_callable__
	bool addElementFast( const IndexType row,
						 const IndexType column,
						 const RealType& value,
						 const RealType& thisElementMultiplicator = 1.0 );

	bool setRow( const IndexType row,
				 const IndexType* columns,
				 const RealType* values,
				 const IndexType numberOfElements );

	bool addRow( const IndexType row,
				 const IndexType* columns,
				 const RealType* values,
				 const IndexType numberOfElements,
				 const RealType& thisElementMultiplicator = 1.0 );

	RealType getElement( const IndexType row,
					 	 const IndexType column ) const;

   __cuda_callable__
	RealType getElementFast( const IndexType row,
							 const IndexType column ) const;

   // TODO: Change this to return MatrixRow type like in CSR format
	void getRow( const IndexType row,
			 	    IndexType* columns,
			 	    RealType* values ) const;

   __cuda_callable__
	IndexType getGroupLength( const IndexType strip,
							  const IndexType group ) const;

	template< typename InVector,
			  typename OutVector >
	void vectorProduct( const InVector& inVector,
						OutVector& outVector ) const;

	template< typename InVector,
			  typename OutVector >
	void vectorProductHost( const InVector& inVector,
							OutVector& outVector ) const;

	void setVirtualRows(const IndexType rows);

   __cuda_callable__
	IndexType getNumberOfGroups( const IndexType row ) const;

	bool vectorProductTest() const;

        // copy assignment
        BiEllpack& operator=( const BiEllpack& matrix );

        // cross-device copy assignment
        template< typename Real2, typename Device2, typename Index2,
                 typename = typename Enabler< Device2 >::type >
        BiEllpack& operator=( const BiEllpack< Real2, Device2, Index2 >& matrix );

	void save( File& file ) const override;

	void load( File& file ) override;

	void save( const String& fileName ) const;

	void load( const String& fileName );

	void print( std::ostream& str ) const override;

   void printValues() const;

	void performRowBubbleSort( Containers::Vector< Index, Device, Index >& tempRowLengths );
	void computeColumnSizes( Containers::Vector< Index, Device, Index >& tempRowLengths );

//	void verifyRowLengths( const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths );

	template< typename InVector,
			  typename OutVector >
#ifdef HAVE_CUDA
   __device__
#endif
	void spmvCuda( const InVector& inVector,
				   OutVector& outVector,
				   /*const IndexType warpStart,
				   const IndexType inWarpIdx*/
				   int globalIdx ) const;

   __cuda_callable__
	IndexType getStripLength( const IndexType strip ) const;

   __cuda_callable__
	void performRowBubbleSortCudaKernel( const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths,
										 const IndexType strip );

   __cuda_callable__
	void computeColumnSizesCudaKernel( const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths,
									   const IndexType numberOfStrips,
									   const IndexType strip );

   __cuda_callable__
	IndexType power( const IndexType number,
				     const IndexType exponent ) const;

	typedef BiEllpackDeviceDependentCode< DeviceType > DeviceDependentCode;
	friend class BiEllpackDeviceDependentCode< DeviceType >;
        friend class BiEllpack< RealType, Devices::Host, IndexType >;
        friend class BiEllpack< RealType, Devices::Cuda, IndexType >;

private:

	IndexType warpSize;

	IndexType logWarpSize;

	IndexType virtualRows;

	Containers::Vector< Index, Device, Index > rowPermArray;

	Containers::Vector< Index, Device, Index > groupPointers;

};
      			} //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/BiEllpack_impl.h>

