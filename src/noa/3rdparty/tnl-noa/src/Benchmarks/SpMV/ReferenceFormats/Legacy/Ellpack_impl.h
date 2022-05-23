#pragma once

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real,
          typename Device,
          typename Index >
Ellpack< Real, Device, Index > :: Ellpack()
: rowLengths( 0 ), alignedRows( 0 )
{
};

template< typename Real,
          typename Device,
          typename Index >
std::string Ellpack< Real, Device, Index >::getSerializationType()
{
   return "Matrices::Ellpack< " +
          TNL::getType< Real >() +
          ", [any device], " +
          getType< Index >() +
          " >";
}

template< typename Real,
          typename Device,
          typename Index >
std::string Ellpack< Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::setDimensions( const IndexType rows,
                                                    const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
              std::cerr << "rows = " << rows
                   << " columns = " << columns << std::endl );
   this->rows = rows;
   this->columns = columns;

   if( std::is_same< Device, Devices::Cuda >::value )
   {
       this->alignedRows = roundToMultiple( columns, Cuda::getWarpSize() );
       if( this->rows - this->alignedRows > 0 )
       {
           IndexType missingRows = this->rows - this->alignedRows;

           missingRows = roundToMultiple( missingRows, Cuda::getWarpSize() );

           this->alignedRows +=  missingRows;
       }
   }
   else this->alignedRows = rows;

   if( this->rowLengths != 0 )
      allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths )
{
   TNL_ASSERT_GT( this->getRows(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_GT( this->getColumns(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_EQ( this->getRows(), rowLengths.getSize(), "wrong size of the rowLengths vector" );

   this->rowLengths = this->maxRowLength = max( rowLengths );

   allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::setRowCapacities( ConstRowsCapacitiesTypeView rowLengths )
{
   setCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::getCompressedRowLengths( RowsCapacitiesTypeView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::setConstantCompressedRowLengths( const IndexType& rowLengths )
{
   TNL_ASSERT( rowLengths > 0,
              std::cerr << " rowLengths = " << rowLengths );
   this->rowLengths = rowLengths;
   if( this->rows > 0 )
      allocateElements();
}

template< typename Real,
          typename Device,
          typename Index >
Index Ellpack< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   return this->rowLengths;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index Ellpack< Real, Device, Index >::getRowLengthFast( const IndexType row ) const
{
   return this->rowLengths;
}

template< typename Real,
          typename Device,
          typename Index >
Index Ellpack< Real, Device, Index >::getNonZeroRowLength( const IndexType row ) const
{
    ConstMatrixRow matrixRow = getRow( row );
    return matrixRow.getNonZeroElementsCount( getType< Device >() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
void Ellpack< Real, Device, Index >::setLike( const Ellpack< Real2, Device2, Index2 >& matrix )
{
   Sparse< Real, Device, Index >::setLike( matrix );
   this->rowLengths = matrix.rowLengths;
   if( std::is_same< Device, Devices::Cuda >::value )
      this->alignedRows = roundToMultiple( this->getRows(), Cuda::getWarpSize() );
   else this->alignedRows = this->getRows();
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index > :: reset()
{
   Sparse< Real, Device, Index >::reset();
   this->rowLengths = 0;
   this->alignedRows = 0;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool Ellpack< Real, Device, Index >::operator == ( const Ellpack< Real2, Device2, Index2 >& matrix ) const
{
   TNL_ASSERT( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
              std::cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns() );
   // TODO: implement this
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool Ellpack< Real, Device, Index >::operator != ( const Ellpack< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Ellpack< Real, Device, Index > :: setElementFast( const IndexType row,
                                                                const IndexType column,
                                                                const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool Ellpack< Real, Device, Index > :: setElement( const IndexType row,
                                                            const IndexType column,
                                                            const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Ellpack< Real, Device, Index > :: addElementFast( const IndexType row,
                                                                const IndexType column,
                                                                const RealType& value,
                                                                const RealType& thisElementMultiplicator )
{
   // TODO: return this back when CUDA kernels support std::cerr
   /*TNL_ASSERT( row >= 0 && row < this->rows &&
              column >= 0 && column <= this->rows,
              std::cerr << " row = " << row
                   << " column = " << column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/
   typedef EllpackDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd &&
         this->columnIndexes[ i ] < column &&
         this->columnIndexes[ i ] != this->getPaddingIndex() ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes[ i ] == column )
   {
      this->values[ i ] = thisElementMultiplicator * this->values[ i ] + value;
      return true;
   }
   else
      if( this->columnIndexes[ i ] == this->getPaddingIndex() ) // artificial zero
      {
         this->columnIndexes[ i ] = column;
         this->values[ i ] = value;
      }
      else
      {
         Index j = rowEnd - step;
         while( j > i )
         {
            this->columnIndexes[ j ] = this->columnIndexes[ j - step ];
            this->values[ j ] = this->values[ j - step ];
            j -= step;
         }
         this->columnIndexes[ i ] = column;
         this->values[ i ] = value;
      }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Ellpack< Real, Device, Index > :: addElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
   typedef EllpackDeviceDependentCode< DeviceType > DDCType;
   IndexType i = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( i < rowEnd &&
          this->columnIndexes.getElement( i ) < column &&
          this->columnIndexes.getElement( i ) != this->getPaddingIndex() ) i += step;
   if( i == rowEnd )
      return false;
   if( this->columnIndexes.getElement( i ) == column )
   {
      this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
      return true;
   }
   else
      if( this->columnIndexes.getElement( i ) == this->getPaddingIndex() )
      {
         this->columnIndexes.setElement( i, column );
         this->values.setElement( i, value );
      }
      else
      {
         IndexType j = rowEnd - step;
         while( j > i )
         {
            this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - step ) );
            this->values.setElement( j, this->values.getElement( j - step ) );
            j -= step;
         }
         this->columnIndexes.setElement( i, column );
         this->values.setElement( i, value );
      }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Ellpack< Real, Device, Index > :: setRowFast( const IndexType row,
                                                            const IndexType* columnIndexes,
                                                            const RealType* values,
                                                            const IndexType elements )
{
   typedef EllpackDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPointer = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   if( elements > this->rowLengths )
      return false;
   for( Index i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes[ elementPointer ] = column;
      this->values[ elementPointer ] = values[ i ];
      elementPointer += step;
   }
   for( Index i = elements; i < this->rowLengths; i++ )
   {
      this->columnIndexes[ elementPointer ] = this->getPaddingIndex();
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
bool Ellpack< Real, Device, Index > :: setRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   typedef EllpackDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPointer = DDCType::getRowBegin( *this, row );
   //const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   if( elements > this->rowLengths )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      const IndexType column = columnIndexes[ i ];
      if( column < 0 || column >= this->getColumns() )
         return false;
      this->columnIndexes.setElement( elementPointer, column );
      this->values.setElement( elementPointer, values[ i ] );
      elementPointer += step;
   }
   for( IndexType i = elements; i < this->rowLengths; i++ )
   {
      this->columnIndexes.setElement( elementPointer, this->getPaddingIndex() );
      elementPointer += step;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool Ellpack< Real, Device, Index > :: addRowFast( const IndexType row,
                                                            const IndexType* columns,
                                                            const RealType* values,
                                                            const IndexType numberOfElements,
                                                            const RealType& thisElementMultiplicator )
{
   // TODO: implement
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool Ellpack< Real, Device, Index > :: addRow( const IndexType row,
                                                        const IndexType* columns,
                                                        const RealType* values,
                                                        const IndexType numberOfElements,
                                                        const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements );
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real Ellpack< Real, Device, Index >::getElementFast( const IndexType row,
                                                              const IndexType column ) const
{
   typedef EllpackDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd &&
          this->columnIndexes[ elementPtr ] < column &&
          this->columnIndexes[ elementPtr ] != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr < rowEnd && this->columnIndexes[ elementPtr ] == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
Real Ellpack< Real, Device, Index >::getElement( const IndexType row,
                                                          const IndexType column ) const
{
   typedef EllpackDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DDCType::getRowBegin( *this, row );
   const IndexType rowEnd = DDCType::getRowEnd( *this, row );
   const IndexType step = DDCType::getElementStep( *this );

   while( elementPtr < rowEnd &&
          this->columnIndexes.getElement( elementPtr ) < column &&
          this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() ) elementPtr += step;
   if( elementPtr < rowEnd && this->columnIndexes.getElement( elementPtr ) == column )
      return this->values.getElement( elementPtr );
   return 0.0;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void Ellpack< Real, Device, Index >::getRowFast( const IndexType row,
                                                          IndexType* columns,
                                                          RealType* values ) const
{
   //typedef EllpackDeviceDependentCode< DeviceType > DDCType;
   IndexType elementPtr = DeviceDependentCode::getRowBegin( *this, row );
   const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
   const IndexType step = DeviceDependentCode::getElementStep( *this );

   for( IndexType i = 0; i < this->rowLengths; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPtr ];
      values[ i ] = this->values[ elementPtr ];
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename Ellpack< Real, Device, Index >::MatrixRow
Ellpack< Real, Device, Index >::
getRow( const IndexType rowIndex )
{
   //printf( "this->rowLengths = %d this = %p \n", this->rowLengths, this );
   IndexType rowBegin = DeviceDependentCode::getRowBegin( *this, rowIndex );
   return MatrixRow( &this->columnIndexes[ rowBegin ],
                     &this->values[ rowBegin ],
                     this->rowLengths,
                     DeviceDependentCode::getElementStep( *this ) );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename Ellpack< Real, Device, Index >::ConstMatrixRow
Ellpack< Real, Device, Index >::
getRow( const IndexType rowIndex ) const
{
   //printf( "this->rowLengths = %d this = %p \n", this->rowLengths, this );
   IndexType rowBegin = DeviceDependentCode::getRowBegin( *this, rowIndex );
   return ConstMatrixRow( &this->columnIndexes[ rowBegin ],
                          &this->values[ rowBegin ],
                          this->rowLengths,
                          DeviceDependentCode::getElementStep( *this ) );
}

template< typename Real,
          typename Device,
          typename Index >
  template< typename Vector >
__cuda_callable__
typename Vector::RealType Ellpack< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                     const Vector& vector ) const
{
   IndexType i = DeviceDependentCode::getRowBegin( *this, row );
   const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
   const IndexType step = DeviceDependentCode::getElementStep( *this );

   Real result = 0.0;
   while( i < rowEnd && this->columnIndexes[ i ] != this->getPaddingIndex() )
   {
      const Index column = this->columnIndexes[ i ];
      result += this->values[ i ] * vector[ column ];
      i += step;
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void Ellpack< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                    OutVector& outVector,
                                                    RealType multiplicator ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector, multiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void Ellpack< Real, Device, Index > :: addMatrix( const Ellpack< Real2, Device, Index2 >& matrix,
                                                                 const RealType& matrixMultiplicator,
                                                                 const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "Ellpack::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void Ellpack< Real, Device, Index >::getTransposition( const Ellpack< Real2, Device, Index2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "Ellpack::getTransposition is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2 >
bool Ellpack< Real, Device, Index > :: performSORIteration( const Vector1& b,
                                                            const IndexType row,
                                                            Vector2& x,
                                                            const RealType& omega ) const
{
   TNL_ASSERT( row >=0 && row < this->getRows(),
              std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   IndexType i = DeviceDependentCode::getRowBegin( *this, row );
   const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
   //const IndexType step = DeviceDependentCode::getElementStep( *this );

   IndexType column;
   while( i < rowEnd && ( column = this->columnIndexes[ i ] ) < this->columns )
   {
      if( column == row )
         diagonalValue = this->values[ i ];
      else
         sum += this->values[ i ] * x[ column ];
      i++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      std::cerr << "There is zero on the diagonal in " << row << "-th row of a matrix. I cannot perform SOR iteration." << std::endl;
      return false;
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool
Ellpack< Real, Device, Index >::
performJacobiIteration( const Vector& b,
                        const IndexType row,
                        const Vector& old_x,
                        Vector& x,
                        const RealType& omega ) const
{
   TNL_ASSERT( row >=0 && row < this->getRows(),
               std::cerr << "row = " << row << " this->getRows() = " << this->getRows() << std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   IndexType i = DeviceDependentCode::getRowBegin( *this, row );
   const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
   //const IndexType step = DeviceDependentCode::getElementStep( *this );

   IndexType column;
   while( i < rowEnd && ( column = this->columnIndexes[ i ] ) < this->columns )
   {
	  if( column == row )
		 diagonalValue = this->values[ i ];
	  else
		 sum += this->values[ i ] * old_x[ column ];
	  i++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
	  std::cerr << "There is zero on the diagonal in " << row << "-th row of the matrix. I cannot perform the Jacobi iteration." << std::endl;
	  return false;
   }
   x[ row ] = ( 1.0 - omega ) * old_x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
   return true;
}


// copy assignment
template< typename Real,
          typename Device,
          typename Index >
Ellpack< Real, Device, Index >&
Ellpack< Real, Device, Index >::operator=( const Ellpack& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2, typename >
Ellpack< Real, Device, Index >&
Ellpack< Real, Device, Index >::operator=( const Ellpack< Real2, Device2, Index2 >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device2, Devices::Host >::value || std::is_same< Device2, Devices::Cuda >::value,
                  "unknown device" );

   // setLike does not work here due to different alignment on Cuda and Host
   this->rowLengths = matrix.rowLengths;
   this->setDimensions( matrix.getRows(), matrix.getColumns() );

   const int blockSize = 32;
   const int blocks = roundUpDivision( this->getRows(), blockSize );

   // host -> cuda
   if( std::is_same< Device, Devices::Cuda >::value ) {
      typename ValuesVector::template Self< typename ValuesVector::ValueType, Devices::Sequential > tmpValues;
      typename ColumnIndexesVector::template Self< typename ColumnIndexesVector::ValueType, Devices::Sequential > tmpColumnIndexes;
      tmpValues.setLike( this->values );
      tmpColumnIndexes.setLike( this->columnIndexes );

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( Index b = 0; b < blocks; b++ ) {
         const Index offset = b * blockSize;
         for( Index j = 0; j < rowLengths; j++ )
            for( Index i = 0; i < blockSize && offset + i < this->getRows(); i++ ) {
               tmpValues[ offset + j * alignedRows + i ] = matrix.values[ ( offset + i ) * rowLengths + j ];
               tmpColumnIndexes[ offset + j * alignedRows + i ] = matrix.columnIndexes[ ( offset + i ) * rowLengths + j ];
            }
      }

      this->values = tmpValues;
      this->columnIndexes = tmpColumnIndexes;
   }

   // cuda -> host
   if( std::is_same< Device, Devices::Host >::value ) {
      ValuesVector tmpValues;
      ColumnIndexesVector tmpColumnIndexes;
      tmpValues.setLike( matrix.values );
      tmpColumnIndexes.setLike( matrix.columnIndexes );
      tmpValues = matrix.values;
      tmpColumnIndexes = matrix.columnIndexes;

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( Index b = 0; b < blocks; b++ ) {
         const Index offset = b * rowLengths;
         for( Index i = 0; i < blockSize && b * blockSize + i < this->getRows(); i++ )
            for( Index j = 0; j < rowLengths; j++ ) {
               this->values[ offset + i * rowLengths + j ] = tmpValues[ b * blockSize + j * matrix.alignedRows + i ];
               this->columnIndexes[ offset + i * rowLengths + j ] = tmpColumnIndexes[ b * blockSize + j * matrix.alignedRows + i ];
            }
      }
   }

   return *this;
}


template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file);
   file.save( &this->rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file);
   file.load( &this->rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType i = DeviceDependentCode::getRowBegin( *this, row );
      const IndexType rowEnd = DeviceDependentCode::getRowEnd( *this, row );
      const IndexType step = DeviceDependentCode::getElementStep( *this );
      while( i < rowEnd &&
             this->columnIndexes.getElement( i ) < this->columns &&
             this->columnIndexes.getElement( i ) != this->getPaddingIndex() )
      {
         const Index column = this->columnIndexes.getElement( i );
         str << " Col:" << column << "->" << this->values.getElement( i ) << "\t";
         i += step;
      }
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void Ellpack< Real, Device, Index >::allocateElements()
{
   const IndexType numMtxElmnts = this->alignedRows * this->rowLengths;

   if( this->alignedRows != 0 )
   {
       TNL_ASSERT_EQ( numMtxElmnts / this->alignedRows, this->rowLengths,
                      "Ellpack cannot store this matrix. The number of matrix elements has overflown the value that IndexType is capable of storing" );
   }

   Sparse< Real, Device, Index >::allocateMatrixElements( numMtxElmnts );
}

template<>
class EllpackDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getRowBegin( const Ellpack< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getRowEnd( const Ellpack< Real, Device, Index >& matrix,
                                const Index row )
      {
         return ( row + 1 ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getElementStep( const Ellpack< Real, Device, Index >& matrix )
      {
         return 1;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Ellpack< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector,
                                 Real multiplicator )
      {
#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector ) * multiplicator;
         /*Index col;
         for( Index row = 0; row < matrix.getRows(); row ++ )
         {
            outVector[ row ] = 0.0;
            const Index rowEnd = ( row + 1 ) * matrix.rowLengths;
            for( Index i = row * matrix.rowLengths; i < rowEnd; i++ )
               if( ( col = matrix.columnIndexes[ i ] ) < matrix.columns )
                  outVector[ row ] += matrix.values[ i ] * inVector[ col ];
         }*/
      }
};

#ifdef HAVE_CUDA
template<
   typename Real,
   typename Index >
__global__ void EllpackVectorProductCudaKernel(
   const Index rows,
   const Index columns,
   const Index compressedRowLengths,
   const Index alignedRows,
   const Index paddingIndex,
   const Index* columnIndexes,
   const Real* values,
   const Real* inVector,
   Real* outVector,
   Real multiplicator,
   const Index gridIdx )
{
   const Index rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx >= rows )
      return;
   Index i = rowIdx;
   Index el( 0 );
   Real result( 0.0 );
   Index columnIndex;
   while( el++ < compressedRowLengths &&
          ( columnIndex = columnIndexes[ i ] ) < columns &&
          columnIndex != paddingIndex )
   {
      result += values[ i ] * inVector[ columnIndex ];
      i += alignedRows;
   }
   outVector[ rowIdx ] = result * multiplicator;
}
#endif



template<>
class EllpackDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getRowBegin( const Ellpack< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row;
      }

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getRowEnd( const Ellpack< Real, Device, Index >& matrix,
                                const Index row )
      {
         return row + getElementStep( matrix ) * matrix.rowLengths;
      }

      template< typename Real,
                typename Index >
      __cuda_callable__
      static Index getElementStep( const Ellpack< Real, Device, Index >& matrix )
      {
         return matrix.alignedRows;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const Ellpack< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector,
                                 Real multiplicator )
      {
         //MatrixVectorProductCuda( matrix, inVector, outVector );
         #ifdef HAVE_CUDA
            typedef Ellpack< Real, Device, Index > Matrix;
            typedef typename Matrix::IndexType IndexType;
            //Matrix* kernel_this = Cuda::passToDevice( matrix );
            //InVector* kernel_inVector = Cuda::passToDevice( inVector );
            //OutVector* kernel_outVector = Cuda::passToDevice( outVector );
            dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
            const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
            const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
            for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
            {
               if( gridIdx == cudaGrids - 1 )
                  cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
               EllpackVectorProductCudaKernel
               < Real, Index >
                <<< cudaGridSize, cudaBlockSize >>>
                ( matrix.getRows(),
                  matrix.getColumns(),
                  matrix.rowLengths,
                  matrix.alignedRows,
                  matrix.getPaddingIndex(),
                  matrix.columnIndexes.getData(),
                  matrix.values.getData(),
                  inVector.getData(),
                  outVector.getData(),
                  multiplicator,
                  gridIdx );
               TNL_CHECK_CUDA_DEVICE;
            }
            //Cuda::freeFromDevice( kernel_this );
            //Cuda::freeFromDevice( kernel_inVector );
            //Cuda::freeFromDevice( kernel_outVector );
            TNL_CHECK_CUDA_DEVICE;
            cudaDeviceSynchronize();
         #endif
      }
};

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL
