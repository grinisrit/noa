#include <Benchmarks/SpMV/ReferenceFormats/Legacy/AdEllpack.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include <TNL/TypeInfo.h>

#pragma once

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
                namespace Legacy {

/*
 * Auxiliary list implementation
 */
template< typename MatrixType >
warpList< MatrixType >::warpList()
{
    this->head = new warpInfo< MatrixType >;
    this->tail = new warpInfo< MatrixType >;
    this->head->previous = NULL;
    this->head->next = this->tail;
    this->tail->previous = this->head;
    this->tail->next = NULL;

    this->numberOfWarps = 0;
}

template< typename MatrixType >
bool warpList< MatrixType >::addWarp( const IndexType offset,
                                      const IndexType rowOffset,
                                      const IndexType localLoad,
                                      const IndexType* reduceMap )
{
    warpInfo< MatrixType >* temp = new warpInfo< MatrixType >();
    if( !temp )
        return false;
    temp->offset = offset;
    temp->rowOffset = rowOffset;
    temp->localLoad = localLoad;
    for( int i = 0; i < 32; i++ )
        temp->reduceMap[ i ] = reduceMap[ i ];
    temp->next = this->tail;
    temp->previous = this->tail->previous;
    temp->previous->next = temp;
    this->tail->previous = temp;

    this->numberOfWarps++;

    return true;
}

template< typename MatrixType >
warpInfo< MatrixType >* warpList< MatrixType >::splitInHalf( warpInfo< MatrixType >* warp )
{
    warpInfo< MatrixType >* firstHalf = new warpInfo< MatrixType >();
    warpInfo< MatrixType >* secondHalf = new warpInfo< MatrixType >();

    IndexType localLoad = ( warp->localLoad / 2 ) + ( warp->localLoad % 2 == 0 ? 0 : 1 );

    IndexType rowOffset = warp->rowOffset;

    // first half split
    firstHalf->localLoad = localLoad;
    firstHalf->rowOffset = warp->rowOffset;
    firstHalf->offset = warp->offset;

    firstHalf->reduceMap[ 0 ] = 1;
    firstHalf->reduceMap[ 1 ] = 0;
    for( int i = 1; i < 16; i++ )
    {
        if( warp->reduceMap[ i ] == 1 )
        {
            rowOffset++;
            firstHalf->reduceMap[ 2 * i ] = 1;
            firstHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
        else
        {
            firstHalf->reduceMap[ 2 * i ] = 0;
            firstHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
    }

    // second half split
    secondHalf->rowOffset = rowOffset;
    if( warp->reduceMap[ 16 ] == 1 )
        secondHalf->rowOffset = rowOffset + 1;
    secondHalf->offset = 32 * firstHalf->localLoad + firstHalf->offset;
    secondHalf->reduceMap[ 0 ] = 1;
    secondHalf->reduceMap[ 1 ] = 0;
    for( int i = 1; i < 16; i++ )
    {
        if( warp->reduceMap[ i + 16 ] == 1 )
        {
            secondHalf->reduceMap[ 2 * i ] = 1;
            secondHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
        else
        {
            secondHalf->reduceMap[ 2 * i ] = 0;
            secondHalf->reduceMap[ 2 * i + 1 ] = 0;
        }
    }
    secondHalf->localLoad = localLoad;

    // and warps must be connected to the list
    firstHalf->next = secondHalf;
    secondHalf->previous = firstHalf;
    firstHalf->previous = warp->previous;
    warp->previous->next = firstHalf;
    secondHalf->next = warp->next;
    warp->next->previous = secondHalf;

    // if original load was odd, all next warp offsets must be changed!
    if( ( warp->localLoad % 2 ) != 0 )
    {
        warp = secondHalf->next;
        while( warp != this->tail )
        {
            warp->offset = warp->previous->offset + warp->previous->localLoad * 32;
            warp = warp->next;
        }
    }
    this->numberOfWarps++;

    // method returns the first of the two splitted warps
    return firstHalf;
}

template< typename MatrixType >
warpList< MatrixType >::~warpList()
{
    while( this->head->next != NULL )
    {
        warpInfo< MatrixType >* temp = new warpInfo< MatrixType >;
        temp = this->head->next;
        this->head->next = temp->next;
        delete temp;
    }
    delete this->head;
}


/*
 * Adaptive Ellpack implementation
 */

template< typename Real,
          typename Device,
          typename Index >
AdEllpack< Real, Device, Index >::AdEllpack()
:
warpSize( 32 )
{}

template< typename Real,
          typename Device,
          typename Index >
void
AdEllpack< Real, Device, Index >::
setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths )
{

    TNL_ASSERT( this->getRows() > 0, );
    TNL_ASSERT( this->getColumns() > 0, );

    if( std::is_same< DeviceType, Devices::Host >::value )
    {

        RealType average = 0.0;
        for( IndexType row = 0; row < this->getRows(); row++ )
           average += rowLengths.getElement( row );
        average /= ( RealType ) this->getRows();
        this->totalLoad = average;

        warpList< AdEllpack >* list = new warpList< AdEllpack >();

        if( !this->balanceLoad( average, rowLengths, list ) )
            throw 0; // TODO: Make better exception

        IndexType SMs = 15;
        IndexType threadsPerSM = 2048;

        this->computeWarps( SMs, threadsPerSM, list );

        if( !this->createArrays( list ) )
            throw 0; // TODO: Make better excpetion
    }

    if( std::is_same< DeviceType, Devices::Cuda >::value )
    {

        AdEllpack< RealType, Devices::Host, IndexType > hostMatrix;
        hostMatrix.setDimensions( this->getRows(), this->getColumns() );
        Containers::Vector< IndexType, Devices::Host, IndexType > hostRowLengths;
        hostRowLengths.setLike( rowLengths );
        hostRowLengths = rowLengths;
        hostMatrix.setCompressedRowLengths( hostRowLengths );

        this->offset.setLike( hostMatrix.offset );
        this->offset = hostMatrix.offset;
        this->rowOffset.setLike( hostMatrix.rowOffset );
        this->rowOffset = hostMatrix.rowOffset;
        this->localLoad.setLike( hostMatrix.localLoad );
        this->localLoad = hostMatrix.localLoad;
        this->reduceMap.setLike( hostMatrix.reduceMap );
        this->reduceMap = hostMatrix.reduceMap;
        this->totalLoad = hostMatrix.getTotalLoad();

        this->allocateMatrixElements( this->offset.getElement( this->offset.getSize() - 1 ) );
    }
}

template< typename Real,
          typename Device,
          typename Index >
void
AdEllpack< Real, Device, Index >::
setRowCapacities( ConstRowsCapacitiesTypeView rowLengths )
{
   setCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::getCompressedRowLengths( RowsCapacitiesTypeView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index >
Index AdEllpack< Real, Device, Index >::getTotalLoad() const
{
    return this->totalLoad;
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::performRowLengthsTest( ConstRowsCapacitiesTypeView rowLengths )
{
    bool found = false;
    for( IndexType row = 0; row < this->getRows(); row++ )
    {
        found = false;
        IndexType warp = this->getWarp( row );
        IndexType inWarpOffset = this->getInWarpOffset( row, warp );
        if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
            warp++;
        IndexType rowLength = this->localLoad.getElement( warp );
        while( !found )
        {
            if( ( inWarpOffset < this->warpSize - 1 ) &&
                ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
		{
                    inWarpOffset++;
		    rowLength += this->localLoad.getElement( warp );
		}
            else if( ( inWarpOffset == this->warpSize - 1 ) &&
                     ( this->rowOffset.getElement( warp + 1 ) == row ) )
            {
                warp++;
                inWarpOffset = 0;
		rowLength += localLoad.getElement( warp );
            }
            else
                found = true;
        }
	if( rowLength < rowLengths.getElement( row ) )
	    std::cout << "Row: " << row << " is short!";
    }
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::performRowTest()
{
    for( IndexType warp = 0; warp < this->localLoad.getSize(); warp++ )
    {
	IndexType row = this->rowOffset.getElement( warp );
        for( IndexType i = warp * this->warpSize + 1; i < ( warp + 1 ) * this->warpSize; i++ )
	{
	    if( this->reduceMap.getElement( i ) == 1 )
		row++;
	}
	if( row == this->rowOffset.getElement( warp + 1 ) || row + 1 == this->rowOffset.getElement( warp + 1 ) )
	    ;
	else
        {
	    std::cout << "Error warp = " << warp << std::endl;
	    std::cout << "Row: " << row << ", Row offset: " << this->rowOffset.getElement( warp + 1 ) << std::endl;
        }
    }
}

template< typename Real,
          typename Device,
          typename Index >
Index AdEllpack< Real, Device, Index >::getWarp( const IndexType row ) const
{
    for( IndexType searchedWarp = 0; searchedWarp < this->getRows(); searchedWarp++ )
    {
        if( ( this->rowOffset.getElement( searchedWarp ) == row ) ||
            ( ( this->rowOffset.getElement( searchedWarp ) < row ) && ( this->rowOffset.getElement( searchedWarp + 1 ) >= row ) ) )
            return searchedWarp;
    }
    // FIXME: non-void function always has to return something sensible
    throw "bug - row was not found";
}

template< typename Real,
          typename Device,
          typename Index >
Index AdEllpack< Real, Device, Index >::getInWarpOffset( const IndexType row,
                                                                  const IndexType warp ) const
{
    IndexType inWarpOffset = warp * this->warpSize;
    while( ( inWarpOffset < ( warp + 1 ) * this->warpSize ) && ( this->reduceMap.getElement( inWarpOffset ) < row ) )
        inWarpOffset++;
    return ( inWarpOffset & ( this->warpSize - 1 ) );
}

template< typename Real,
          typename Device,
          typename Index >
Index AdEllpack< Real, Device, Index >::getRowLength( const IndexType row ) const
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType length = 0;
    IndexType elementPtr;
    while( !found )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
        {
            if( this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
                length++;
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
            inWarpOffset++;
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return length;
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Real2,
          typename Device2,
          typename Index2 >
void AdEllpack< Real, Device, Index >::setLike( const AdEllpack< Real2, Device2, Index2 >& matrix )
{
    Sparse< Real, Device, Index >::setLike( matrix );
    this->offset.setLike( matrix.offset );
    this->rowOffset.setLike( matrix.rowOffset );
    this->localLoad.setLike( matrix.localLoad );
    this->reduceMap.setLike( matrix.reduceMap );
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::reset()
{
    Sparse< Real, Device, Index >::reset();
    this->offset.reset();
    this->rowOffset.reset();
    this->localLoad.reset();
    this->reduceMap.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool AdEllpack< Real, Device, Index >::operator == ( const AdEllpack< Real2, Device2, Index2 >& matrix ) const
{
   TNL_ASSERT( this->getRows() == matrix.getRows() &&
               this->getColumns() == matrix.getColumns(),
               std::cerr << "this->getRows() = " << this->getRows()
                    << " matrix.getRows() = " << matrix.getRows()
                    << " this->getColumns() = " << this->getColumns()
                    << " matrix.getColumns() = " << matrix.getColumns() );

   TNL_ASSERT_TRUE( false, "operator == is not yet implemented for AdEllpack.");

   // TODO: implement this
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool AdEllpack< Real, Device, Index >::operator != ( const AdEllpack< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index >
bool AdEllpack< Real, Device, Index >::setElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value )
{
    return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool AdEllpack< Real, Device, Index >::addElement( const IndexType row,
                                                            const IndexType column,
                                                            const RealType& value,
                                                            const RealType& thisElementMultiplicator )
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType elementPtr;
    IndexType iterator = 0;
    while( !found )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
        {
            iterator++;
            if( this->columnIndexes.getElement( elementPtr ) == column )
            {
                this->values.setElement( elementPtr, thisElementMultiplicator * this->values.getElement( elementPtr ) + value );
                return true;
            }
            if( this->columnIndexes.getElement( elementPtr ) == this->getPaddingIndex() )
            {
                this->columnIndexes.setElement( elementPtr, column );
                this->values.setElement( elementPtr, value );
                return true;
            }
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
	{
            inWarpOffset++;
	}
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool AdEllpack< Real, Device, Index >::setRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType elementPtr;
    IndexType elPtr = 0;
    while( ( !found ) && ( elPtr < elements ) )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; ( i < this->localLoad.getElement( warp ) ) && ( elPtr < elements ); i++ )
        {
            this->values.setElement( elementPtr, values[ elPtr ] );
            this->columnIndexes.setElement( elementPtr, columnIndexes[ elPtr ] );
            elPtr++;
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
            inWarpOffset++;
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return true;
}

template< typename Real,
          typename Device,

          typename Index >
bool AdEllpack< Real, Device, Index >::addRow( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements,
                                                        const RealType& thisElementMultiplicator )
{
    return false;
}

template< typename Real,
          typename Device,
          typename Index >
Real AdEllpack< Real, Device, Index >::getElement( const IndexType row,
                                                            const IndexType column ) const
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType elementPtr;
    while( !found )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
        {
            if( this->columnIndexes.getElement( elementPtr ) == column )
                return this->values.getElement( elementPtr );
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
            inWarpOffset++;
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
    return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::getRow( const IndexType row,
                                                        IndexType* columns,
                                                        RealType* values ) const
{
    IndexType warp = this->getWarp( row );
    IndexType inWarpOffset = this->getInWarpOffset( row, warp );
    if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
        warp++;

    bool found = false;
    IndexType arrayElementPtr = 0;
    IndexType elementPtr;
    while( !found )
    {
        elementPtr = this->offset.getElement( warp ) + inWarpOffset;
        for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
        {
            if( this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
            {
                columns[ arrayElementPtr ] = this->columnIndexes.getElement( elementPtr );
                values[ arrayElementPtr ] = this->values.getElement( elementPtr );
                arrayElementPtr++;
            }
            elementPtr += this->warpSize;
        }
        if( ( inWarpOffset < this->warpSize - 1 ) &&
            ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
            inWarpOffset++;
        else if( ( inWarpOffset == this->warpSize - 1 ) &&
                 ( this->rowOffset.getElement( warp + 1 ) == row ) )
        {
            warp++;
            inWarpOffset = 0;
        }
        else
            found = true;
    }
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
void AdEllpack< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                      OutVector& outVector ) const
{
    DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

// copy assignment
template< typename Real,
          typename Device,
          typename Index >
AdEllpack< Real, Device, Index >&
AdEllpack< Real, Device, Index >::operator=( const AdEllpack& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->offset = matrix.offset;
   this->rowOffset = matrix.rowOffset;
   this->localLoad = matrix.localLoad;
   this->reduceMap = matrix.reduceMap;
   this->totalLoad = matrix.totalLoad;
   this->warpSize = matrix.warpSize;
   return *this;
}


// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2, typename >
AdEllpack< Real, Device, Index >&
AdEllpack< Real, Device, Index >::operator=( const AdEllpack< Real2, Device2, Index2 >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device2, Devices::Host >::value || std::is_same< Device2, Devices::Cuda >::value,
                  "unknown device" );

   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->offset = matrix.offset;
   this->rowOffset = matrix.rowOffset;
   this->localLoad = matrix.localLoad;
   this->reduceMap = matrix.reduceMap;
   this->totalLoad = matrix.totalLoad;
   this->warpSize = matrix.warpSize;

   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::save( File& file ) const
{
    Sparse< Real, Device, Index >::save( file );
    file << this->offset << this->rowOffset << this->localLoad << this->reduceMap;
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::load( File& file )
{
    Sparse< Real, Device, Index >::load( file );
    file >> this->offset >> this->rowOffset >> this->localLoad >> this->reduceMap;
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::save( const String& fileName ) const
{
    Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::load( const String& fileName )
{
    Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::print( std::ostream& str ) const
{
    for( IndexType row = 0; row < this->getRows(); row++ )
    {
        str  << "Row: " << row << " -> ";

        IndexType warp = this->getWarp( row );
        IndexType inWarpOffset = this->getInWarpOffset( row, warp );
        if( inWarpOffset == 0 && rowOffset.getElement( warp ) != row )
            warp++;

        bool found = false;
        IndexType elementPtr;
        while( !found )
        {
            elementPtr = this->offset.getElement( warp ) + inWarpOffset;
            for( IndexType i = 0; i < this->localLoad.getElement( warp ); i++ )
            {
                if( this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
                    str << " Col:" << this->columnIndexes.getElement( elementPtr ) << "->"
                        << this->values.getElement( elementPtr ) << "\t";
                elementPtr += this->warpSize;
            }
            if( ( inWarpOffset < this->warpSize - 1 ) &&
                ( this->reduceMap.getElement( this->warpSize * warp + inWarpOffset + 1 ) == row ) )
                inWarpOffset++;
            else if( ( inWarpOffset == this->warpSize - 1 ) &&
                     ( this->rowOffset.getElement( warp + 1 ) == row ) )
            {
                warp++;
                inWarpOffset = 0;
            }
            else
                found = true;
        }
        str << std::endl;
    }
}

template< typename Real,
          typename Device,
          typename Index >
bool AdEllpack< Real, Device, Index >::balanceLoad( const RealType average,
                                                    ConstRowsCapacitiesTypeView rowLengths,
                                                    warpList< AdEllpack >* list )
{
    IndexType offset, rowOffset, localLoad, reduceMap[ 32 ];

    IndexType numberOfThreads = 0;
    IndexType ave = average * 1.1;
    offset = 0;
    rowOffset = 0;
    localLoad = 0;
    bool addedWarp = false;
    IndexType warpId = 0;

    for( IndexType row = 0; row < this->getRows(); row++ )
    {
        addedWarp = false;
        // takes every row
        IndexType rowLength = rowLengths.getElement( row );
        // counts necessary number of threads
        IndexType threadsPerRow = ( IndexType ) ( rowLength / ave ) + ( rowLength % ave == 0 ? 0 : 1 );
        // if this number added to number of total threads in this warp is higher than 32
        // then spreads them in as many as necessary

        while( numberOfThreads + threadsPerRow >= this->warpSize )
        {
            // counts number of usable threads
            IndexType usedThreads = this->warpSize - numberOfThreads;

            // if this condition applies we can finish the warp
            if( usedThreads == threadsPerRow )
            {
                localLoad = max( localLoad, roundUpDivision( rowLength, threadsPerRow ) );

                reduceMap[ numberOfThreads ] = 1;
                for( IndexType i = numberOfThreads + 1; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;

                if( !list->addWarp( offset, rowOffset, localLoad, reduceMap ) )
                    return false;

                offset += this->warpSize * localLoad;
                rowOffset = row + 1;

                // IMPORTANT TO RESET VARIABLES
                localLoad = 0;
                for( IndexType i = 0; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;
                numberOfThreads = 0;
                threadsPerRow = 0;
                addedWarp = true;
		warpId++;
            }

            // if it doesnt apply and if local load isn't equal 0 it will use old local load ( for better balance )
            else if( localLoad != 0 )
            {
                // subtract unmber of used elements and number of used threads
                rowLength -= localLoad * usedThreads;

                threadsPerRow = ( IndexType ) ( rowLength / ave ) + ( rowLength % ave == 0 ? 0 : 1 );

                // fill the reduction map
                reduceMap[ numberOfThreads ] = 1;
                for( IndexType i = numberOfThreads + 1; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;

                // count new offsets, add new warp and reset variables
                if( !list->addWarp( offset, rowOffset, localLoad, reduceMap ) )
                    return false;
                offset += this->warpSize * localLoad;
                rowOffset = row;

                // RESET VARIABLES
                localLoad = 0;
                for( IndexType i = 0; i < this->warpSize; i++ )
                    reduceMap[ i ] = 0;
                numberOfThreads = 0;
                addedWarp = true;
		warpId++;
            }
            // otherwise we are starting a new warp
            else
            {
                threadsPerRow = ( IndexType ) ( rowLength / ave ) + ( rowLength % ave == 0 ? 0 : 1 );
                if( threadsPerRow < this->warpSize )
                    break;

                localLoad = ave;
            }
        }
	if( threadsPerRow <= 0 )
	{
	    threadsPerRow = 1;
	    continue;
	}
        localLoad = max( localLoad, roundUpDivision( rowLength, threadsPerRow ) );
        reduceMap[ numberOfThreads ] = 1;
        for( IndexType i = numberOfThreads + 1; i < numberOfThreads + threadsPerRow; i++ )
            reduceMap[ i ] = 0;

        numberOfThreads += threadsPerRow;

        // if last row doesnt fill the whole warp or it fills more warps and threads still remain
        if( ( ( row == this->getRows() - 1 ) && !addedWarp ) ||
            ( ( row == this->getRows() - 1 ) && ( threadsPerRow == numberOfThreads ) && ( numberOfThreads > 0 ) ) )
        {
            list->addWarp( offset, rowOffset, localLoad, reduceMap );
        }
    }
    return true;
}

template< typename Real,
          typename Device,
          typename Index >
void AdEllpack< Real, Device, Index >::computeWarps( const IndexType SMs,
                                                     const IndexType threadsPerSM,
                                                     warpList< AdEllpack >* list )
{
    IndexType averageLoad = 0;
    warpInfo< AdEllpack >* temp = list->getHead()->next;

    while( temp/*->next*/ != list->getTail() )
    {
        averageLoad += temp->localLoad;
        temp = temp->next;
    }
    averageLoad = roundUpDivision( averageLoad, list->getNumberOfWarps() );

    IndexType totalWarps = SMs * ( threadsPerSM / this->warpSize );
    IndexType remainingThreads = list->getNumberOfWarps();
    bool warpsToSplit = true;

    while( remainingThreads < ( totalWarps / 2 ) && warpsToSplit )
    {
        warpsToSplit = false;
        temp = list->getHead()->next;
        while( temp != list->getTail() )
        {
            if( temp->localLoad > averageLoad )
            {
                temp = list->splitInHalf( temp );
                warpsToSplit = true;
            }
            temp = temp->next;
        }
	remainingThreads = list->getNumberOfWarps();
    }
}

template< typename Real,
          typename Device,
          typename Index >
bool AdEllpack< Real, Device, Index >::createArrays( warpList< AdEllpack >* list )
{
    IndexType length = list->getNumberOfWarps();

    this->offset.setSize( length + 1 );
    this->rowOffset.setSize( length + 1 );
    this->localLoad.setSize( length );
    this->reduceMap.setSize( length * this->warpSize );

    IndexType iteration = 0;
    warpInfo< AdEllpack >* warp = list->getHead()->next;
    while( warp != list->getTail() )
    {
        this->offset.setElement( iteration, warp->offset );
        this->rowOffset.setElement( iteration, warp->rowOffset );
        this->localLoad.setElement( iteration, warp->localLoad );
	IndexType row = this->rowOffset.getElement( iteration );
	this->reduceMap.setElement( iteration * this->warpSize, row );
        for( int i = iteration * this->warpSize + 1; i < ( iteration + 1 ) * this->warpSize; i++ )
	{
	    if( warp->reduceMap[ i & ( this->warpSize - 1 ) ] == 1 )
		row++;
            this->reduceMap.setElement( i, row );
	}
   // TODO: Fix, operator > is not defined for vectors
	//if( localLoad > this->totalLoad )
	//    std::cout << "Error localLoad!!" << std::endl;
        iteration++;
        warp = warp->next;
    }
    this->rowOffset.setElement( length, this->getRows() );
    this->offset.setElement( length, this->offset.getElement( length - 1 ) + this->warpSize * this->localLoad.getElement( length - 1 ) );
    this->allocateMatrixElements( this->offset.getElement( length ) );

    return true;
}

template<>
class AdEllpackDeviceDependentCode< Devices::Host >
{
public:

    typedef Devices::Host Device;

    template< typename Real,
              typename Index,
              typename InVector,
              typename OutVector >
    static void vectorProduct( const AdEllpack< Real, Device, Index >& matrix,
                               const InVector& inVector,
                               OutVector& outVector )
    {
	// parallel vector product simulation
	const Index blockSize = 256;
	const Index blocks = ( Index ) ( matrix.reduceMap.getSize() / blockSize ) + ( matrix.reduceMap.getSize() % blockSize != 0 );
	for( Index block = 0; block < blocks; block++ )
	{
	    Containers::Vector< Real, Device, Index > temp, reduceMap;
	    temp.setSize( blockSize );
	    reduceMap.setSize( blockSize );
	    for( Index threadIdx = 0; threadIdx < blockSize; threadIdx++ )
	    {
		Index globalIdx = block * blockSize + threadIdx;
		Index warpIdx = globalIdx >> 5;
		Index inWarpIdx = globalIdx & ( matrix.warpSize - 1 );
		if( globalIdx >= matrix.reduceMap.getSize() )
		{
		    for( Index i = globalIdx % blockSize; i < blockSize; i++ )
			reduceMap.setElement( i, -1 );
		    break;
		}
		temp.setElement( threadIdx, 0.0 );
		reduceMap.setElement( threadIdx, matrix.reduceMap.getElement( globalIdx ) );
		Index elementPtr = matrix.offset.getElement( warpIdx ) + inWarpIdx;
		for( Index i = 0; i < matrix.localLoad.getElement( warpIdx ); i++ )
		{
		    if( matrix.columnIndexes.getElement( elementPtr ) != matrix.getPaddingIndex() )
		        temp.setElement( threadIdx, temp.getElement( threadIdx ) + matrix.values.getElement( elementPtr ) * inVector[ matrix.columnIndexes.getElement( elementPtr ) ] );
		    elementPtr += matrix.warpSize;
		}
	    }
	    for( Index threadIdx = 0; threadIdx < blockSize; threadIdx++ )
	    {
		Index globalIdx = block * blockSize + threadIdx;
		Index warpIdx = globalIdx >> 5;
		Index inWarpIdx = globalIdx & ( matrix.warpSize - 1 );
		if( globalIdx >= matrix.reduceMap.getSize() )
		    break;
		if( ( inWarpIdx == 0 ) || ( reduceMap.getElement( threadIdx - 1 ) < reduceMap.getElement( threadIdx ) ) )
		{
		    Index end = ( warpIdx + 1 ) << 5;
		    Index elementPtr = threadIdx + 1;
		    globalIdx++;
		    while( globalIdx < end && reduceMap.getElement( elementPtr ) == reduceMap.getElement( threadIdx ) )
		    {
			temp.setElement( threadIdx, temp.getElement( elementPtr ) + temp.getElement( threadIdx ) );
			elementPtr++;
			globalIdx++;
		    }
		    outVector[ reduceMap.getElement( threadIdx ) ] += temp.getElement( threadIdx );
		}
	    }
	}
    }

};

#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void AdEllpack< Real, Device, Index >::spmvCuda2( const InVector& inVector,
                                                  OutVector& outVector,
                                                  const int gridIdx ) const
{
    IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    IndexType warpIdx = globalIdx >> 5;
    IndexType inWarpIdx = globalIdx & ( this->warpSize - 1 );
    if( globalIdx >= this->reduceMap.getSize() )
	return;

    const int blockSize = 256;
    Real* temp = Cuda::getSharedMemory< Real >();
    __shared__ IndexType reduceMap[ blockSize ];
    reduceMap[ threadIdx.x ] = this->reduceMap[ globalIdx ];
    temp[ threadIdx.x ] = 0.0;

    IndexType i = 0;
    IndexType elementPtr = this->offset[ warpIdx ] + inWarpIdx;
    const IndexType warpLoad = this->localLoad[ warpIdx ];

    for( ; i < warpLoad; i++ )
    {
        if( this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
            temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
            elementPtr += this->warpSize;
        }
    }

    if( ( inWarpIdx == 0 ) || ( reduceMap[ threadIdx.x ] > reduceMap[ threadIdx.x - 1 ] ) )
    {
	IndexType elementPtr = threadIdx.x + 1;
        globalIdx++;
        while( globalIdx < ( ( warpIdx + 1 ) << 5 ) &&
               reduceMap[ elementPtr ] == reduceMap[ threadIdx.x ] )
        {
            temp[ threadIdx.x ] += temp[ elementPtr ];
            elementPtr++;
            globalIdx++;
        }
        outVector[ reduceMap[ threadIdx.x] ] += temp[ threadIdx.x ];
    }
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void AdEllpack< Real, Device, Index >::spmvCuda4( const InVector& inVector,
                                                           OutVector& outVector,
                                                           const int gridIdx ) const
{
    IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    IndexType warpIdx = globalIdx >> 5;
    IndexType inWarpIdx = globalIdx & ( this->warpSize - 1 );
    if( globalIdx >= this->reduceMap.getSize() )
	return;

    const int blockSize = 192;
    Real* temp = Cuda::getSharedMemory< Real >();
    __shared__ IndexType reduceMap[ blockSize ];
    reduceMap[ threadIdx.x ] = this->reduceMap[ globalIdx ];
    temp[ threadIdx.x ] = 0.0;

    IndexType i = 0;
    IndexType elementPtr = this->offset[ warpIdx ] + inWarpIdx;
    const IndexType warpLoad = this->localLoad[ warpIdx ];

    if( ( warpLoad & 1 ) == 1 )
    {
        if( this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
            temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
            elementPtr += this->warpSize;
            i++;
        }
    }

    for( ; i < warpLoad; i += 2 )
    {
        #pragma unroll
        for( IndexType j = 0; j < 2; j++ )
        {
            if( this->columnIndexes[ elementPtr ] < this->getColumns() )
            {
                temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
                elementPtr += this->warpSize;
            }
        }
    }

    if( ( inWarpIdx == 0 ) || ( reduceMap[ threadIdx.x ] > reduceMap[ threadIdx.x - 1 ] ) )
    {
        IndexType end = ( warpIdx + 1 ) << 5;
	IndexType elementPtr = threadIdx.x + 1;
        globalIdx++;
        while( globalIdx < end &&
               reduceMap[ elementPtr ] == reduceMap[ threadIdx.x ] )
        {
            temp[ threadIdx.x ] += temp[ elementPtr ];
            elementPtr++;
            globalIdx++;
        }
        outVector[ reduceMap[ threadIdx.x ] ] += temp[ threadIdx.x ];
    }
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void AdEllpack< Real, Device, Index >::spmvCuda8( const InVector& inVector,
                                                           OutVector& outVector,
                                                           const int gridIdx ) const
{
    IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    IndexType warpIdx = globalIdx >> 5;
    IndexType inWarpIdx = globalIdx & ( this->warpSize - 1 );
    if( globalIdx >= this->reduceMap.getSize() )
        return;

    const int blockSize = 128;
    Real* temp = Cuda::getSharedMemory< Real >();
    __shared__ IndexType reduceMap[ blockSize ];
    reduceMap[ threadIdx.x ] = this->reduceMap[ globalIdx ];
    temp[ threadIdx.x ] = 0.0;

    IndexType i = 0;
    IndexType elementPtr = this->offset[ warpIdx ] + inWarpIdx;
    const IndexType warpLoad = this->localLoad[ warpIdx ];

    if( warpLoad < 4 )
    {
        while( i < warpLoad &&
               this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
            temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
            elementPtr += this->warpSize;
            i++;
        }
    }
    else
    {
        IndexType alignUnroll = warpLoad & 3;

        while( alignUnroll != 0 &&
               this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
                temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
                elementPtr += this->warpSize;
                i++;
                alignUnroll--;
        }
    }
    for( ; i < this->localLoad[ warpIdx ]; i += 4 )
    {
        #pragma unroll
        for( IndexType j = 0; j < 4; j++ )
        {
           if( this->columnIndexes[ elementPtr ] < this->getColumns() )
            {
               temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
               elementPtr += this->warpSize;
            }
        }
    }

    if( ( inWarpIdx == 0 ) || ( reduceMap[ threadIdx.x ] > reduceMap[ threadIdx.x - 1 ] ) )
    {
        IndexType end = ( warpIdx + 1 ) << 5;
	IndexType elementPtr = threadIdx.x + 1;
        globalIdx++;
        while( globalIdx < end &&
               reduceMap[ elementPtr ] == reduceMap[ threadIdx.x ] )
        {
            temp[ threadIdx.x ] += temp[ elementPtr ];
            elementPtr++;
            globalIdx++;
        }
        outVector[ reduceMap[ threadIdx.x ] ] += temp[ threadIdx.x ];
    }
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void AdEllpack< Real, Device, Index >::spmvCuda16( const InVector& inVector,
                                                         OutVector& outVector,
                                                         const int gridIdx ) const
{
    IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    IndexType warpIdx = globalIdx >> 5;
    IndexType inWarpIdx = globalIdx & ( this->warpSize - 1 );
    if( globalIdx >= this->reduceMap.getSize() )
        return;

    const int blockSize = 128;
    Real* temp = Cuda::getSharedMemory< Real >();
    __shared__ IndexType reduceMap[ blockSize ];
    reduceMap[ threadIdx.x ] = this->reduceMap[ globalIdx ];
    temp[ threadIdx.x ] = 0.0;

    IndexType i = 0;
    IndexType elementPtr = this->offset[ warpIdx ] + inWarpIdx;
    const IndexType warpLoad = this->localLoad[ warpIdx ];

    if( warpLoad < 8 )
    {
        while( i < warpLoad &&
               this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
            temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
            elementPtr += this->warpSize;
            i++;
        }
    }
    else
    {
        IndexType alignUnroll = warpLoad & 7;

        while( alignUnroll != 0 &&
               this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
            temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
            elementPtr += this->warpSize;
            i++;
            alignUnroll--;
        }
    }

    for( ; i < warpLoad; i += 8 )
    {
        #pragma unroll
        for( IndexType j = 0; j < 8; j++ )
        {
            if( this->columnIndexes[ elementPtr ] < this->getColumns() )
            {
                temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
                elementPtr += this->warpSize;
            }
        }
    }

    if( ( inWarpIdx == 0 ) || ( reduceMap[ threadIdx.x ] > reduceMap[ threadIdx.x - 1 ] ) )
    {
        IndexType end = ( warpIdx + 1 ) << 5;
	IndexType elementPtr = threadIdx.x + 1;
        globalIdx++;
        while( globalIdx < end &&
               reduceMap[ elementPtr ] == reduceMap[ threadIdx.x ] )
        {
            temp[ threadIdx.x ] += temp[ elementPtr ];
            elementPtr++;
            globalIdx++;
        }
        outVector[ reduceMap[ threadIdx.x ] ] += temp[ threadIdx.x ];
    }
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void AdEllpack< Real, Device, Index >::spmvCuda32( const InVector& inVector,
                                                            OutVector& outVector,
                                                            const int gridIdx ) const
{
    IndexType globalIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
    IndexType warpIdx = globalIdx >> 5;
    IndexType inWarpIdx = globalIdx & ( this->warpSize - 1 );
    if( globalIdx >= this->reduceMap.getSize() )
	return;

    const int blockSize = 96;
    Real* temp = Cuda::getSharedMemory< Real >();
    __shared__ IndexType reduceMap[ blockSize ];
    reduceMap[ threadIdx.x ] = this->reduceMap[ globalIdx ];
    temp[ threadIdx.x ] = 0.0;

    IndexType i = 0;
    IndexType elementPtr = this->offset[ warpIdx ] + inWarpIdx;
    const IndexType warpLoad = this->localLoad[ warpIdx ];

    if( warpLoad < 16 )
    {
        while( i < warpLoad &&
               this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
            temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
            elementPtr += this->warpSize;
            i++;
        }
    }
    else
    {
        IndexType alignUnroll = warpLoad & 15;

        while( alignUnroll != 0 &&
               this->columnIndexes[ elementPtr ] < this->getColumns() )
        {
            temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
            elementPtr += this->warpSize;
            i++;
            alignUnroll--;
        }
    }

    for( ; i < warpLoad; i += 16 )
    {
        #pragma unroll
        for( IndexType j = 0; j < 16; j++ )
        {
            if( this->columnIndexes[ elementPtr ] < this->getColumns() )
            {
                temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
                elementPtr += this->warpSize;
            }
        }
    }

    if( ( inWarpIdx == 0 ) || ( reduceMap[ threadIdx.x ] > reduceMap[ threadIdx.x - 1 ] ) )
    {
        IndexType end = ( warpIdx + 1 ) << 5;
	IndexType elementPtr = threadIdx.x + 1;
        globalIdx++;
        while( globalIdx < end &&
               reduceMap[ elementPtr ] == reduceMap[ threadIdx.x ] )
        {
            temp[ threadIdx.x ] += temp[ elementPtr ];
            elementPtr++;
            globalIdx++;
        }
        outVector[ reduceMap[ threadIdx.x] ] += temp[ threadIdx.x ];
    }
}
#endif

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void AdEllpackVectorProductCuda2( const AdEllpack< Real, Devices::Cuda, Index >* matrix,
                                           const InVector* inVector,
                                           OutVector* outVector,
                                           const int gridIdx )
{
    matrix->spmvCuda2( *inVector, *outVector, gridIdx );
}

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void AdEllpackVectorProductCuda4( const AdEllpack< Real, Devices::Cuda, Index >* matrix,
                                           const InVector* inVector,
                                           OutVector* outVector,
                                           const int gridIdx )
{
    matrix->spmvCuda4( *inVector, *outVector, gridIdx );
}

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void AdEllpackVectorProductCuda8( const AdEllpack< Real, Devices::Cuda, Index >* matrix,
                                           const InVector* inVector,
                                           OutVector* outVector,
                                           const int gridIdx )
{
    matrix->spmvCuda8( *inVector, *outVector, gridIdx );
}

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void AdEllpackVectorProductCuda16( const AdEllpack< Real, Devices::Cuda, Index >* matrix,
                                            const InVector* inVector,
                                            OutVector* outVector,
                                            const int gridIdx )
{
    matrix->spmvCuda16( *inVector, *outVector, gridIdx );
}

template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void AdEllpackVectorProductCuda32( const AdEllpack< Real, Devices::Cuda, Index >* matrix,
                                            const InVector* inVector,
                                            OutVector* outVector,
                                            const int gridIdx )
{
    matrix->spmvCuda32( *inVector, *outVector, gridIdx );
}
#endif

template<>
class AdEllpackDeviceDependentCode< Devices::Cuda >
{
public:

    typedef Devices::Cuda Device;

    template< typename Real,
              typename Index,
              typename InVector,
              typename OutVector >
    static void vectorProduct( const AdEllpack< Real, Device, Index >& matrix,
                               const InVector& inVector,
                               OutVector& outVector )
    {
#ifdef HAVE_CUDA
      typedef AdEllpack< Real, Devices::Cuda, Index > Matrix;
      typedef typename Matrix::IndexType IndexType;
	   Matrix* kernel_this = Cuda::passToDevice( matrix );
	   InVector* kernel_inVector = Cuda::passToDevice( inVector );
	   OutVector* kernel_outVector = Cuda::passToDevice( outVector );
      TNL_CHECK_CUDA_DEVICE;

      if( matrix.totalLoad < 2 )
	   {
	    dim3 blockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
	    IndexType cudaBlocks = roundUpDivision( matrix.reduceMap.getSize(), blockSize.x );
	    IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
	    for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
	    {
	        if( gridIdx == cudaGrids - 1 )
		    cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
	        const int sharedMemory = blockSize.x * sizeof( Real );
	        AdEllpackVectorProductCuda2< Real, Index, InVector, OutVector >
                                                    <<< cudaGridSize, blockSize, sharedMemory >>>
                                                    ( kernel_this,
                                                      kernel_inVector,
                                                      kernel_outVector,
                                                      gridIdx );
	    }
	    TNL_CHECK_CUDA_DEVICE;
	    Cuda::freeFromDevice( kernel_this );
	    Cuda::freeFromDevice( kernel_inVector );
	    Cuda::freeFromDevice( kernel_outVector );
	    TNL_CHECK_CUDA_DEVICE;
	}
	else if( matrix.totalLoad < 4 )
	{
	    dim3 blockSize( 192 ), cudaGridSize( Cuda::getMaxGridSize() );
	    IndexType cudaBlocks = roundUpDivision( matrix.reduceMap.getSize(), blockSize.x );
	    IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
	    for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
	    {
	        if( gridIdx == cudaGrids - 1 )
		    cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
	        const int sharedMemory = blockSize.x * sizeof( Real );
	        AdEllpackVectorProductCuda4< Real, Index, InVector, OutVector >
                                                    <<< cudaGridSize, blockSize, sharedMemory >>>
                                                    ( kernel_this,
                                                      kernel_inVector,
                                                      kernel_outVector,
                                                      gridIdx );
	    }
	    TNL_CHECK_CUDA_DEVICE;
	    Cuda::freeFromDevice( kernel_this );
	    Cuda::freeFromDevice( kernel_inVector );
	    Cuda::freeFromDevice( kernel_outVector );
	    TNL_CHECK_CUDA_DEVICE;
	}
	else if( matrix.totalLoad < 8 )
	{
	    dim3 blockSize( 128 ), cudaGridSize( Cuda::getMaxGridSize() );
	    IndexType cudaBlocks = roundUpDivision( matrix.reduceMap.getSize(), blockSize.x );
	    IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
	    for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
	    {
	        if( gridIdx == cudaGrids - 1 )
		    cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
	        const int sharedMemory = blockSize.x * sizeof( Real );
	        AdEllpackVectorProductCuda8< Real, Index, InVector, OutVector >
                                                    <<< cudaGridSize, blockSize, sharedMemory >>>
                                                    ( kernel_this,
                                                      kernel_inVector,
                                                      kernel_outVector,
                                                      gridIdx );
	    }
	    TNL_CHECK_CUDA_DEVICE;
	    Cuda::freeFromDevice( kernel_this );
	    Cuda::freeFromDevice( kernel_inVector );
	    Cuda::freeFromDevice( kernel_outVector );
	    TNL_CHECK_CUDA_DEVICE;
	}
	else if( matrix.totalLoad < 16 )
	{
	    dim3 blockSize( 128 ), cudaGridSize( Cuda::getMaxGridSize() );
	    IndexType cudaBlocks = roundUpDivision( matrix.reduceMap.getSize(), blockSize.x );
	    IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
            for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
	    {
	        if( gridIdx == cudaGrids - 1 )
		    cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
	        const int sharedMemory = blockSize.x * sizeof( Real );
	        AdEllpackVectorProductCuda16< Real, Index, InVector, OutVector >
                                                     <<< cudaGridSize, blockSize, sharedMemory >>>
                                                     ( kernel_this,
                                                       kernel_inVector,
                                                       kernel_outVector,
                                                       gridIdx );
	    }
	    TNL_CHECK_CUDA_DEVICE;
	    Cuda::freeFromDevice( kernel_this );
	    Cuda::freeFromDevice( kernel_inVector );
	    Cuda::freeFromDevice( kernel_outVector );
	    TNL_CHECK_CUDA_DEVICE;
	}
	else
	{
	    dim3 blockSize( 96 ), cudaGridSize( Cuda::getMaxGridSize() );
	    IndexType cudaBlocks = roundUpDivision( matrix.reduceMap.getSize(), blockSize.x );
	    IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
	    for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
	    {
	        if( gridIdx == cudaGrids - 1 )
		    cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
	        const int sharedMemory = blockSize.x * sizeof( Real );
	        AdEllpackVectorProductCuda32< Real, Index, InVector, OutVector >
                                                     <<< cudaGridSize, blockSize, sharedMemory >>>
                                                     ( kernel_this,
                                                       kernel_inVector,
                                                       kernel_outVector,
                                                       gridIdx );
	    }
	    TNL_CHECK_CUDA_DEVICE;
	    Cuda::freeFromDevice( kernel_this );
	    Cuda::freeFromDevice( kernel_inVector );
	    Cuda::freeFromDevice( kernel_outVector );
	    TNL_CHECK_CUDA_DEVICE;
	}
#endif // HAVE_CUDA
   }
};


                } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL
