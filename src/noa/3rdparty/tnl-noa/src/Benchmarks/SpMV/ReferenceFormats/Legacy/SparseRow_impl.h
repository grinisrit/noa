#pragma once

#include "SparseRow.h"
#include <TNL/Exceptions/NotImplementedError.h>

// Following includes are here to enable usage of std::vector and std::cout. To avoid having to include Device type (HOW would this be done anyway)
#include <iostream>
#include <vector>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real, typename Index >
__cuda_callable__
SparseRow< Real, Index >::
SparseRow()
: values( 0 ),
  columns( 0 ),
  length( 0 ),
  step( 0 )
{
}

template< typename Real, typename Index >
__cuda_callable__
SparseRow< Real, Index >::
SparseRow( Index* columns,
                    Real* values,
                    const Index length,
                    const Index step )
: values( values ),
  columns( columns ),
  length( length ),
  step( step )
{
}

template< typename Real, typename Index >
__cuda_callable__
void
SparseRow< Real, Index >::
bind( Index* columns,
      Real* values,
      const Index length,
      const Index step )
{
   this->columns = columns;
   this-> values = values;
   this->length = length;
   this->step = step;
}

template< typename Real, typename Index >
__cuda_callable__
void
SparseRow< Real, Index >::
setElement( const Index& elementIndex,
            const Index& column,
            const Real& value )
{
   TNL_ASSERT( this->columns, );
   TNL_ASSERT( this->values, );
   TNL_ASSERT( this->step > 0,);
   //printf( "elementIndex = %d length = %d \n", elementIndex, this->length );
   TNL_ASSERT( elementIndex >= 0 && elementIndex < this->length,
              std::cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   this->columns[ elementIndex * step ] = column;
   this->values[ elementIndex * step ] = value;
}

template< typename Real, typename Index >
__cuda_callable__
const Index&
SparseRow< Real, Index >::
getElementColumn( const Index& elementIndex ) const
{
   TNL_ASSERT( elementIndex >= 0 && elementIndex < this->length,
              std::cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   return this->columns[ elementIndex * step ];
}

template< typename Real, typename Index >
__cuda_callable__
const Real&
SparseRow< Real, Index >::
getElementValue( const Index& elementIndex ) const
{
   TNL_ASSERT( elementIndex >= 0 && elementIndex < this->length,
              std::cerr << "elementIndex = " << elementIndex << " this->length = " << this->length );

   return this->values[ elementIndex * step ];
}

template< typename Real, typename Index >
__cuda_callable__
Index
SparseRow< Real, Index >::
getLength() const
{
   return length;
}

//#ifdef __CUDACC__
#if 0
template< typename MatrixRow, typename Index >
__global__
void getNonZeroRowLengthCudaKernel( const MatrixRow row, Index* result )
{
//    TODO: Fix/Implement
    throw Exceptions::NotImplementedError( "TODO: Fix/Implement" );
//    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//    if( threadId == 0 )
//    {
//       *result = row.getNonZeroElementsCount();
//    }
}
#endif

template< typename Real, typename Index >
__cuda_callable__
Index
SparseRow< Real, Index >::
getNonZeroElementsCount() const
{
//    TODO: Fix/Implement
    throw Exceptions::NotImplementedError( "TODO: Fix/Implement" );
//    using NonConstIndex = typename std::remove_const< Index >::type;
//
//    NonConstIndex elementCount ( 0 );
//
//    for( NonConstIndex i = 0; i < length; i++ )
//    {
////        std::cout << "this->values[ i * step ] = " << this->values[ i * step ] << " != 0.0" << std::endl;
//        if( this->values[ i * step ] != 0.0 ) // Returns the same amount of elements in a row as does getRowLength() in ChunkedEllpack. WHY?
//            elementCount++;
//    }
//
////    std::cout << "Element Count = " << elementCount << "\n";
//
//    return elementCount;
}

template< typename Real, typename Index >
void
SparseRow< Real, Index >::
print( std::ostream& str ) const
{
   using NonConstIndex = typename std::remove_const< Index >::type;
   NonConstIndex pos( 0 );
   for( NonConstIndex i = 0; i < length; i++ )
   {
      str << " [ " << columns[ pos ] << " ] = " << values[ pos ] << ", ";
      pos += step;
   }
}

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL
