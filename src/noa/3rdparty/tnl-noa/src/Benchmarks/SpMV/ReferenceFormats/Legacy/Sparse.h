#pragma once

#include <TNL/Matrices/Matrix.h>
#include "SparseRow.h"

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real,
          typename Device,
          typename Index >
class Sparse : public TNL::Matrices::Matrix< Real, Device, Index >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename TNL::Matrices::Matrix< RealType, DeviceType, IndexType >::ValuesType ValuesVector;
   typedef Containers::Vector< IndexType, DeviceType, IndexType > ColumnIndexesVector;
   typedef TNL::Matrices::Matrix< Real, Device, Index > BaseType;
   typedef SparseRow< RealType, IndexType > MatrixRow;
   typedef SparseRow< const RealType, const IndexType > ConstMatrixRow;

   Sparse();

   template< typename Real2, typename Device2, typename Index2 >
   void setLike( const Sparse< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowLength() const;

   __cuda_callable__
   IndexType getPaddingIndex() const;

   void reset();

   void save( File& file ) const override;

   void load( File& file ) override;

   void printStructure( std::ostream& str ) const;

   protected:

   void allocateMatrixElements( const IndexType& numberOfMatrixElements );

   Containers::Vector< Index, Device, Index > columnIndexes;

   Index maxRowLength;
};

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include "Sparse_impl.h"
#include <TNL/Matrices/SparseOperations.h>
