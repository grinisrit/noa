#pragma once

#include <type_traits>
#include <ostream>

#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real, typename Index >
class SparseRow
{
   using RealType = Real;
   using IndexType = Index;

   public:

      __cuda_callable__
      SparseRow();

      __cuda_callable__
      SparseRow( Index* columns,
                          Real* values,
                          const Index length,
                          const Index step );

      __cuda_callable__
      void bind( Index* columns,
                 Real* values,
                 const Index length,
                 const Index step );

      __cuda_callable__
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );

      __cuda_callable__
      const Index& getElementColumn( const Index& elementIndex ) const;

      __cuda_callable__
      const Index& getColumnIndex( const Index& elementIndex ) const
      {
         return getElementColumn( elementIndex );
      }


      __cuda_callable__
      const Real& getElementValue( const Index& elementIndex ) const;

      __cuda_callable__
      const Real& getValue( const Index& elementIndex ) const
      {
         return getElementValue( elementIndex );
      }


      __cuda_callable__
      Index getLength() const;

      __cuda_callable__
      Index getSize() const { return length; }


      __cuda_callable__
      Index getNonZeroElementsCount() const;

      void print( std::ostream& str ) const;

   protected:

      Real* values;

      Index* columns;

      Index length, step;
};

template< typename Real, typename Index >
std::ostream& operator<<( std::ostream& str, const SparseRow< Real, Index >& row )
{
   row.print( str );
   return str;
}

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include "SparseRow_impl.h"
