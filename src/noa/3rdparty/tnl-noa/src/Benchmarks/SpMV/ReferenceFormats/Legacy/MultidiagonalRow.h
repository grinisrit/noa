#pragma once

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real, typename Index >
class MultidiagonalRow
{
   public:

      __cuda_callable__
      MultidiagonalRow();

      __cuda_callable__
      MultidiagonalRow( Real* values,
                                 Index* diagonals,
                                 const Index maxRowLength,
                                 const Index row,
                                 const Index columns,
                                 const Index step );

      __cuda_callable__
      void bind( Real* values,
                 Index* diagonals,
                 const Index maxRowLength,
                 const Index row,
                 const Index columns,
                 const Index step );

      __cuda_callable__
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );

   protected:

      Real* values;

      Index* diagonals;

      Index row, columns, maxRowLength, step;
};

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include <TNL/Benchmarks/SpMV/ReferenceFormats/Legacy/MultidiagonalRow_impl.h>

