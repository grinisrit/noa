#include <iostream>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void forRowsExample()
{
   using MatrixType = TNL::Matrices::DenseMatrix< double, Device >;
   using RowView = typename MatrixType::RowView;
   const int size = 5;
   MatrixType matrix( size, size );
   auto view = matrix.getView();

   /***
    * Set the matrix elements.
    */
   auto f = [=] __cuda_callable__ ( RowView& row ) mutable {
      const int& rowIdx = row.getRowIndex();
      if( rowIdx > 0 )
         row.setValue( rowIdx - 1, -1.0 );
      row.setValue( rowIdx, rowIdx + 1.0 );
      if( rowIdx < size - 1 )
         row.setValue( rowIdx + 1, -1.0 );
   };
   view.forAllRows( f );
   std::cout << matrix << std::endl;

   /***
    * Now divide each matrix row by its largest element - with the use of iterators.
    */
   view.forAllRows( [=] __cuda_callable__ ( RowView& row ) mutable {
      double largest = std::numeric_limits< double >::lowest();
      for( auto element : row )
         largest = TNL::max( largest, element.value() );
      for( auto element : row )
         element.value() /= largest;
   } );
   std::cout << "Divide each matrix row by its largest element... " << std::endl;
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   forRowsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   forRowsExample< TNL::Devices::Cuda >();
#endif
}
