#include <iostream>
#include <TNL/FileName.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/ODE/Euler.h>
#include "write.h"

using Real = double;
using Index = int;

template< typename Device >
void solveHeatEquation( const char* file_name )
{
   using Vector = TNL::Containers::Vector< Real, Device, Index >;
   using VectorView = typename Vector::ViewType;
   using ODESolver = TNL::Solvers::ODE::Euler< Vector >;

   /***
    * Parameters of the discretisation
    */
   const Real final_t = 0.05;
   const Real output_time_step = 0.005;
   const Index n = 41;
   const Real h = 1.0 / ( n - 1 );
   const Real tau = 0.1 * h * h;
   const Real h_sqr_inv = 1.0 / ( h * h );

   /***
    * Initial condition
    */
   Vector u( n );
   u.forAllElements( [=] __cuda_callable__ ( Index i, Real& value ) {
      const Real x = i * h;
      if( x >= 0.4 && x <= 0.6 )
         value = 1.0;
      else value = 0.0;
   } );
   std::fstream file;
   file.open( file_name, std::ios::out );
   write( file, u, n, h, ( Real ) 0.0 );

   /***
    * Setup of the solver
    */
   ODESolver solver;
   solver.setTau(  tau );
   solver.setTime( 0.0 );

   /***
    * Time loop
    */
   Index output_idx( 1 );
   while( solver.getTime() < final_t )
   {
      solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
      auto f = [=] __cuda_callable__ ( Index i, const VectorView& u, VectorView& fu ) mutable {
         if( i == 0 || i == n-1 )                // boundary nodes -> boundary conditions
            fu[ i ] = 0.0;
         else                                    // interior nodes -> approximation of the second derivative
             fu[ i ] = h_sqr_inv * (  u[ i - 1 ] - 2.0 * u[ i ] + u[ i + 1 ] );
          };
      auto time_stepping = [=] ( const Real& t, const Real& tau, const VectorView& u, VectorView& fu ) {
         TNL::Algorithms::ParallelFor< Device >::exec( 0, n, f, u, fu ); };
      solver.solve( u, time_stepping );
      write( file, u, n, h, solver.getTime() ); // write the current state to a file
   }
}

int main( int argc, char* argv[] )
{
   TNL::String file_name( argv[ 1 ] );
   file_name += "/ODESolver-HeatEquationExample-result.out";

   solveHeatEquation< TNL::Devices::Host >( file_name.getString() );
#ifdef HAVE_CUDA
   solveHeatEquation< TNL::Devices::Cuda >( file_name.getString() );
#endif
}
