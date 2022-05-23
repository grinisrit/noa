#include <iostream>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Solvers/ODE/StaticEuler.h>

using Real = double;

int main( int argc, char* argv[] )
{
   using Vector = TNL::Containers::StaticVector< 3, Real >;
   using ODESolver = TNL::Solvers::ODE::StaticEuler< Vector >;
   const Real final_t = 25.0;
   const Real tau = 0.001;
   const Real output_time_step = 0.01;
   const Real sigma = 10.0;
   const Real rho = 28.0;
   const Real beta = 8.0 / 3.0;

   ODESolver solver;
   solver.setTau(  tau );
   solver.setTime( 0.0 );
   Vector u( 1.0, 2.0, 3.0 );
   while( solver.getTime() < final_t )
   {
      solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
      auto f = [=] ( const Real& t, const Real& tau, const Vector& u, Vector& fu ) {
         const Real& x = u[ 0 ];
         const Real& y = u[ 1 ];
         const Real& z = u[ 2 ];
         fu[ 0 ] = sigma * (y - x );
         fu[ 1 ] = rho * x - y - x * z;
         fu[ 2 ] = -beta * z + x * y;
      };
      solver.solve( u, f );
      std::cout << u[ 0 ] << " " << u[ 1 ] << " " << u[ 2 ] << std::endl;
   }
}
