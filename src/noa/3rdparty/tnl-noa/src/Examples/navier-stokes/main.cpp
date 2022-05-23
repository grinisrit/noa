#include <cstdlib>
#include "navier-stokes-conf.h"
#include "navierStokesSetter.h"
#include <TNL/Solvers/Solver.h>

int main( int argc, char* argv[] )
{
   Solver< navierStokesSetter > solver;
   if( ! solver. run( CONFIG_FILE, argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


