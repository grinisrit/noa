#include "HamiltonJacobiProblemConfig.h"
#include "HamiltonJacobiProblemSetter.h"
#include <solvers/tnlSolver.h>
#include "MainBuildConfig.h"
#include <solvers/tnlBuildConfigTags.h>

typedef MainBuildConfig BuildConfig;

int main( int argc, char* argv[] )
{
   tnlSolver< HamiltonJacobiProblemSetter, HamiltonJacobiProblemConfig, BuildConfig > solver;
   if( ! solver. run( argc, argv ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}


