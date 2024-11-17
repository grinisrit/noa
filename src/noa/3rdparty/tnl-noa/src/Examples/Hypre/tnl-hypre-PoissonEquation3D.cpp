/**
 * Description:
 *
 * This example solves the 2-D Laplacian problem with zero boundary conditions
 * on an n x n grid.  The number of unknowns is N=n^2.  The standard 5-point
 * stencil is used, and we solve for the interior nodes only.
 *
 * The example is based on the "ex5.c" example.  It demonstrates how the C code
 * can be ported to C++ and TNL.  We recommend comparing this example the
 * original one.
 *
 * Compile with: use the install script from TNL
 *
 * Sample run:   OMP_NUM_THREADS=1 mpirun -np 4 tnl-hypre-ex5
 */

#include <cstdio>
#include <string>

#include <TNL/Math.h>
#include <TNL/MPI.h>
#include <TNL/Hypre.h>
#include <TNL/Containers/HypreParVector.h>
#include <TNL/Matrices/HypreParCSRMatrix.h>
#include <TNL/Solvers/Linear/Hypre.h>

#include "vis.c"

int
hypre_FlexGMRESModifyPCAMGExample( void* precond_data, int iterations, double rel_residual_norm );

int
main( int argc, char* argv[] )
{
   // Initialize MPI
   TNL::MPI::ScopedInitializer mpi( argc, argv );
   const int myid = TNL::MPI::GetRank( MPI_COMM_WORLD );
   const int num_procs = TNL::MPI::GetSize( MPI_COMM_WORLD );

   // Initialize HYPRE and set some global options, notably HYPRE_SetSpGemmUseCusparse(0);
   TNL::Hypre hypre;

   // Default problem parameters
   int n = 33;
   int solver_id = 0;
   int vis = 0;

   // Parse the command line
   {
      int arg_index = 0;
      int print_usage = 0;

      while( arg_index < argc ) {
         if( std::string( argv[ arg_index ] ) == "-n" ) {
            arg_index++;
            n = std::stoi( argv[ arg_index++ ] );
         }
         else if( std::string( argv[ arg_index ] ) == "-solver" ) {
            arg_index++;
            solver_id = std::stoi( argv[ arg_index++ ] );
         }
         else if( std::string( argv[ arg_index ] ) == "-vis" ) {
            arg_index++;
            vis = 1;
         }
         else if( std::string( argv[ arg_index ] ) == "-help" ) {
            print_usage = 1;
            break;
         }
         else {
            arg_index++;
         }
      }

      if( print_usage && myid == 0 ) {
         std::cerr << "\n"
                      "Usage: " << argv[ 0 ] << " [<options>]\n"
                      "\n"
                      "  -n <n>              : problem size in each direction (default: 33)\n"
                      "  -solver <ID>        : solver ID\n"
                      "                        0  - AMG (default) \n"
                      "                        1  - AMG-PCG\n"
                      "                        8  - ParaSails-PCG\n"
                      "                        50 - PCG\n"
                      "                        61 - AMG-FlexGMRES\n"
                      "  -vis                : save the solution for GLVis visualization\n"
                   << std::endl;
      }

      if( print_usage )
         return EXIT_SUCCESS;
   }

   // Set the problem sizes
   // Preliminaries: we want at least one rank per row
   if( n * n * n < num_procs )
      n = std::cbrt( num_procs ) + 1;
   const int N = n * n * n;            // global number of rows
   const double h = 1.0 / ( n + 1 );   // mesh cell size

   /* Each rank knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of ranks. */
   HYPRE_Int local_size = N / num_procs;
   HYPRE_Int extra = N - local_size * num_procs;

   HYPRE_Int ilower = local_size * myid;
   ilower += TNL::min( myid, extra );

   HYPRE_Int iupper = local_size * ( myid + 1 );
   iupper += TNL::min( myid + 1, extra );
   iupper = iupper - 1;

   // How many rows do I have?
   local_size = iupper - ilower + 1;

   // Let each rank create its local matrix in the CSR format in TNL
   using CSR = TNL::Matrices::SparseMatrix< double, TNL::HYPRE_Device, HYPRE_Int >;
   CSR A_local;
   A_local.setDimensions( local_size, N );

   // Allocate row capacities - this must match exactly the sparsity pattern of
   // the matrix
   typename CSR::RowsCapacitiesType capacities;
   capacities.setSize( local_size );
   auto capacities_view = capacities.getView();
   TNL::Algorithms::ParallelFor< TNL::HYPRE_Device >::exec( ilower, iupper + 1,
      [=] __cuda_callable__ ( HYPRE_Int i ) mutable
      {
         int nnz = 0;

         // The z-1 neighbor on the grid: position i-n*n
         if( i - n * n >= 0 )
            nnz++;

         // The y-1 neighbor on the grid: position i-n
         if( i - n >= 0 )
            nnz++;

         // The x-1 neighbor on the grid: position i-1
         if( i % n )
            nnz++;

         // The diagonal: position i
         nnz++;

         // The x+1 neighbor on the grid: position i+1
         if( ( i + 1 ) % n )
            nnz++;

         // The y+1 neighbor on the grid: position i+n
         if( i + n < N )
            nnz++;

         // The z+1 neighbor on the grid: position i+n*n
         if( i + n * n < N )
            nnz++;

         // The row index must be converted from global to local
         capacities_view[ i - ilower ] = nnz;
      } );
   A_local.setRowCapacities( capacities );

   // Now assemble the local matrix. Each row has at most 7 entries.
   A_local.forAllRows( [=] __cuda_callable__ ( typename CSR::RowView& row ) mutable {
         // The row index must be converted from local to global
         const HYPRE_Int i = ilower + row.getRowIndex();
         int nnz = 0;

         // The z-1 neighbor on the grid: position i-n*n
         if( i - n * n >= 0 )
            row.setElement( nnz++, i - n * n, -1.0 );

         // The y-1 neighbor on the grid: position i-n
         if( i - n >= 0 )
            row.setElement( nnz++, i - n, -1.0 );

         // The x-1 neighbor on the grid: position i-1
         if( i % n )
            row.setElement( nnz++, i - 1, -1.0 );

         // The diagonal: position i
         row.setElement( nnz++, i, 6.0 );

         // The x+1 neighbor on the grid: position i+1
         if( ( i + 1 ) % n )
            row.setElement( nnz++, i + 1, -1.0 );

         // The y+1 neighbor on the grid: position i+n
         if( i + n < N )
            row.setElement( nnz++, i + n, -1.0 );

         // The z+1 neighbor on the grid: position i+n*n
         if( i + n * n < N )
            row.setElement( nnz++, i + n * n, -1.0 );
      } );

   // Bind the TNL matrix to HypreCSR
   TNL::Matrices::HypreCSRMatrix A_local_hypre;
   A_local_hypre.bind( A_local );

   // Assemble the distributed matrix in Hypre from the local blocks
   // Note that this is a square matrix, so we indicate the row partition
   // size twice (since number of rows = number of cols)
   using HypreParCSR = TNL::Matrices::HypreParCSRMatrix;
   HypreParCSR parcsr_A = HypreParCSR::fromLocalBlocks( MPI_COMM_WORLD, N, N, {ilower, iupper + 1}, {ilower, iupper + 1}, A_local_hypre );

   // Deallocate the local matrix since it is not needed anymore in this example
   A_local_hypre.reset();
   A_local.reset();


   // Create the rhs and solution vectors
   TNL::Containers::HypreParVector par_b;
   TNL::Containers::HypreParVector par_x;

   par_b.setDistribution( {ilower, iupper + 1}, 0, N, MPI_COMM_WORLD );
   par_x.setDistribution( {ilower, iupper + 1}, 0, N, MPI_COMM_WORLD );

   // Set the rhs values to h^2 and the solution to zero
   par_b.setValue( h * h );
   par_x.setValue( 0.0 );


   // Choose a solver and solve the system

   // AMG
   if( solver_id == 0 ) {
      // Create the solver
      TNL::Solvers::Linear::HypreBoomerAMG solver;

      // Set the matrix of the linear system
      solver.setMatrix( parcsr_A );

      // Set some parameters (See Reference Manual for more parameters)
      // NOTE: The wrapper class sets its own default options that are
      //       different from Hypre. The overriding settings below result in
      //       the same state as the hypre-ex5.c example.
      HYPRE_BoomerAMGSetPrintLevel( solver, 3 );    // Print solve info + parameters
      HYPRE_BoomerAMGSetOldDefault( solver );       // Falgout coarsening with modified classical interpolation
      HYPRE_BoomerAMGSetRelaxType( solver, 6 );     // Sym. G-S/Jacobi hybrid relaxation
      HYPRE_BoomerAMGSetRelaxOrder( solver, 1 );    // Uses C/F relaxation
      HYPRE_BoomerAMGSetNumSweeps( solver, 1 );     // Sweeeps on each level
      HYPRE_BoomerAMGSetTol( solver, 1e-7 );        // Convergence tolerance
      HYPRE_BoomerAMGSetMaxIter( solver, 20 );      // Use as solver: max iterations
      HYPRE_BoomerAMGSetAggNumLevels( solver, 0 );  // number of aggressive coarsening levels

      // Solve the linear system (calls AMG setup, solve, and prints final residual norm)
      solver.solve( par_b, par_x );
   }
   // PCG
   else if( solver_id == 50 ) {
      // Create the solver
      TNL::Solvers::Linear::HyprePCG solver( MPI_COMM_WORLD );

      // Set some parameters (See Reference Manual for more parameters)
      HYPRE_PCGSetMaxIter( solver, 1000 );  // max iterations
      HYPRE_PCGSetTol( solver, 1e-7 );      // convergence tolerance
      HYPRE_PCGSetTwoNorm( solver, 1 );     // use the two norm as the stopping criteria
      HYPRE_PCGSetPrintLevel( solver, 2 );  // prints out the iteration info

      // Set the matrix of the linear system
      solver.setMatrix( parcsr_A );

      // Ignore errors returned from Hypre functions (e.g. when PCG does not
      // converge within the maximum iterations limit).
      solver.setErrorMode( solver.WARN_HYPRE_ERRORS );

      // Solve the linear system (calls PCG setup, solve, and prints final residual norm)
      solver.solve( par_b, par_x );
   }
   // PCG with AMG preconditioner
   else if( solver_id == 1 ) {
      // Create the solver
      TNL::Solvers::Linear::HyprePCG solver( MPI_COMM_WORLD );

      // Create the preconditioner
      TNL::Solvers::Linear::HypreBoomerAMG precond;

      // Set the PCG preconditioner
      solver.setPreconditioner( precond );

      // Set the matrix of the linear system
      solver.setMatrix( parcsr_A );

      // Set some parameters (See Reference Manual for more parameters)
      HYPRE_BoomerAMGSetPrintLevel( precond, 1 );    // Print setup info + parameters
      HYPRE_BoomerAMGSetOldDefault( precond );       // Falgout coarsening with modified classical interpolation
      HYPRE_BoomerAMGSetRelaxType( precond, 6 );     // Sym G.S./Jacobi hybrid relaxation
      HYPRE_BoomerAMGSetRelaxOrder( precond, 1 );    // Uses C/F relaxation
      HYPRE_BoomerAMGSetAggNumLevels( precond, 0 );  // number of aggressive coarsening levels

      // Set some parameters (See Reference Manual for more parameters)
      HYPRE_PCGSetMaxIter( solver, 1000 );  // max iterations
      HYPRE_PCGSetTol( solver, 1e-7 );      // convergence tolerance
      HYPRE_PCGSetTwoNorm( solver, 1 );     // use the two norm as the stopping criteria
      HYPRE_PCGSetPrintLevel( solver, 2 );  // prints out the iteration info

      // Solve the linear system (calls PCG setup, solve, and prints final residual norm)
      solver.solve( par_b, par_x );
   }
   // PCG with ParaSails preconditioner
   else if( solver_id == 8 ) {
      // Create the solver
      TNL::Solvers::Linear::HyprePCG solver( MPI_COMM_WORLD );

      // Create the preconditioner
      TNL::Solvers::Linear::HypreParaSails precond( MPI_COMM_WORLD );

      // Set the PCG preconditioner
      solver.setPreconditioner( precond );

      // Set the matrix of the linear system
      solver.setMatrix( parcsr_A );

      // Set some parameters (See Reference Manual for more parameters)
      HYPRE_PCGSetMaxIter( solver, 1000 );  // max iterations
      HYPRE_PCGSetTol( solver, 1e-7 );      // convergence tolerance
      HYPRE_PCGSetTwoNorm( solver, 1 );     // use the two norm as the stopping criteria
      HYPRE_PCGSetPrintLevel( solver, 2 );  // prints out the iteration info

      // Set some parameters (See Reference Manual for more parameters)
      HYPRE_ParaSailsSetParams( precond, 0.1, 1 );  // threshold and max levels
      HYPRE_ParaSailsSetFilter( precond, 0.05 );
      HYPRE_ParaSailsSetSym( precond, 1 );

      // Solve the linear system (calls PCG setup, solve, and prints final residual norm)
      solver.solve( par_b, par_x );
   }
   // Flexible GMRES with AMG preconditioner
   else if( solver_id == 61 ) {
      // Create the solver
      TNL::Solvers::Linear::HypreFlexGMRES solver( MPI_COMM_WORLD );

      // Create the preconditioner
      TNL::Solvers::Linear::HypreBoomerAMG precond;

      // Set the FlexGMRES preconditioner
      solver.setPreconditioner( precond );

      // Set the matrix of the linear system
      solver.setMatrix( parcsr_A );

      // Set some parameters (See Reference Manual for more parameters)
      HYPRE_FlexGMRESSetKDim( solver, 30 );       // restart parameter
      HYPRE_FlexGMRESSetMaxIter( solver, 1000 );  // max iterations
      HYPRE_FlexGMRESSetTol( solver, 1e-7 );      // convergence tolerance
      HYPRE_FlexGMRESSetPrintLevel( solver, 2 );  // print solve info

      // Set some parameters (See Reference Manual for more parameters)
      HYPRE_BoomerAMGSetPrintLevel( precond, 1 );    // Print setup info + parameters
      HYPRE_BoomerAMGSetOldDefault( precond );       // Falgout coarsening with modified classical interpolation
      HYPRE_BoomerAMGSetRelaxType( precond, 6 );     // Sym G.S./Jacobi hybrid relaxation
      HYPRE_BoomerAMGSetRelaxOrder( precond, 1 );    // Uses C/F relaxation
      HYPRE_BoomerAMGSetAggNumLevels( precond, 0 );  // number of aggressive coarsening levels

      // This is an optional call - if you don't call it,
      // hypre_FlexGMRESModifyPCDefault is used - which does nothing.
      // Otherwise, you can define your own, similar to the one used here
      HYPRE_FlexGMRESSetModifyPC( solver, (HYPRE_PtrToModifyPCFcn) hypre_FlexGMRESModifyPCAMGExample );

      // Solve the linear system (calls FlexGMRES setup, solve, and prints final residual norm)
      solver.solve( par_b, par_x );
   }
   else if( myid == 0 ) {
      std::cerr << "Invalid solver id specified." << std::endl;
      return EXIT_FAILURE;
   }

   // Save the solution for GLVis visualization, see glvis-ex5.sh
   if( vis ) {
      char filename[ 255 ];
      sprintf( filename, "%s.%06d", "vis_tnl/ex5.sol", myid );
      FILE* file = fopen( filename, "w" );
      if( file == nullptr ) {
         printf( "Error: can't open output file %s\n", filename );
         return EXIT_FAILURE;
      }

      // Save the solution
      const auto local_x = par_x.getConstLocalView();
      for( HYPRE_Int i = 0; i < local_size; i++ )
         fprintf( file, "%.14e\n", local_x[ i ] );

      fflush( file );
      fclose( file );

      // Save the global finite element mesh
      if( myid == 0 )
         GLVis_PrintGlobalSquareMesh( "vis_tnl/ex5.mesh", n - 1 );
   }

   return EXIT_SUCCESS;
}

/**
 * This is an example (not recommended) of how we can modify things about AMG
 * that affect the solve phase based on how FlexGMRES is doing... For another
 * preconditioner it may make sense to modify the tolerance.
 */
int
hypre_FlexGMRESModifyPCAMGExample( void* precond_data, int iterations, double rel_residual_norm )
{
   if( rel_residual_norm > .1 )
      HYPRE_BoomerAMGSetNumSweeps( (HYPRE_Solver) precond_data, 10 );
   else
      HYPRE_BoomerAMGSetNumSweeps( (HYPRE_Solver) precond_data, 1 );
   return 0;
}
