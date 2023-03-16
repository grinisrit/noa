// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include "Hypre.h"

namespace noa::TNL {
namespace Solvers {
namespace Linear {

HypreSolver::HypreSolver( const Matrices::HypreParCSRMatrix& A )
{
   this->A = &A;
}

void
HypreSolver::setup( const Containers::HypreParVector& b, Containers::HypreParVector& x ) const
{
   if( A == nullptr )
      throw std::runtime_error( "HypreSolver::setup(...) : HypreParCSRMatrix A is missing" );

   if( setup_called )
      return;

   const HYPRE_Int err_flag = setupFcn()( *this, *A, b, x );
   if( err_flag != 0 ) {
      if( error_mode == WARN_HYPRE_ERRORS )
         std::cout << "Error during setup! Error code: " << err_flag << std::endl;
      else if( error_mode == ABORT_HYPRE_ERRORS )
         throw std::runtime_error( "Error during setup! Error code: " + std::to_string( err_flag ) );
   }
   hypre_error_flag = 0;

   setup_called = true;
}

void
HypreSolver::solve( const Containers::HypreParVector& b, Containers::HypreParVector& x ) const
{
   if( A == nullptr )
      throw std::runtime_error( "HypreSolver::solve(...) : HypreParCSRMatrix A is missing" );

   if( ! setup_called )
      setup( b, x );

   const HYPRE_Int err_flag = solveFcn()( *this, *A, b, x );
   if( err_flag != 0 ) {
      if( error_mode == WARN_HYPRE_ERRORS )
         std::cout << "Error during solve! Error code: " << err_flag << std::endl;
      else if( error_mode == ABORT_HYPRE_ERRORS )
         throw std::runtime_error( "Error during solve! Error code: " + std::to_string( err_flag ) );
   }
   hypre_error_flag = 0;

   postSolveHook();
}

HyprePCG::HyprePCG( MPI_Comm comm )
{
   HYPRE_ParCSRPCGCreate( comm, &solver );
}

HyprePCG::HyprePCG( const Matrices::HypreParCSRMatrix& A ) : HypreSolver( A )
{
   const MPI_Comm comm = A.getCommunicator();
   HYPRE_ParCSRPCGCreate( comm, &solver );
}

HyprePCG::~HyprePCG()
{
   HYPRE_ParCSRPCGDestroy( solver );
}

void
HyprePCG::setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup )
{
   HypreSolver::setMatrix( op, reuse_setup );
   if( precond != nullptr )
      precond->setMatrix( *A, reuse_setup );
}

void
HyprePCG::setPreconditioner( HypreSolver& precond_ )
{
   precond = &precond_;
   HYPRE_ParCSRPCGSetPrecond( solver, precond_.solveFcn(), precond_.setupFcn(), precond_ );
}

void
HyprePCG::setResidualConvergenceOptions( int res_frequency, double rtol )
{
   HYPRE_PCGSetTwoNorm( solver, 1 );
   if( res_frequency > 0 )
      HYPRE_PCGSetRecomputeResidualP( solver, res_frequency );
   if( rtol > 0.0 )
      HYPRE_PCGSetResidualTol( solver, rtol );
}

void
HyprePCG::postSolveHook() const
{
   HYPRE_Int print_level;
   HYPRE_PCGGetPrintLevel( solver, &print_level );

   if( print_level > 0 ) {
      HYPRE_Int num_iterations;
      HYPRE_PCGGetNumIterations( solver, &num_iterations );

      double final_res_norm;
      HYPRE_PCGGetFinalRelativeResidualNorm( solver, &final_res_norm );

      const MPI_Comm comm = A->getCommunicator();
      if( MPI::GetRank( comm ) == 0 ) {
         std::cout << "PCG Iterations = " << num_iterations << std::endl
                   << "Final PCG Relative Residual Norm = " << final_res_norm << std::endl;
      }
   }
}

HypreBiCGSTAB::HypreBiCGSTAB( MPI_Comm comm )
{
   HYPRE_ParCSRBiCGSTABCreate( comm, &solver );
}

HypreBiCGSTAB::HypreBiCGSTAB( const Matrices::HypreParCSRMatrix& A_ ) : HypreSolver( A_ )
{
   const MPI_Comm comm = A->getCommunicator();
   HYPRE_ParCSRBiCGSTABCreate( comm, &solver );
}

HypreBiCGSTAB::~HypreBiCGSTAB()
{
   HYPRE_ParCSRBiCGSTABDestroy( solver );
}

void
HypreBiCGSTAB::setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup )
{
   HypreSolver::setMatrix( op, reuse_setup );
   if( precond != nullptr )
      precond->setMatrix( *A, reuse_setup );
}

void
HypreBiCGSTAB::setPreconditioner( HypreSolver& precond_ )
{
   precond = &precond_;

   HYPRE_ParCSRBiCGSTABSetPrecond( solver, precond_.solveFcn(), precond_.setupFcn(), precond_ );
}

void
HypreBiCGSTAB::postSolveHook() const
{
   if( print_level > 0 ) {
      HYPRE_Int num_iterations;
      HYPRE_BiCGSTABGetNumIterations( solver, &num_iterations );

      double final_res_norm;
      HYPRE_BiCGSTABGetFinalRelativeResidualNorm( solver, &final_res_norm );

      const MPI_Comm comm = A->getCommunicator();
      if( MPI::GetRank( comm ) == 0 ) {
         std::cout << "BiCGSTAB Iterations = " << num_iterations << std::endl
                   << "Final BiCGSTAB Relative Residual Norm = " << final_res_norm << std::endl;
      }
   }
}

HypreGMRES::HypreGMRES( MPI_Comm comm )
{
   HYPRE_ParCSRGMRESCreate( comm, &solver );
}

HypreGMRES::HypreGMRES( const Matrices::HypreParCSRMatrix& A_ ) : HypreSolver( A_ )
{
   const MPI_Comm comm = A->getCommunicator();
   HYPRE_ParCSRGMRESCreate( comm, &solver );
}

HypreGMRES::~HypreGMRES()
{
   HYPRE_ParCSRGMRESDestroy( solver );
}

void
HypreGMRES::setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup )
{
   HypreSolver::setMatrix( op, reuse_setup );
   if( precond != nullptr )
      precond->setMatrix( *A, reuse_setup );
}

void
HypreGMRES::setPreconditioner( HypreSolver& precond_ )
{
   precond = &precond_;
   HYPRE_ParCSRGMRESSetPrecond( solver, precond_.solveFcn(), precond_.setupFcn(), precond_ );
}

void
HypreGMRES::postSolveHook() const
{
   HYPRE_Int print_level;
   HYPRE_GMRESGetPrintLevel( solver, &print_level );

   if( print_level > 0 ) {
      HYPRE_Int num_iterations;
      HYPRE_GMRESGetNumIterations( solver, &num_iterations );

      double final_res_norm;
      HYPRE_GMRESGetFinalRelativeResidualNorm( solver, &final_res_norm );

      const MPI_Comm comm = A->getCommunicator();
      if( MPI::GetRank( comm ) == 0 ) {
         std::cout << "GMRES Iterations = " << num_iterations << std::endl
                   << "Final GMRES Relative Residual Norm = " << final_res_norm << std::endl;
      }
   }
}

HypreFlexGMRES::HypreFlexGMRES( MPI_Comm comm )
{
   HYPRE_ParCSRFlexGMRESCreate( comm, &solver );
}

HypreFlexGMRES::HypreFlexGMRES( const Matrices::HypreParCSRMatrix& A_ ) : HypreSolver( A_ )
{
   const MPI_Comm comm = A->getCommunicator();
   HYPRE_ParCSRFlexGMRESCreate( comm, &solver );
}

HypreFlexGMRES::~HypreFlexGMRES()
{
   HYPRE_ParCSRFlexGMRESDestroy( solver );
}

void
HypreFlexGMRES::setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup )
{
   HypreSolver::setMatrix( op, reuse_setup );
   if( precond != nullptr )
      precond->setMatrix( *A, reuse_setup );
}

void
HypreFlexGMRES::setPreconditioner( HypreSolver& precond_ )
{
   precond = &precond_;
   HYPRE_ParCSRFlexGMRESSetPrecond( solver, precond_.solveFcn(), precond_.setupFcn(), precond_ );
}

void
HypreFlexGMRES::postSolveHook() const
{
   HYPRE_Int print_level;
   HYPRE_FlexGMRESGetPrintLevel( solver, &print_level );

   if( print_level > 0 ) {
      HYPRE_Int num_iterations;
      HYPRE_FlexGMRESGetNumIterations( solver, &num_iterations );

      double final_res_norm;
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm( solver, &final_res_norm );

      const MPI_Comm comm = A->getCommunicator();
      if( MPI::GetRank( comm ) == 0 ) {
         std::cout << "FlexGMRES Iterations = " << num_iterations << std::endl
                   << "Final FlexGMRES Relative Residual Norm = " << final_res_norm << std::endl;
      }
   }
}

HypreParaSails::HypreParaSails( MPI_Comm comm )
{
   HYPRE_ParaSailsCreate( comm, &solver );
}

HypreParaSails::HypreParaSails( const Matrices::HypreParCSRMatrix& A ) : HypreSolver( A )
{
   HYPRE_ParaSailsCreate( A.getCommunicator(), &solver );
}

HypreParaSails::~HypreParaSails()
{
   HYPRE_ParaSailsDestroy( solver );
}

HypreEuclid::HypreEuclid( MPI_Comm comm )
{
   HYPRE_EuclidCreate( comm, &solver );
}

HypreEuclid::HypreEuclid( const Matrices::HypreParCSRMatrix& A ) : HypreSolver( A )
{
   HYPRE_EuclidCreate( A.getCommunicator(), &solver );
}

HypreEuclid::~HypreEuclid()
{
   HYPRE_EuclidDestroy( solver );
}

HypreILU::HypreILU()
{
   HYPRE_ILUCreate( &solver );
   setDefaultOptions();
}

HypreILU::HypreILU( const Matrices::HypreParCSRMatrix& A ) : HypreSolver( A )
{
   HYPRE_ILUCreate( &solver );
   setDefaultOptions();
}

HypreILU::~HypreILU()
{
   HYPRE_ILUDestroy( solver );
}

void
HypreILU::setDefaultOptions()
{
   // Maximum iterations; 1 iter for preconditioning
   HYPRE_ILUSetMaxIter( solver, 1 );

   // The tolerance when used as a smoother; set to 0.0 for preconditioner
   HYPRE_ILUSetTol( solver, 0.0 );

   // The type of incomplete LU used locally and globally (see class doc)
   HYPRE_ILUSetType( solver, 0 );  // 0: ILU(k) locally and block Jacobi globally

   // Fill level 'k' for ILU(k)
   HYPRE_ILUSetLevelOfFill( solver, 0 );

   // Local reordering scheme; 0 = no reordering, 1 = reverse Cuthill-McKee
   HYPRE_ILUSetLocalReordering( solver, 1 );
}

HypreBoomerAMG::HypreBoomerAMG()
{
   HYPRE_BoomerAMGCreate( &solver );
   setDefaultOptions();
}

HypreBoomerAMG::HypreBoomerAMG( const Matrices::HypreParCSRMatrix& A ) : HypreSolver( A )
{
   HYPRE_BoomerAMGCreate( &solver );
   setDefaultOptions();
}

HypreBoomerAMG::~HypreBoomerAMG()
{
   HYPRE_BoomerAMGDestroy( solver );
}

void
HypreBoomerAMG::setDefaultOptions()
{
   #if ! defined( HYPRE_USING_GPU )
   // AMG coarsening options:
   const int coarsen_type = 10;  // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP

   // AMG relaxation options:
   const int relax_type = 8;  // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
   #else
   // AMG coarsening options:
   const int coarsen_type = 8;  // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP

   // AMG relaxation options:
   const int relax_type = 18;  // 18 = l1-Jacobi, or 16 = Chebyshev
   #endif

   // AMG coarsening options:
   const int agg_levels = 1;   // number of aggressive coarsening levels
   const double theta = 0.25;  // strength threshold: 0.25, 0.5, 0.8

   // AMG relaxation options:
   const int relax_sweeps = 1;  // relaxation sweeps on each level

   // AMG interpolation options:
   const int interp_type = 6;  // 6 = extended+i, or 18 = extended+e
   const int Pmax = 4;         // max number of elements per row in P

   // Additional options:
   const int print_level = 0;  // 0 = none, 1 = setup, 2 = solve, 3 = setup+solve
   const int max_levels = 25;  // max number of levels in AMG hierarchy

   HYPRE_BoomerAMGSetCoarsenType( solver, coarsen_type );
   HYPRE_BoomerAMGSetAggNumLevels( solver, agg_levels );
   HYPRE_BoomerAMGSetRelaxType( solver, relax_type );
   HYPRE_BoomerAMGSetNumSweeps( solver, relax_sweeps );
   HYPRE_BoomerAMGSetStrongThreshold( solver, theta );
   HYPRE_BoomerAMGSetInterpType( solver, interp_type );
   HYPRE_BoomerAMGSetPMaxElmts( solver, Pmax );
   HYPRE_BoomerAMGSetPrintLevel( solver, print_level );
   HYPRE_BoomerAMGSetMaxLevels( solver, max_levels );

   // Additional options related to GPU performance, see https://github.com/hypre-space/hypre/wiki/GPUs
   #if defined( HYPRE_USING_GPU )
   // interpolation used on levels of aggressive coarsening (use 5 or 7)
   HYPRE_BoomerAMGSetAggInterpType( solver, 7 );
   // keep local interpolation transposes to avoid SpMTV
   HYPRE_BoomerAMGSetKeepTranspose( solver, 1 );
   // RAP in two matrix products instead of three (default: 0)
   HYPRE_BoomerAMGSetRAP2( solver, 1 );
   #endif

   // Use as a preconditioner (one V-cycle, zero tolerance)
   HYPRE_BoomerAMGSetMaxIter( solver, 1 );
   HYPRE_BoomerAMGSetTol( solver, 0.0 );
}

void
HypreBoomerAMG::setSystemsOptions( int dim, bool order_bynodes )
{
   HYPRE_BoomerAMGSetNumFunctions( solver, dim );

   // The default "system" ordering in Hypre is byVDIM. When using byNODES
   // ordering, we have to specify the ordering explicitly.
   if( order_bynodes ) {
      TNL_ASSERT_TRUE( A->getRows() % dim == 0, "Ordering does not work as claimed!" );
      const HYPRE_Int nnodes = A->getRows() / dim;

      // generate DofFunc mapping on the host
      HYPRE_Int* h_mapping = hypre_CTAlloc( HYPRE_Int, A->getRows(), HYPRE_MEMORY_HOST );
      HYPRE_Int k = 0;
      for( int i = 0; i < dim; i++ )
         for( HYPRE_Int j = 0; j < nnodes; j++ )
            h_mapping[ k++ ] = i;

   #if defined( HYPRE_USING_GPU )
      // the mapping is assumed to be a device pointer
      HYPRE_Int* mapping = hypre_CTAlloc( HYPRE_Int, A->getRows(), HYPRE_MEMORY_DEVICE );
      hypre_TMemcpy( mapping, h_mapping, HYPRE_Int, A->getRows(), HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST );
      hypre_TFree( h_mapping, HYPRE_MEMORY_HOST );
   #else
      HYPRE_Int* mapping = h_mapping;
   #endif

      // Hypre takes ownership and frees the pointer in HYPRE_BoomerAMGDestroy
      HYPRE_BoomerAMGSetDofFunc( solver, mapping );
   }

   // more robust options with respect to convergence
   HYPRE_BoomerAMGSetAggNumLevels( solver, 0 );
   HYPRE_BoomerAMGSetStrongThreshold( solver, 0.5 );
}

void
HypreBoomerAMG::setAdvectiveOptions( int restrict_type, int relax_order )
{
   HYPRE_BoomerAMGSetRestriction( solver, restrict_type );
   HYPRE_BoomerAMGSetRelaxOrder( solver, relax_order );

   const int interp_type = 100;
   const int relax_type = 8;
   const int coarsen_type = 6;
   const double strength_tolC = 0.1;
   const double strength_tolR = 0.01;
   const double filter_tolR = 0.0;
   const double filterA_tol = 0.0;

   HYPRE_BoomerAMGSetInterpType( solver, interp_type );
   HYPRE_BoomerAMGSetRelaxType( solver, relax_type );
   HYPRE_BoomerAMGSetCoarsenType( solver, coarsen_type );
   HYPRE_BoomerAMGSetStrongThreshold( solver, strength_tolC );
   HYPRE_BoomerAMGSetStrongThresholdR( solver, strength_tolR );
   HYPRE_BoomerAMGSetFilterThresholdR( solver, filter_tolR );
   HYPRE_BoomerAMGSetADropTol( solver, filterA_tol );
   // type = -1: drop based on row inf-norm
   HYPRE_BoomerAMGSetADropType( solver, -1 );

   // disable aggressive coarsening
   HYPRE_BoomerAMGSetAggNumLevels( solver, 0 );

   // set number of sweeps (up and down cycles, and the coarsest level)
   HYPRE_BoomerAMGSetNumSweeps( solver, 1 );
}

void
HypreBoomerAMG::postSolveHook() const
{
   HYPRE_Int print_level;
   HYPRE_BoomerAMGGetPrintLevel( solver, &print_level );

   if( print_level > 1 ) {
      HYPRE_Int num_iterations;
      HYPRE_BoomerAMGGetNumIterations( solver, &num_iterations );

      double final_res_norm;
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm( solver, &final_res_norm );

      const MPI_Comm comm = A->getCommunicator();
      if( MPI::GetRank( comm ) == 0 ) {
         std::cout << "BoomerAMG Iterations = " << num_iterations << std::endl
                   << "Final BoomerAMG Relative Residual Norm = " << final_res_norm << std::endl;
      }
   }
}

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL

#endif  // HAVE_HYPRE
