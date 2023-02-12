// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include <noa/3rdparty/tnl-noa/src/TNL/Containers/HypreParVector.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Matrices/HypreParCSRMatrix.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

//! \brief Abstract class for Hypre's solvers and preconditioners
//! \ingroup Hypre
class HypreSolver
{
public:
   //! \brief How to treat errors returned by Hypre function calls
   enum ErrorMode
   {
      IGNORE_HYPRE_ERRORS,  ///< Ignore Hypre errors
      WARN_HYPRE_ERRORS,    ///< Issue warnings on Hypre errors
      ABORT_HYPRE_ERRORS    ///< Abort on Hypre errors (default in base class)
   };

protected:
   //! Handle for the Hypre solver
   HYPRE_Solver solver = nullptr;

   //! The linear system matrix
   const Matrices::HypreParCSRMatrix* A = nullptr;

   //! Indicates if Hypre's setup function was already called
   mutable bool setup_called = false;

   //! How to treat Hypre errors
   mutable ErrorMode error_mode = ABORT_HYPRE_ERRORS;

   //! Hook function that is called at the end of \ref solve
   virtual void
   postSolveHook() const
   {}

public:
   HypreSolver() = default;

   explicit HypreSolver( const Matrices::HypreParCSRMatrix& A );

   //! Type-cast to \e HYPRE_Solver
   virtual operator HYPRE_Solver() const
   {
      return solver;
   }

   //! Hypre's internal setup function
   virtual HYPRE_PtrToParSolverFcn
   setupFcn() const = 0;

   //! Hypre's internal solve function
   virtual HYPRE_PtrToParSolverFcn
   solveFcn() const = 0;

   /**
    * \brief Set the matrix of the linear system to be solved.
    *
    * This function also resets the internal flag indicating whether the Hypre
    * setup function was called for the current matrix.
    *
    * \param op The input matrix.
    * \param reuse_setup When true, the result of the previous setup phase will
    *                    be preserved, i.e., the solver (and preconditioner)
    *                    will not be updated for the new matrix when calling
    *                    the \ref solve method.
    */
   virtual void
   setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup = false )
   {
      A = &op;
      if( setup_called && reuse_setup )
         setup_called = true;
      else
         setup_called = false;
   }

   /**
    * \brief Setup the solver for solving the linear system Ax=b.
    *
    * Calling this function repeatedly has no effect until the internal flag is
    * reset via the \ref setMatrix function.
    */
   virtual void
   setup( const Containers::HypreParVector& b, Containers::HypreParVector& x ) const;

   /**
    * \brief Solve the linear system Ax=b.
    *
    * This function checks if the \ref setup function was already called and
    * calls it if necessary.
    */
   virtual void
   solve( const Containers::HypreParVector& b, Containers::HypreParVector& x ) const;

   /**
    * \brief Set the behavior for treating Hypre errors, see the \ref ErrorMode
    * enum. The default mode in the base class is \ref ABORT_HYPRE_ERRORS.
    */
   void
   setErrorMode( ErrorMode err_mode ) const
   {
      error_mode = err_mode;
   }

   virtual ~HypreSolver() = default;
};

/**
 * \brief Wrapper for the PCG solver in Hypre
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HyprePCG solver;
 * HYPRE_PCGSetTol(solver, 1e-7);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HyprePCG : public HypreSolver
{
private:
   HypreSolver* precond = nullptr;

public:
   explicit HyprePCG( MPI_Comm comm );

   explicit HyprePCG( const Matrices::HypreParCSRMatrix& A );

   ~HyprePCG() override;

   void
   setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup = false ) override;

   //! Set the Hypre solver to be used as a preconditioner
   void
   setPreconditioner( HypreSolver& precond );

   void
   setTol( double tol )
   {
      HYPRE_PCGSetTol( solver, tol );
   }

   void
   setAbsTol( double atol )
   {
      HYPRE_PCGSetAbsoluteTol( solver, atol );
   }

   void
   setMaxIter( int max_iter )
   {
      HYPRE_PCGSetMaxIter( solver, max_iter );
   }

   void
   setPrintLevel( int print_level )
   {
      HYPRE_PCGSetPrintLevel( solver, print_level );
   }

   /**
    * \brief Use the L2 norm of the residual for measuring PCG convergence,
    * plus (optionally) 1) periodically recompute true residuals from scratch;
    * and 2) enable residual-based stopping criteria.
    */
   void
   setResidualConvergenceOptions( int res_frequency = -1, double rtol = 0.0 );

   HYPRE_Int
   getNumIterations() const
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRPCGGetNumIterations( solver, &num_it );
      return num_it;
   }

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRPCGSolve;
   }

protected:
   void
   postSolveHook() const override;
};

/**
 * \brief Wrapper for the BiCGSTAB solver in Hypre
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HypreBiCGSTAB solver;
 * HYPRE_BiCGSTABSetTol(solver, 1e-7);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HypreBiCGSTAB : public HypreSolver
{
private:
   HypreSolver* precond = nullptr;

   // Hypre does not provide a way to query this from the BiCGSTAB solver
   // https://github.com/hypre-space/hypre/issues/627
   HYPRE_Int print_level = 0;

public:
   explicit HypreBiCGSTAB( MPI_Comm comm );

   explicit HypreBiCGSTAB( const Matrices::HypreParCSRMatrix& A_ );

   ~HypreBiCGSTAB() override;

   void
   setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup = false ) override;

   //! Set the Hypre solver to be used as a preconditioner
   void
   setPreconditioner( HypreSolver& precond );

   void
   setTol( double tol )
   {
      HYPRE_BiCGSTABSetTol( solver, tol );
   }

   void
   setAbsTol( double atol )
   {
      HYPRE_BiCGSTABSetAbsoluteTol( solver, atol );
   }

   void
   setMaxIter( int max_iter )
   {
      HYPRE_BiCGSTABSetMaxIter( solver, max_iter );
   }

   void
   setPrintLevel( int print_level )
   {
      HYPRE_BiCGSTABSetPrintLevel( solver, print_level );
      this->print_level = print_level;
   }

   HYPRE_Int
   getNumIterations() const
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRBiCGSTABGetNumIterations( solver, &num_it );
      return num_it;
   }

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRBiCGSTABSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRBiCGSTABSolve;
   }

protected:
   void
   postSolveHook() const override;
};

/**
 * \brief Wrapper for the GMRES solver in Hypre
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HypreGMRES solver;
 * HYPRE_GMRESSetTol(solver, 1e-7);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HypreGMRES : public HypreSolver
{
private:
   HypreSolver* precond = nullptr;

public:
   explicit HypreGMRES( MPI_Comm comm );

   explicit HypreGMRES( const Matrices::HypreParCSRMatrix& A_ );

   ~HypreGMRES() override;

   void
   setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup = false ) override;

   //! Set the Hypre solver to be used as a preconditioner
   void
   setPreconditioner( HypreSolver& precond );

   void
   setTol( double tol )
   {
      HYPRE_GMRESSetTol( solver, tol );
   }

   void
   setAbsTol( double tol )
   {
      HYPRE_GMRESSetAbsoluteTol( solver, tol );
   }

   void
   setMaxIter( int max_iter )
   {
      HYPRE_GMRESSetMaxIter( solver, max_iter );
   }

   void
   setKDim( int k_dim )
   {
      HYPRE_GMRESSetKDim( solver, k_dim );
   }

   void
   setPrintLevel( int print_level )
   {
      HYPRE_GMRESSetPrintLevel( solver, print_level );
   }

   HYPRE_Int
   getNumIterations() const
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRGMRESGetNumIterations( solver, &num_it );
      return num_it;
   }

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRGMRESSolve;
   }

protected:
   void
   postSolveHook() const override;
};

/**
 * \brief Wrapper for the Flexible GMRES solver in Hypre
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HypreFlexGMRES solver;
 * HYPRE_FlexGMRESSetTol(solver, 1e-7);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HypreFlexGMRES : public HypreSolver
{
private:
   HypreSolver* precond = nullptr;

public:
   explicit HypreFlexGMRES( MPI_Comm comm );

   explicit HypreFlexGMRES( const Matrices::HypreParCSRMatrix& A_ );

   ~HypreFlexGMRES() override;

   void
   setMatrix( const Matrices::HypreParCSRMatrix& op, bool reuse_setup = false ) override;

   //! Set the Hypre solver to be used as a preconditioner
   void
   setPreconditioner( HypreSolver& precond );

   void
   setTol( double tol )
   {
      HYPRE_FlexGMRESSetTol( solver, tol );
   }

   void
   setMaxIter( int max_iter )
   {
      HYPRE_FlexGMRESSetMaxIter( solver, max_iter );
   }

   void
   setKDim( int k_dim )
   {
      HYPRE_FlexGMRESSetKDim( solver, k_dim );
   }

   void
   setPrintLevel( int print_level )
   {
      HYPRE_FlexGMRESSetPrintLevel( solver, print_level );
   }

   HYPRE_Int
   getNumIterations() const
   {
      HYPRE_Int num_it;
      HYPRE_ParCSRFlexGMRESGetNumIterations( solver, &num_it );
      return num_it;
   }

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRFlexGMRESSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRFlexGMRESSolve;
   }

protected:
   void
   postSolveHook() const override;
};

//! \brief Wrapper for the identity operator as a Hypre solver
//! \ingroup Hypre
class HypreIdentity : public HypreSolver
{
public:
   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentitySetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) hypre_ParKrylovIdentity;
   }
};

//! \brief Wrapper for the Jacobi preconditioner in Hypre
//! \ingroup Hypre
class HypreDiagScale : public HypreSolver
{
public:
   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScaleSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSRDiagScale;
   }
};

/**
 * \brief Wrapper for Hypre's preconditioner that is intended for matrices that
 * are triangular in some ordering.
 *
 * Finds correct ordering and performs forward substitution on processor as
 * approximate inverse. Exact on one processor.
 *
 * \ingroup Hypre
 */
class HypreTriSolve : public HypreSolver
{
public:
   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSROnProcTriSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParCSROnProcTriSolve;
   }
};

/**
 * \brief Wrapper for the ParaSails preconditioner in Hypre
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HypreParaSails precond;
 * HYPRE_ParaSailsSetSym(precond, 1);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HypreParaSails : public HypreSolver
{
public:
   explicit HypreParaSails( MPI_Comm comm );

   explicit HypreParaSails( const Matrices::HypreParCSRMatrix& A );

   ~HypreParaSails() override;

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ParaSailsSolve;
   }
};

/**
 * \brief Wrapper for the Euclid preconditioner in Hypre
 *
 * Euclid implements the Parallel Incomplete LU factorization technique. For
 * more information see:
 * "A Scalable Parallel Algorithm for Incomplete Factor Preconditioning" by
 * David Hysom and Alex Pothen, https://doi.org/10.1137/S1064827500376193
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HypreEuclid precond;
 * HYPRE_EuclidSetLevel(precond, 2);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HypreEuclid : public HypreSolver
{
public:
   explicit HypreEuclid( MPI_Comm comm );

   explicit HypreEuclid( const Matrices::HypreParCSRMatrix& A );

   ~HypreEuclid() override;

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_EuclidSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_EuclidSolve;
   }
};

/**
 * \brief Wrapper for Hypre's native parallel ILU preconditioner.
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HypreILU precond;
 * HYPRE_ILUSetLevelOfFill(precond, 1);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HypreILU : public HypreSolver
{
private:
   //! \brief Set the ILU default options
   void
   setDefaultOptions();

public:
   HypreILU();

   explicit HypreILU( const Matrices::HypreParCSRMatrix& A );

   ~HypreILU() override;

   //! Set the print level: 0 = none, 1 = setup, 2 = solve, 3 = setup+solve
   void
   setPrintLevel( HYPRE_Int print_level )
   {
      HYPRE_ILUSetPrintLevel( solver, print_level );
   }

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_ILUSolve;
   }
};

/** \brief Wrapper for the BoomerAMG solver/preconditioner in Hypre
 *
 * Note that the wrapper class sets its own default options that are different
 * from Hypre. By default, the instance is set up for preconditioner use, i.e.
 * with zero tolerance and one V cycle. At least these two parameters need to
 * be changed when used as a solver.
 *
 * Parameters can be set using native Hypre functions, e.g.
 *
 * \code
 * TNL::Solvers::Linear::HypreBoomerAMG solver;
 * HYPRE_BoomerAMGSetTol(solver, 1e-7);
 * HYPRE_BoomerAMGSetMaxIter(solver, 20);
 * \endcode
 *
 * See the [Hypre Reference Manual][manual] for the available parameters.
 *
 * [manual]: https://hypre.readthedocs.io/_/downloads/en/latest/pdf/
 *
 * \ingroup Hypre
 */
class HypreBoomerAMG : public HypreSolver
{
private:
   //! \brief Default, generally robust, BoomerAMG options
   void
   setDefaultOptions();

public:
   HypreBoomerAMG();

   explicit HypreBoomerAMG( const Matrices::HypreParCSRMatrix& A );

   ~HypreBoomerAMG() override;

   //! \brief More robust options for systems of PDEs
   void
   setSystemsOptions( int dim, bool order_bynodes = false );

   /**
    * \brief Set parameters to use AIR AMG solver for advection-dominated problems.
    *
    * See "Nonsymmetric Algebraic Multigrid Based on Local Approximate Ideal
    * Restriction (AIR)," Manteuffel, Ruge, Southworth, SISC (2018),
    * DOI:/10.1137/17M1144350.
    *
    * \param restrict_type Defines which parallel restriction operator is used.
    *                      There are the following options:
    *                      0: transpose of the interpolation operator
    *                      1: AIR-1 - Approximate Ideal Restriction (distance 1)
    *                      2: AIR-2 - Approximate Ideal Restriction (distance 2)
    * \param relax_order Defines in which order the points are relaxed. There
    *                    are the following options:
    *                    0: the points are relaxed in natural or lexicographic
    *                       order on each processor
    *                    1: CF-relaxation is used
    */
   void
   setAdvectiveOptions( int restrict_type = 0, int relax_order = 1 );

   //! Set the print level: 0 = none, 1 = setup, 2 = solve, 3 = setup+solve
   void
   setPrintLevel( int print_level )
   {
      HYPRE_BoomerAMGSetPrintLevel( solver, print_level );
   }

   void
   setMaxIter( int max_iter )
   {
      HYPRE_BoomerAMGSetMaxIter( solver, max_iter );
   }

   void
   setTol( double tol )
   {
      HYPRE_BoomerAMGSetTol( solver, tol );
   }

   HYPRE_Int
   getNumIterations() const
   {
      HYPRE_Int num_it;
      HYPRE_BoomerAMGGetNumIterations( solver, &num_it );
      return num_it;
   }

   HYPRE_PtrToParSolverFcn
   setupFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSetup;
   }

   HYPRE_PtrToParSolverFcn
   solveFcn() const override
   {
      return (HYPRE_PtrToParSolverFcn) HYPRE_BoomerAMGSolve;
   }

protected:
   void
   postSolveHook() const override;
};

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL

   #include "Hypre.hpp"

#endif  // HAVE_HYPRE
