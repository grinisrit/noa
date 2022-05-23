# Linear solvers tutorial

[TOC]

## Introduction

Solvers of linear systems are one of the most important algorithms in scientific computations. TNL offers the followiing iterative methods:

1. Stationary methods
   1. [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method) (\ref TNL::Solvers::Linear::Jacobi)
   2. [Successive-overrelaxation method, SOR]([https://en.wikipedia.org/wiki/Successive_over-relaxation]) (\ref TNL::Solvers::Linear::SOR)
2. Krylov subspace methods
   1. [Conjugate gradient method, CG](https://en.wikipedia.org/wiki/Conjugate_gradient_method) (\ref TNL::Solvers::Linear::CG)
   2. [Biconjugate gradient stabilized method, BICGStab](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method) (\ref TNL::Solvers::Linear::BICGStab)
   3. [Biconjugate gradient stabilized method, BICGStab(l)](https://dspace.library.uu.nl/bitstream/handle/1874/16827/sleijpen_93_bicgstab.pdf) (\ref TNL::Solvers::Linear::BICGStabL)
   4. [Transpose-free quasi-minimal residual method, TFQMR]([https://second.wiki/wiki/algoritmo_tfqmr]) (\ref TNL::Solvers::Linear::TFQMR)
   5. [Generalized minimal residual method, GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) (\ref TNL::Solvers::Linear::GMRES) with various methods of orthogonalization
      1. Classical Gramm-Schmidt, CGS
      2. Classical Gramm-Schmidt with reorthogonalization, CGSR
      3. Modified Gramm-Schmidt, MGS
      4. Modified Gramm-Schmidt with reorthogonalization, MGSR
      5. Compact WY form of the Householder reflections, CWY

The iterative solvers (not the stationary solvers like \ref TNL::Solvers::Linear::Jacobi and \ref TNL::Solevrs::Linear::SOR) can be combined with the following preconditioners:

1. [Diagonal or Jacobi](http://netlib.org/linalg/html_templates/node55.html) - \ref TNL::Solvers::Linear::Preconditioners::Diagonal
2. ILU (Incomplete LU) - CPU only currently
   1. [ILU(0)](https://en.wikipedia.org/wiki/Incomplete_LU_factorization) \ref TNL::Solvers::Linear::Preconditioners::ILU0
   2. [ILUT (ILU with thresholding)](https://www-users.cse.umn.edu/~saad/PDF/umsi-92-38.pdf) \ref TNL::Solvers::Linear::Preconditioners::ILUT

## Iterative solvers of linear systems

### Basic setup

All iterative solvers for linear systems can be found in the namespace \ref TNL::Solvers::Linear. The following example shows the use the iterative solvers:

\includelineno Solvers/Linear/IterativeLinearSolverExample.cpp

In this example we solve a linear system \f$ A \vec x = \vec b \f$ where

\f[
A = \left(
\begin{array}{cccc}
 2.5 & -1   &      &      &      \\
-1   &  2.5 & -1   &      &      \\
     & -1   &  2.5 & -1   &      \\
     &      & -1   &  2.5 & -1   \\
     &      &      & -1   &  2.5 \\
\end{array}
\right)
\f]

The right-hand side vector \f$\vec b \f$ is set to \f$( 1.5, 0.5, 0.5, 0.5, 1.5 )^T \f$ so that the exact solution is \f$ \vec x = ( 1, 1, 1, 1, 1 )^T\f$. The matrix elements of \f$A $\f$ is set on the lines 12-51 by the means of the method \ref TNL::Matrices::SparseMatrix::forAllElements. In this example, we use the sparse matrix but any other matrix type can be used as well (see the namespace \ref TNL::Matrices). Next we set the solution vector \f$ \vec x = ( 1, 1, 1, 1, 1 )^T\f$ (line 57) and multiply it with matrix \f$ A \f$ to get the right-hand side vector \f$\vec b\f$ (lines 58-59). Finally, we reset the vector \f$\vec x \f$ to zero vector.

To solve the linear system, we use TFQMR method (line 66), as an example. Other solvers can be used as well (see the namespace \ref TNL::Solvers::Linear). The solver needs only one template parameter which is the matrix type. Next we create an instance of the solver (line 67 ) and set the matrix of the linear system (line 68). Note, that matrix is passed to the solver as a shared smart pointer (\ref std::shared_ptr). This is why we created an instance of the smart pointer on the line 24 instead of the sparse matrix itself. On the line 69, we set the value of residue under which the convergence occur and the solver is stopped. It serves as a convergence criterion. The solver is executed on the line 70 by calling the method \ref TNL::Solvers::Linear::LinearSolver::solve. The method accepts the right-hand side vector \f$ \vec b\f$ and the solution vector \f$ \vec x\f$.

The result looks as follows:

\include IterativeLinearSolverExample.out

### Setup with a solver monitor

Solution of large linear systems may take a lot of time. In such situations, it is useful to be able to monitor the convergence of the solver of the solver status in general. For this purpose, TNL offers solver monitors. The solver monitor prints (or somehow visualizes) the number of iterations, the residue of the current solution approximation or some other metrics. Sometimes such information is printed after each iteration or after every ten iterations. The problem of this approach is the fact that one iteration of the solver may take only few milliseconds but also several minutes. In the former case, the monitor creates overwhelming amount of output which may even slowdown the solver. In the later case, the user waits long time for update of the solver status. The monitor in TNL rather runs in separate thread and it refreshes the status of the solver in preset time periods. The use of the iterative solver monitor is demonstrated in the following example.

\includelineno Solvers/Linear/IterativeLinearSolverWithMonitorExample.cpp

On the lines 1-70, we setup the same linear system as in the previous example, we create an instance of the Jacobi solver and we pass the matrix of the linear system to the solver. On the line 71, we set the relaxation parameter \f$ \omega \f$ of the Jacobi solver to 0.0005 (\ref TNL::Solvers::Linear::Jacobi). The reason is to slowdown the convergence because we want to see some iterations in this example. Next we create an instance of the solver monitor (lines 76 and 77) and we create a special thread for the monitor (line 78, \ref TNL::Solvers::SolverMonitorThread ). We set the refresh rate of the monitor to 10 milliseconds (line 79, \ref TNL::Solvers::SolverMonitor::setRefreshRate). We set a verbosity of the monitor to 1 (line 80 \ref TNL::Solvers::IterativeSolverMonitor::setVerbosity ). Next we set a name of the solver stage (line 81, \ref TNL::Solvers::IterativeSolverMonitor::setStage). The monitor stages serve for distinguishing between different phases or stages of more complex solvers (for example when the linear system solver is embedded into a time dependent PDE solver). Next we connect the solver with the monitor (line 82, \ref TNL::Solvers::IterativeSolver::setSolverMonitor) and we set the convergence criterion based on the value of the residue (line 83). Finally we start the solver (line 84, \ref TNL::Solvers::Linear::Jacobi::start) and when the solver finishes we have to stop the monitor (line 84, \ref TNL::Solvers::SolverMonitor::stopMainLoop).

The result looks as follows:

\include IterativeLinearSolverWithMonitorExample.out

The monitoring of the solver can be improved by time elapsed since the beginning of the computation as demonstrated in the following example:

\includelineno Solvers/Linear/IterativeLinearSolverWithTimerExample.cpp

The only changes happen on lines 83-85 where we create an instance of TNL timer (line 83, \ref TNL::Timer), connect it with the monitor (line 84, \ref TNL::Solvers::SolverMonitor::setTimer) and start the timer (line 85, \ref TNL::Timer::start).

The result looks as follows:

\include IterativeLinearSolverWithTimerExample.out

### Setup with preconditioner

Preconditioners of iterative solvers can significantly improve the performance of the solver. In the case of the linear systems, they are used mainly with the Krylov subspace methods. Preconditioners cannot be used with the starionary methods (\ref TNL::Solvers::Linear::Jacobi and \ref TNL::Solvers::Linear::SOR). The following example shows how to setup an iterative solver of linear systems with preconditioning.

\includelineno Solvers/Linear/IterativeLinearSolverWithPreconditionerExample.cpp

In this example, we solve the same problem as in all other examples in this section. The only differences concerning the preconditioner happen on the lines (68-72). Similar to the matrix of the linear system, the preconditioner is passed to the solver by the means of  smart shared pointer (\ref std::shared_ptr). The instance is created on the lines 68 and 69. Next we have to initialize the preconditioner (line 70, \ref TNL::Solvers::Linear::Preconditioners::Preconditioner::update). The method `update` has to be called everytime the matrix of the linear system changes. This is important for example when solving time dependent PDEs but it does not happen in this example. Finally, we need to connect the solver with the preconditioner (line 73, \ref TNL::Solvers::Linear::LinearSolver).

The result looks as follows:

\include IterativeLinearSolverWithPreconditionerExample.out

### Choosing the solver and preconditioner type at runtime

When developing a numerical solver, one often has to search for a combination of various methods and algorithms that fit given requirements the best. To make this easier, TNL offers choosing the type of both linear solver and preconditioner at runtime by means of functions \ref TNL::Solvers::getLinearSolver and \ref TNL::Solvers::getPreconditioner. The following example shows how to use these functions:

\includelineno Solvers/Linear/IterativeLinearSolverWithRuntimeTypesExample.cpp

We still stay with the same problem and the only changes can be seen on lines 66-70. We first create an instance of shared pointer holding the solver (line 66, \ref TNL::Solvers::getLinearSolver) and the same with the preconditioner (line 67, \ref TNL::Solvers::getPreconditioner). The rest of the code is the same as in the previous examples with the only difference that we work with the pointer `solver_ptr` instead of the direct instance `solver` of the solver type.

The result looks as follows:

\include IterativeLinearSolverWithRuntimeTypesExample.out
