// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include "BICGStab.h"

namespace noa::TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
void
BICGStab< Matrix >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< bool >(
      prefix + "bicgstab-exact-residue",
      "Whether the BiCGstab should compute the exact residue in each step (true) or to use a cheap approximation (false).",
      false );
}

template< typename Matrix >
bool
BICGStab< Matrix >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   if( parameters.checkParameter( prefix + "bicgstab-exact-residue" ) )
      exact_residue = parameters.getParameter< bool >( "bicgstab-exact-residue" );
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
bool
BICGStab< Matrix >::solve( ConstVectorViewType b, VectorViewType x )
{
   this->setSize( x );

   RealType alpha, beta, omega, rho, rho_old, b_norm, r_ast_sqnorm;

   // initialize the norm of the preconditioned right-hand-side
   if( this->preconditioner ) {
      this->preconditioner->solve( b, M_tmp );
      b_norm = lpNorm( M_tmp, 2.0 );
   }
   else
      b_norm = lpNorm( b, 2.0 );

   // check for zero rhs - solution is the null vector
   if( b_norm == 0 ) {
      x = 0;
      return true;
   }

   // r = M.solve(b - A * x);
   compute_residue( r, x, b );

   p = r_ast = r;
   s.setValue( 0.0 );
   r_ast_sqnorm = rho = ( r, r_ast );

   const RealType eps2 = std::numeric_limits< RealType >::epsilon() * std::numeric_limits< RealType >::epsilon();

   this->resetIterations();
   this->setResidue( std::sqrt( rho ) / b_norm );

   while( this->nextIteration() ) {
      // alpha_j = ( r_j, r^ast_0 ) / ( A * p_j, r^ast_0 )
      preconditioned_matvec( p, Ap );
      alpha = rho / ( Ap, r_ast );

      // s_j = r_j - alpha_j * A p_j
      s = r - alpha * Ap;

      // omega_j = ( A s_j, s_j ) / ( A s_j, A s_j )
      preconditioned_matvec( s, As );
      omega = ( As, s ) / ( As, As );

      // x_{j+1} = x_j + alpha_j * p_j + omega_j * s_j
      x += alpha * p + omega * s;

      // r_{j+1} = s_j - omega_j * A s_j
      r = s - omega * As;

      // compute scalar product of the residual vectors
      rho_old = rho;
      rho = ( r, r_ast );
      if( abs( rho ) < eps2 * r_ast_sqnorm ) {
         // The new residual vector has become too orthogonal to the arbitrarily chosen direction r_ast.
         // Let's restart with a new r0:
         compute_residue( r, x, b );
         r_ast = r;
         r_ast_sqnorm = rho = ( r, r_ast );
      }

      // beta = alpha_j / omega_j * ( r_{j+1}, r^ast_0 ) / ( r_j, r^ast_0 )
      beta = ( rho / rho_old ) * ( alpha / omega );

      // p_{j+1} = r_{j+1} + beta_j * ( p_j - omega_j * A p_j )
      p = r + beta * p - ( beta * omega ) * Ap;

      if( exact_residue ) {
         // Compute the exact preconditioned residue into the 's' vector.
         compute_residue( s, x, b );
         const RealType residue = lpNorm( s, 2.0 );
         this->setResidue( residue / b_norm );
      }
      else {
         // Use the "orthogonal residue vector" for stopping.
         const RealType residue = lpNorm( r, 2.0 );
         this->setResidue( residue / b_norm );
      }
   }

   return this->checkConvergence();
}

template< typename Matrix >
void
BICGStab< Matrix >::compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b )
{
   // r = M.solve(b - A * x);
   if( this->preconditioner ) {
      this->matrix->vectorProduct( x, M_tmp );
      M_tmp = b - M_tmp;
      this->preconditioner->solve( M_tmp, r );
   }
   else {
      this->matrix->vectorProduct( x, r );
      r = b - r;
   }
}

template< typename Matrix >
void
BICGStab< Matrix >::preconditioned_matvec( ConstVectorViewType src, VectorViewType dst )
{
   if( this->preconditioner ) {
      this->matrix->vectorProduct( src, M_tmp );
      this->preconditioner->solve( M_tmp, dst );
   }
   else {
      this->matrix->vectorProduct( src, dst );
   }
}

template< typename Matrix >
void
BICGStab< Matrix >::setSize( const VectorViewType& x )
{
   r.setLike( x );
   r_ast.setLike( x );
   p.setLike( x );
   s.setLike( x );
   Ap.setLike( x );
   As.setLike( x );
   M_tmp.setLike( x );
}

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL
