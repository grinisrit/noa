// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>

#include "TFQMR.h"

namespace noa::TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
bool TFQMR< Matrix >::solve( ConstVectorViewType b, VectorViewType x )
{
   this->setSize( x );

   RealType tau, theta, eta, rho, alpha, b_norm, w_norm;

   if( this->preconditioner ) {
      this->preconditioner->solve( b, M_tmp );
      b_norm = lpNorm( M_tmp, 2.0 );

      this->matrix->vectorProduct( x, M_tmp );
      M_tmp = b - M_tmp;
      this->preconditioner->solve( M_tmp, r );
   }
   else {
      b_norm = lpNorm( b, 2.0 );
      this->matrix->vectorProduct( x, r );
      r = b - r;
   }
   w = u = r;
   if( this->preconditioner ) {
      this->matrix->vectorProduct( u, M_tmp );
      this->preconditioner->solve( M_tmp, Au );
   }
   else {
      this->matrix->vectorProduct( u, Au );
   }
   v = Au;
   d.setValue( 0.0 );
   tau = lpNorm( r, 2.0 );
   theta = eta = 0.0;
   r_ast = r;
   rho = (r_ast, r);
   // only to avoid compiler warning; alpha is initialized inside the loop
   alpha = 0.0;

   if( b_norm == 0.0 )
       b_norm = 1.0;

   this->resetIterations();
   this->setResidue( tau / b_norm );

   while( this->nextIteration() )
   {
      const IndexType iter = this->getIterations();

      if( iter % 2 == 1 ) {
         alpha = rho / (v, r_ast);
      }
      else {
         // not necessary in odd iter since the previous iteration
         // already computed v_{m+1} = A*u_{m+1}
         if( this->preconditioner ) {
            this->matrix->vectorProduct( u, M_tmp );
            this->preconditioner->solve( M_tmp, Au );
         }
         else {
            this->matrix->vectorProduct( u, Au );
         }
      }
      w -= alpha * Au;
      d = u + (theta * theta * eta / alpha) * d;
      w_norm = lpNorm( w, 2.0 );
      theta = w_norm / tau;
      const RealType c = 1.0 / std::sqrt( 1.0 + theta * theta );
      tau = tau * theta * c;
      eta = c * c  * alpha;
      x += eta * d;

      this->setResidue( tau * std::sqrt(iter+1) / b_norm );
      if( iter > this->getMinIterations() && this->getResidue() < this->getConvergenceResidue() ) {
          break;
      }

      if( iter % 2 == 0 ) {
         const RealType rho_new = (w, r_ast);
         const RealType beta = rho_new / rho;
         rho = rho_new;

         u = w + beta * u;
         v = beta * Au + (beta * beta) * v;
         if( this->preconditioner ) {
            this->matrix->vectorProduct( u, M_tmp );
            this->preconditioner->solve( M_tmp, Au );
         }
         else {
            this->matrix->vectorProduct( u, Au );
         }
         v += Au;
      }
      else {
         u -= alpha * v;
      }
   }
   return this->checkConvergence();
}

template< typename Matrix >
void TFQMR< Matrix > :: setSize( const VectorViewType& x )
{
   d.setLike( x );
   r.setLike( x );
   w.setLike( x );
   u.setLike( x );
   v.setLike( x );
   r_ast.setLike( x );
   Au.setLike( x );
   M_tmp.setLike( x );
}

} // namespace Linear
} // namespace Solvers
} // namespace noa::TNL
