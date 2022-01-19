// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "CG.h"

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
bool
CG< Matrix >::
solve( ConstVectorViewType b, VectorViewType x )
{
   this->setSize( x );
   this->resetIterations();

   RealType alpha, beta, s1, s2;

   // initialize the norm of the preconditioned right-hand-side
   RealType normb;
   if( this->preconditioner ) {
      this->preconditioner->solve( b, r );
      normb = lpNorm( r, 2.0 );
   }
   else
      normb = lpNorm( b, 2.0 );
   if( normb == 0.0 )
      normb = 1.0;

   /****
    * r_0 = b - A x_0, p_0 = r_0
    */
   this->matrix->vectorProduct( x, r );
   r = b - r;

   if( this->preconditioner ) {
      // z_0 = M^{-1} r_0
      this->preconditioner->solve( r, z );
      // p_0 = z_0
      p = z;
      // s1 = (r_0, z_0)
      s1 = (r, z);
   }
   else {
      // p_0 = r_0
      p = r;
      // s1 = (r_0, r_0)
      s1 = (r, r);
   }

   this->setResidue( std::sqrt(s1) / normb );

   while( this->nextIteration() )
   {
      // s2 = (A * p_j, p_j)
      this->matrix->vectorProduct( p, Ap );
      s2 = (Ap, p);

      // if s2 = 0 => p = 0 => r = 0 => we have the solution (provided A != 0)
      if( s2 == 0.0 ) {
         this->setResidue( 0.0 );
         break;
      }

      // alpha_j = (r_j, z_j) / (A * p_j, p_j)
      alpha = s1 / s2;

      // x_{j+1} = x_j + alpha_j p_j
      x += alpha * p;

      // r_{j+1} = r_j - alpha_j A * p_j
      r -= alpha * Ap;

      if( this->preconditioner ) {
         // z_{j+1} = M^{-1} * r_{j+1}
         this->preconditioner->solve( r, z );
         // beta_j = (r_{j+1}, z_{j+1}) / (r_j, z_j)
         s2 = s1;
         s1 = (r, z);
      }
      else {
         // beta_j = (r_{j+1}, r_{j+1}) / (r_j, r_j)
         s2 = s1;
         s1 = (r, r);
      }

      // if s2 = 0 => r = 0 => we have the solution
      if( s2 == 0.0 ) beta = 0.0;
      else beta = s1 / s2;

      if( this->preconditioner )
         // p_{j+1} = z_{j+1} + beta_j * p_j
         p = z + beta * p;
      else
         // p_{j+1} = r_{j+1} + beta_j * p_j
         p = r + beta * p;

      this->setResidue( std::sqrt(s1) / normb );
   }
   return this->checkConvergence();
}

template< typename Matrix >
void CG< Matrix >::
setSize( const VectorViewType& x )
{
   r.setLike( x );
   p.setLike( x );
   Ap.setLike( x );
   z.setLike( x );
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
