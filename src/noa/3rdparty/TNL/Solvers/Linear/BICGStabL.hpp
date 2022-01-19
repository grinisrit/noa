// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include "BICGStabL.h"

#include <TNL/Matrices/MatrixOperations.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
void
BICGStabL< Matrix >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< int >( prefix + "bicgstab-ell", "Number of Bi-CG iterations before the MR part starts.", 1 );
   config.addEntry< bool >( prefix + "bicgstab-exact-residue", "Whether the BiCGstab should compute the exact residue in each step (true) or to use a cheap approximation (false).", false );
}

template< typename Matrix >
bool
BICGStabL< Matrix >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( parameters.checkParameter( prefix + "bicgstab-ell" ) )
      ell = parameters.getParameter< int >( "bicgstab-ell" );
   if( parameters.checkParameter( prefix + "bicgstab-exact-residue" ) )
      exact_residue = parameters.getParameter< bool >( "bicgstab-exact-residue" );
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
bool
BICGStabL< Matrix >::
solve( ConstVectorViewType b, VectorViewType x )
{
   this->setSize( x );

   RealType alpha, beta, gamma, rho_0, rho_1, omega, b_norm;
   // initial binding to M_tmp sets the correct local range, global size and
   // communicator for distributed views
   VectorViewType r_0( M_tmp ), r_j( M_tmp ), r_i( M_tmp ), u_0( M_tmp ), Au( M_tmp ), u( M_tmp );
   r_0.bind( R.getData(), size );
   u_0.bind( U.getData(), size );

   // initialize the norm of the preconditioned right-hand-side
   if( this->preconditioner ) {
      this->preconditioner->solve( b, M_tmp );
      b_norm = lpNorm( M_tmp, 2.0 );
   }
   else
      b_norm = lpNorm( b, 2.0 );
   if( b_norm == 0.0 )
      b_norm = 1.0;

   // r_0 = M.solve(b - A * x);
   compute_residue( r_0, x, b );

   sigma[ 0 ] = lpNorm( r_0, 2.0 );
   if( std::isnan( sigma[ 0 ] ) )
      throw std::runtime_error( "BiCGstab(ell): initial residue is NAN" );

   r_ast = r_0;
   r_ast /= sigma[ 0 ];
   rho_0 = 1.0;
   alpha = 0.0;
   omega = 1.0;
   u_0.setValue( 0.0 );

   this->resetIterations();
   this->setResidue( sigma[ 0 ] / b_norm );

   while( this->checkNextIteration() )
   {
      rho_0 = - omega * rho_0;

      /****
       * Bi-CG part
       */
      for( int j = 0; j < ell; j++ ) {
         this->nextIteration();
         r_j.bind( &R.getData()[ j * ldSize ], size );

         rho_1 = (r_ast, r_j);
         beta = alpha * rho_1 / rho_0;
         rho_0 = rho_1;

         /****
          * U_[0:j] := R_[0:j] - beta * U_[0:j]
          */
         Matrices::MatrixOperations< DeviceType >::
            geam( size, (IndexType) j + 1,
                  (RealType) 1.0, R.getData(), ldSize,
                  -beta, U.getData(), ldSize,
                  U.getData(), ldSize );

         /****
          * u_{j+1} = A u_j
          */
         u.bind( &U.getData()[ j * ldSize ], size );
         Au.bind( &U.getData()[ (j + 1) * ldSize ], size );
         preconditioned_matvec( u, Au );

         gamma = (r_ast, Au);
         alpha = rho_0 / gamma;

         /****
          * R_[0:j] := R_[0:j] - alpha * U_[1:j+1]
          */
         Matrices::MatrixOperations< DeviceType >::
            geam( size, (IndexType) j + 1,
                  (RealType) 1.0, R.getData(), ldSize,
                  -alpha, U.getData() + ldSize, ldSize,
                  R.getData(), ldSize );

         /****
          * r_{j+1} = A r_j
          */
         r_j.bind( &R.getData()[ j * ldSize ], size );
         r_i.bind( &R.getData()[ (j + 1) * ldSize ], size );
         preconditioned_matvec( r_j, r_i );

         /****
          * x_0 := x_0 + alpha * u_0
          */
         x += alpha * u_0;
      }

      /****
       * MGS part
       */
      for( int j = 1; j <= ell; j++ ) {
         r_j.bind( &R.getData()[ j * ldSize ], size );

         // MGS without reorthogonalization
         for( int i = 1; i < j; i++ ) {
            r_i.bind( &R.getData()[ i * ldSize ], size );
            /****
             * T_{i,j} = (r_i, r_j) / sigma_i
             * r_j := r_j - T_{i,j} * r_i
             */
            const int ij = (i-1) + (j-1) * ell;
            T[ ij ] = (r_i, r_j) / sigma[ i ];
            r_j -= T[ ij ] * r_i;
         }

         // MGS with reorthogonalization
//         for( int i = 1; i < j; i++ ) {
//            const int ij = (i-1) + (j-1) * ell;
//            T[ ij ] = 0.0;
//         }
//         for( int l = 0; l < 2; l++ )
//            for( int i = 1; i < j; i++ ) {
//               r_i.bind( &R.getData()[ i * ldSize ], size );
//               /****
//                * T_{i,j} = (r_i, r_j) / sigma_i
//                * r_j := r_j - T_{i,j} * r_i
//                */
//               const int ij = (i-1) + (j-1) * ell;
//               const RealType T_ij = (r_i, r_j) / sigma[ i ];
//               T[ ij ] += T_ij;
//               r_j -= T_ij * r_i );
//            }

         sigma[ j ] = (r_j, r_j);
         g_1[ j ] = (r_0, r_j) / sigma[ j ];
      }

      omega = g_1[ ell ];

      /****
       * g_0 = T^{-1} g_1
       */
      for( int j = ell; j >= 1; j-- ) {
         g_0[ j ] = g_1[ j ];
         for( int i = j + 1; i <= ell; i++ )
            g_0[ j ] -= T[ (j-1) + (i-1) * ell ] * g_0[ i ];
      }

      /****
       * g_2 = T * S * g_0,
       * where S e_1 = 0, S e_j = e_{j-1} for j = 2, ... ell
       */
      for( int j = 1; j < ell; j++ ) {
         g_2[ j ] = g_0[ j + 1 ];
         for( int i = j + 1; i < ell; i++ )
            g_2[ j ] += T[ (j-1) + (i-1) * ell ] * g_0[ i + 1 ];
      }

      /****
       * Final updates
       */
      // x := x + R_[0:ell-1] * g_2
      g_2[ 0 ] = g_0[ 1 ];
      Matrices::MatrixOperations< DeviceType >::
         gemv( size, (IndexType) ell,
               (RealType) 1.0, R.getData(), ldSize, g_2.getData(),
               (RealType) 1.0, Traits::getLocalView( x ).getData() );
      // r_0 := r_0 - R_[1:ell] * g_1_[1:ell]
      Matrices::MatrixOperations< DeviceType >::
         gemv( size, (IndexType) ell,
               (RealType) -1.0, R.getData() + ldSize, ldSize, &g_1[ 1 ],
               (RealType) 1.0, Traits::getLocalView( r_0 ).getData() );
      // u_0 := u_0 - U_[1:ell] * g_0_[1:ell]
      Matrices::MatrixOperations< DeviceType >::
         gemv( size, (IndexType) ell,
               (RealType) -1.0, U.getData() + ldSize, ldSize, &g_0[ 1 ],
               (RealType) 1.0, Traits::getLocalView( u_0 ).getData() );

      if( exact_residue ) {
         /****
          * Compute the exact preconditioned residue into the 's' vector.
          */
         compute_residue( res_tmp, x, b );
         sigma[ 0 ] = lpNorm( res_tmp, 2.0 );
         this->setResidue( sigma[ 0 ] / b_norm );
      }
      else {
         /****
          * Use the "orthogonal residue vector" for stopping.
          */
         sigma[ 0 ] = lpNorm( r_0, 2.0 );
         this->setResidue( sigma[ 0 ] / b_norm );
      }
   }
   return this->checkConvergence();
}

template< typename Matrix >
void
BICGStabL< Matrix >::
compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b )
{
   /****
    * r = M.solve(b - A * x);
    */
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
BICGStabL< Matrix >::
preconditioned_matvec( ConstVectorViewType src, VectorViewType dst )
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
BICGStabL< Matrix >::
setSize( const VectorViewType& x )
{
   this->size = ldSize = Traits::getConstLocalView( x ).getSize();
   R.setSize( (ell + 1) * ldSize );
   U.setSize( (ell + 1) * ldSize );
   r_ast.setLike( x );
   M_tmp.setLike( x );
   if( exact_residue )
      res_tmp.setLike( x );
   T.setSize( ell * ell );
   sigma.setSize( ell + 1 );
   g_0.setSize( ell + 1 );
   g_1.setSize( ell + 1 );
   g_2.setSize( ell + 1 );
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL
