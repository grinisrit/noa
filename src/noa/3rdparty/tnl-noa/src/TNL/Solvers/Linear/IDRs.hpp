// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <random>

#include "IDRs.h"

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Multireduction.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Factorization/LUsequential.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
void
IDRs< Matrix >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< int >( prefix + "idr-s", "Dimension of the shadow space in IDR(s).", 4 );
   config.addEntry< bool >( prefix + "idr-residual-smoothing", "Enables residual smoothing in IDR(s).", false );
}

template< typename Matrix >
bool
IDRs< Matrix >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   if( parameters.checkParameter( prefix + "idr-s" ) )
      s = parameters.getParameter< int >( "idr-s" );
   if( parameters.checkParameter( prefix + "idr-residual-smoothing" ) )
      smoothing = parameters.getParameter< bool >( "idr-residual-smoothing" );
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
void
IDRs< Matrix >::setShadowSpaceDimension( int s )
{
   this->s = s;
}

template< typename Matrix >
void
IDRs< Matrix >::setResidualSmoothing( bool smoothing )
{
   this->smoothing = smoothing;
}

template< typename Matrix >
bool
IDRs< Matrix >::solve( ConstVectorViewType b, VectorViewType x )
{
   this->setSize( x );

   // initial binding to x sets the correct local range, global size and
   // communicator for distributed views
   VectorViewType P_i( x ), U_k( x ), U_i( x ), G_k( x ), G_i( x );

   // initialize the norm of the right-hand-side
   RealType b_norm = TNL::l2Norm( b );

   // check for zero rhs - solution is the null vector
   if( b_norm == 0 ) {
      x = 0;
      return true;
   }

   // r = b - A * x
   this->matrix->vectorProduct( x, r );
   r = b - r;
   RealType r_norm = TNL::l2Norm( r );
   if( std::isnan( r_norm ) )
      throw std::runtime_error( "IDR(s): initial residue is NAN" );

   // initialization
   const RealType angle = 0.7;
   RealType omega = 1.0;
   G.setValue( 0 );
   U.setValue( 0 );
   // initialize M as the identity matrix (ones on the diagonal)
   M.setValue( 0 );
   for( int i = 0; i < s; i++ )
      M( i, i ) = 1;

   if( smoothing ) {
      x_s = x;
      r_s = r;
   }

   this->resetIterations();
   this->setResidue( r_norm / b_norm );

   // main iteration loop, build G-spaces:
   while( this->checkNextIteration() ) {
      // new right-hand size for small system
      // f = P.dot(r)
      const RealType* _P = P.getData();
      const RealType* _r = Traits::getLocalView( r ).getData();
      const IndexType sizeWithGhosts = this->sizeWithGhosts;
      auto fetch = [ _P, _r, sizeWithGhosts ] __cuda_callable__( IndexType idx, int i )
      {
         return _P[ idx + i * sizeWithGhosts ] * _r[ idx ];
      };
      Algorithms::Multireduction< DeviceType >::reduce( (RealType) 0, fetch, std::plus<>{}, size, s, f.getData() );
      // no-op if the problem is not distributed
      MPI::Allreduce( f.getData(), s, MPI_SUM, Traits::getCommunicator( *this->matrix ) );

      for( int k = 0; k < s; k++ ) {
         U_k.bind( &U.getData()[ k * sizeWithGhosts ], sizeWithGhosts );
         G_k.bind( &G.getData()[ k * sizeWithGhosts ], sizeWithGhosts );

         // solve the small system
         // c = solve(M[k:s, k:s], f[k:s])
         std::unique_ptr< RealType[] > c{ new RealType[ s - k ] };
         {
            HostMatrix matrix;
            matrix.setDimensions( s - k, s - k );
            for( int i = k; i < s; i++ )
               for( int j = k; j < s; j++ )
                  matrix( i - k, j - k ) = M( i, j );
            Matrices::Factorization::LU_sequential_factorize( matrix );
            typename HostVector::ViewType c_view( c.get(), s - k );
            Matrices::Factorization::LU_sequential_solve( matrix, f.getView( k, s ), c_view );
         }

         // make v orthogonal to P
         // v = r - G[:, k:s].dot(c)
         v = r;
         Matrices::MatrixOperations< DeviceType >::gemv( sizeWithGhosts,
                                                         IndexType( s - k ),
                                                         RealType( -1 ),
                                                         G.getData() + k * sizeWithGhosts,
                                                         sizeWithGhosts,
                                                         c.get(),
                                                         RealType( 1 ),
                                                         Traits::getLocalView( v ).getData() );
         // preconditioning
         psolve( v, v );

         // compute new U(:,k)
         // U_k = U[:, k:s].dot(c) + omega * v;
         // GOTCHA: U_k is included in the lhs as well as rhs, but it works with
         //         this gemv implementation (as long as beta=0)
         Matrices::MatrixOperations< DeviceType >::gemv( sizeWithGhosts,
                                                         IndexType( s - k ),
                                                         RealType( 1 ),
                                                         U.getData() + k * sizeWithGhosts,
                                                         sizeWithGhosts,
                                                         c.get(),
                                                         RealType( 0 ),
                                                         Traits::getLocalView( U_k ).getData() );
         U_k += omega * v;

         // compute new G(:,k) in the space G_j
         matvec( U_k, G_k );

         // bi-orthogonalize the new basis vectors
         for( int i = 0; i < k; i++ ) {
            P_i.bind( &P.getData()[ i * sizeWithGhosts ], sizeWithGhosts );
            U_i.bind( &U.getData()[ i * sizeWithGhosts ], sizeWithGhosts );
            G_i.bind( &G.getData()[ i * sizeWithGhosts ], sizeWithGhosts );

            const RealType alpha = TNL::dot( P_i, G_k ) / M( i, i );
            U_k = U_k - alpha * U_i;
            G_k = G_k - alpha * G_i;
         }

         // aux[0:s-k] = P[:,k:s] * G[:,k]
         std::unique_ptr< RealType[] > aux{ new RealType[ s - k ] };
         {
            const RealType* _P = P.getData();
            const RealType* _G = G.getData();
            const IndexType sizeWithGhosts = this->sizeWithGhosts;
            auto fetch = [ _P, _G, sizeWithGhosts, k ] __cuda_callable__( IndexType idx, int i )
            {
               return _P[ idx + ( i + k ) * sizeWithGhosts ] * _G[ idx + k * sizeWithGhosts ];
            };
            Algorithms::Multireduction< DeviceType >::reduce( (RealType) 0, fetch, std::plus<>{}, size, s - k, aux.get() );
            // no-op if the problem is not distributed
            MPI::Allreduce( aux.get(), s - k, MPI_SUM, Traits::getCommunicator( *this->matrix ) );
         }

         // new column of M = P'*G  (first k entries are zero)
         // M[k:s,k] = aux[0:s-k]
         for( int i = k; i < s; i++ )
            M( i, k ) = aux[ i - k ];

         // check breakdown
         if( M( k, k ) == 0 )
            return false;

         // make r orthogonal to g_i, i = 0..k-1
         const RealType beta = f[ k ] / M( k, k );
         x = x + beta * U_k;
         r = r - beta * G_k;
         r_norm = TNL::l2Norm( r );

         // residual smoothing
         if( smoothing ) {
            t = r_s - r;
            const RealType gamma = TNL::dot( t, r_s ) / TNL::dot( t, t );
            r_s = r_s - gamma * t;
            x_s = x_s - gamma * ( x_s - x );
            r_norm = TNL::l2Norm( r_s );
         }

         this->setResidue( r_norm / b_norm );
         if( ! this->nextIteration() ) {
            if( smoothing )
               x = x_s;
            return this->checkConvergence();
         }

         // new f = P'*r (first k components are zero)
         if( k < s - 1 )
            // f[k + 1:s] = f[k + 1:s] - beta * M[k + 1:s,k];
            for( int i = k + 1; i < s; i++ )
               f[ i ] = f[ i ] - beta * M( i, k );
      }

      // Now we have sufficient vectors in G_j to compute residual in G_j+1
      // Note: r is already perpendicular to P so v = r

      // preconditioning
      psolve( r, v );

      // matrix-vector multiplication
      matvec( v, t );

      // Computation of a new omega
      const RealType t_norm = TNL::l2Norm( t );
      const RealType tr = TNL::dot( t, r );
      const RealType rho = TNL::abs( tr / ( t_norm * r_norm ) );
      omega = tr / ( t_norm * t_norm );
      if( rho < angle )
         omega = omega * angle / rho;

      // check breakdown
      if( omega == 0 )
         return false;

      // new vector in G_j+1
      x = x + omega * v;
      r = r - omega * t;
      r_norm = TNL::l2Norm( r );

      // residual smoothing
      if( smoothing ) {
         t = r_s - r;
         const RealType gamma = TNL::dot( t, r_s ) / TNL::dot( t, t );
         r_s = r_s - gamma * t;
         x_s = x_s - gamma * ( x_s - x );
         r_norm = TNL::l2Norm( r_s );
      }

      this->setResidue( r_norm / b_norm );
      this->nextIteration();
   }

   if( smoothing )
      x = x_s;

   return this->checkConvergence();
}

template< typename Matrix >
void
IDRs< Matrix >::psolve( ConstVectorViewType src, VectorViewType dst )
{
   if( this->preconditioner )
      this->preconditioner->solve( src, dst );
   else
      dst = src;
}

template< typename Matrix >
void
IDRs< Matrix >::matvec( ConstVectorViewType src, VectorViewType dst )
{
   this->matrix->vectorProduct( src, dst );
}

template< typename Matrix >
void
IDRs< Matrix >::setSize( const VectorViewType& x )
{
   size = Traits::getConstLocalView( x ).getSize();
   sizeWithGhosts = Traits::getConstLocalViewWithGhosts( x ).getSize();
   P.setSize( s * sizeWithGhosts );
   G.setSize( s * sizeWithGhosts );
   U.setSize( s * sizeWithGhosts );
   r.setLike( x );
   v.setLike( x );
   t.setLike( x );
   if( smoothing ) {
      x_s.setLike( x );
      r_s.setLike( x );
   }
   else {
      x_s.reset();
      r_s.reset();
   }
   M.setDimensions( s, s );
   f.setSize( s );

   // initialize random number generator
   std::mt19937 rng;
   rng.seed( 0 );
   std::normal_distribution< RealType > dist( 0, 1 );

   // generate P on the host, copy to the device
   HostVector P_host( s * sizeWithGhosts );
   for( IndexType i = 0; i < P_host.getSize(); i++ )
      P_host[ i ] = dist( rng );
   P = P_host;

   // initial binding to x sets the correct local range, global size and
   // communicator for distributed views
   VectorViewType P_i( x ), P_j( x );

   // make the columns of P orthonormal
   for( int i = 0; i < s; i++ ) {
      P_i.bind( &P.getData()[ i * sizeWithGhosts ], sizeWithGhosts );

      for( int j = 0; j < i; j++ ) {
         P_j.bind( &P.getData()[ j * sizeWithGhosts ], sizeWithGhosts );
         P_i = P_i - TNL::dot( P_i, P_j );
      }

      // normalize the column
      P_i /= TNL::l2Norm( P_i );
   }
}

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL
