// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <type_traits>
#include <cmath>

#include <noa/3rdparty/TNL/Algorithms/Multireduction.h>
#include <noa/3rdparty/TNL/Matrices/MatrixOperations.h>

#include "GMRES.h"

namespace noa::TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
void
GMRES< Matrix >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< String >( prefix + "gmres-variant", "Algorithm used for the orthogonalization.", "MGSR" );
      config.addEntryEnum( "CGS" );
      config.addEntryEnum( "CGSR" );
      config.addEntryEnum( "MGS" );
      config.addEntryEnum( "MGSR" );
      config.addEntryEnum( "CWY" );
   config.addEntry< int >( prefix + "gmres-restarting-min", "Minimal number of iterations after which the GMRES restarts.", 10 );
   config.addEntry< int >( prefix + "gmres-restarting-max", "Maximal number of iterations after which the GMRES restarts.", 10 );
   config.addEntry< int >( prefix + "gmres-restarting-step-min", "Minimal adjusting step for the adaptivity of the GMRES restarting parameter.", 3 );
   config.addEntry< int >( prefix + "gmres-restarting-step-max", "Maximal adjusting step for the adaptivity of the GMRES restarting parameter.", 3 );
}

template< typename Matrix >
bool
GMRES< Matrix >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( parameters.checkParameter( prefix + "gmres-variant" ) ) {
      const String var = parameters.getParameter< String >( prefix + "gmres-variant" );
      if( var == "CGS" )
         variant = Variant::CGS;
      else if( var == "CGSR" )
         variant = Variant::CGSR;
      else if( var == "MGS" )
         variant = Variant::MGS;
      else if( var == "MGSR" )
         variant = Variant::MGSR;
      else if( var == "CWY" )
         variant = Variant::CWY;
      else
         return false;
   }

   if( parameters.checkParameter( prefix + "gmres-restarting-min" ) )
      restarting_min = parameters.getParameter< int >( prefix + "gmres-restarting-min" );
   if( parameters.checkParameter( prefix + "gmres-restarting-max" ) )
      restarting_max = parameters.getParameter< int >( prefix + "gmres-restarting-max" );
   if( parameters.checkParameter( prefix + "gmres-restarting-step-min" ) )
      restarting_step_min = parameters.getParameter< int >( prefix + "gmres-restarting-step-min" );
   if( parameters.checkParameter( prefix + "gmres-restarting-step-max" ) )
      restarting_step_max = parameters.getParameter< int >( prefix + "gmres-restarting-step-max" );

   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
bool
GMRES< Matrix >::
solve( ConstVectorViewType b, VectorViewType x )
{
   TNL_ASSERT_TRUE( this->matrix, "No matrix was set in GMRES. Call setMatrix() before solve()." );
   if( restarting_min <= 0 || restarting_max <= 0 || restarting_min > restarting_max ) {
      std::cerr << "Wrong value for the GMRES restarting parameters: r_min = " << restarting_min
                << ", r_max = " << restarting_max << std::endl;
      return false;
   }
   if( restarting_step_min < 0 || restarting_step_max < 0 || restarting_step_min > restarting_step_max ) {
      std::cerr << "Wrong value for the GMRES restarting adjustment parameters: d_min = " << restarting_step_min
                << ", d_max = " << restarting_step_max << std::endl;
      return false;
   }
   setSize( x );

   // initialize the norm of the preconditioned right-hand-side
   RealType normb;
   if( this->preconditioner ) {
      this->preconditioner->solve( b, _M_tmp );
      normb = lpNorm( _M_tmp, 2.0 );
   }
   else
      normb = lpNorm( b, 2.0 );
   if( normb == 0.0 )
      normb = 1.0;

   // r = M.solve(b - A * x);
   compute_residue( r, x, b );
   RealType beta = lpNorm( r, 2.0 );

   // initialize stopping criterion
   this->resetIterations();
   this->setResidue( beta / normb );

   // parameters for the adaptivity of the restarting parameter
         RealType beta_ratio = 1;           // = beta / beta_ratio (small value indicates good convergence rate)
   const RealType max_beta_ratio = 0.99;    // = cos(8°) \approx 0.99
   const RealType min_beta_ratio = 0.175;   // = cos(80°) \approx 0.175
         int restart_cycles = 0;    // counter of restart cycles
         int m = restarting_max;    // current restarting parameter

   while( this->checkNextIteration() ) {
      // adaptivity of the restarting parameter
      // reference:  A.H. Baker, E.R. Jessup, Tz.V. Kolev - A simple strategy for varying the restart parameter in GMRES(m)
      //             http://www.sciencedirect.com/science/article/pii/S0377042709000132
      if( restarting_max > restarting_min && restart_cycles > 0 ) {
         if( beta_ratio > max_beta_ratio )
            // near stagnation -> set maximum
            m = restarting_max;
         else if( beta_ratio >= min_beta_ratio ) {
            // the step size is determined based on current m using linear interpolation
            // between restarting_step_min and restarting_step_max
            const int step = restarting_step_min + (float) ( restarting_step_max - restarting_step_min ) /
                                                           ( restarting_max - restarting_min ) *
                                                           ( m - restarting_min );
            if( m - step >= restarting_min )
               m -= step;
            else
               // set restarting_max when we hit restarting_min (see Baker et al. (2009))
               m = restarting_max;
         }
      }

      // orthogonalization
      int o_steps = 0;
      switch( variant ) {
         case Variant::CGS:
         case Variant::CGSR:
            o_steps = orthogonalize_CGS( m, normb, beta );
            break;
         case Variant::MGS:
         case Variant::MGSR:
            o_steps = orthogonalize_MGS( m, normb, beta );
            break;
         case Variant::CWY:
            o_steps = orthogonalize_CWY( m, normb, beta );
            break;
      }

      if( o_steps < m ) {
         // exact solution has been reached early
         update( o_steps, m, H, s, V, x );
         return this->checkConvergence();
      }

      // update the solution approximation
      update( m - 1, m, H, s, V, x );

      // compute the new residual vector
      compute_residue( r, x, b );
      const RealType beta_old = beta;
      beta = lpNorm( r, 2.0 );
      this->setResidue( beta / normb );

      // update parameters for the adaptivity of the restarting parameter
      ++restart_cycles;
      beta_ratio = beta / beta_old;
   }
   return this->checkConvergence();
}

template< typename Matrix >
int
GMRES< Matrix >::
orthogonalize_CGS( const int m, const RealType normb, const RealType beta )
{
   // initial binding to _M_tmp sets the correct local range, global size and
   // communicator for distributed views
   VectorViewType v_i( _M_tmp.getView() );
//   VectorViewType v_k( _M_tmp.getView() );

   /***
    * v_0 = r / | r | =  1.0 / beta * r
    */
   v_i.bind( V.getData(), size );
   v_i = (1.0 / beta) * r;

   H.setValue( 0.0 );
   s.setValue( 0.0 );
   s[ 0 ] = beta;

   /****
    * Starting m-loop
    */
   for( int i = 0; i < m && this->nextIteration(); i++ ) {
      v_i.bind( &V.getData()[ i * ldSize ], size );
      /****
       * Solve w from M w = A v_i
       */
      preconditioned_matvec( w, v_i );

      for( int k = 0; k <= i; k++ )
         H[ k + i * (m + 1) ] = 0.0;
      const int reorthogonalize = (variant == Variant::CGSR) ? 2 : 1;
      for( int l = 0; l < reorthogonalize; l++ ) {
         // auxiliary array for the H coefficients of the current l-loop
         RealType H_l[i + 1];

         // CGS part 1: compute projection coefficients
//         for( int k = 0; k <= i; k++ ) {
//            v_k.bind( &V.getData()[ k * ldSize ], size );
//            H_l[k] = (w, v_k);
//            H[ k + i * (m + 1) ] += H_l[k];
//         }
         // H_l = V_i^T * w
         const RealType* _V = V.getData();
         const RealType* _w = Traits::getConstLocalView( w ).getData();
         const IndexType ldSize = this->ldSize;
         auto fetch = [_V, _w, ldSize] __cuda_callable__ ( IndexType idx, int k ) { return _V[ idx + k * ldSize ] * _w[ idx ]; };
         Algorithms::Multireduction< DeviceType >::reduce
                  ( (RealType) 0,
                    fetch,
                    std::plus<>{},
                    size,
                    i + 1,
                    H_l );
         for( int k = 0; k <= i; k++ )
            H[ k + i * (m + 1) ] += H_l[k];

         // CGS part 2: subtract the projections
//         for( int k = 0; k <= i; k++ ) {
//            v_k.bind( &V.getData()[ k * ldSize ], size );
//            w = w - H_l[k] * v_k;
//         }
         // w := w - V_i * H_l
         Matrices::MatrixOperations< DeviceType >::
            gemv( size, (IndexType) i + 1,
                  (RealType) -1.0, V.getData(), ldSize, H_l,
                  (RealType) 1.0, Traits::getLocalView( w ).getData() );
      }
      /***
       * H_{i+1,i} = |w|
       */
      RealType normw = lpNorm( w, 2.0 );
      H[ i + 1 + i * (m + 1) ] = normw;

      /***
       * v_{i+1} = w / |w|
       */
      v_i.bind( &V.getData()[ ( i + 1 ) * ldSize ], size );
      v_i = (1.0 / normw) * w;

      /****
       * Applying the Givens rotations G_0, ..., G_i
       */
      apply_givens_rotations( i, m );

      this->setResidue( std::fabs( s[ i + 1 ] ) / normb );
      if( ! this->checkNextIteration() )
         return i;
   }

   return m;
}

template< typename Matrix >
int
GMRES< Matrix >::
orthogonalize_MGS( const int m, const RealType normb, const RealType beta )
{
   // initial binding to _M_tmp sets the correct local range, global size and
   // communicator for distributed views
   VectorViewType v_i( _M_tmp.getView() );
   VectorViewType v_k( _M_tmp.getView() );

   /***
    * v_0 = r / | r | =  1.0 / beta * r
    */
   v_i.bind( V.getData(), size );
   v_i = (1.0 / beta) * r;

   H.setValue( 0.0 );
   s.setValue( 0.0 );
   s[ 0 ] = beta;

   /****
    * Starting m-loop
    */
   for( int i = 0; i < m && this->nextIteration(); i++ ) {
      v_i.bind( &V.getData()[ i * ldSize ], size );
      /****
       * Solve w from M w = A v_i
       */
      preconditioned_matvec( w, v_i );

      for( int k = 0; k <= i; k++ )
         H[ k + i * (m + 1) ] = 0.0;
      const int reorthogonalize = (variant == Variant::MGSR) ? 2 : 1;
      for( int l = 0; l < reorthogonalize; l++ )
         for( int k = 0; k <= i; k++ ) {
            v_k.bind( &V.getData()[ k * ldSize ], size );
            /***
             * H_{k,i} = (w, v_k)
             */
            RealType H_k_i = (w, v_k);
            H[ k + i * (m + 1) ] += H_k_i;

            /****
             * w = w - H_{k,i} v_k
             */
            w = w - H_k_i * v_k;
         }
      /***
       * H_{i+1,i} = |w|
       */
      RealType normw = lpNorm( w, 2.0 );
      H[ i + 1 + i * (m + 1) ] = normw;

      /***
       * v_{i+1} = w / |w|
       */
      v_i.bind( &V.getData()[ ( i + 1 ) * ldSize ], size );
      v_i = (1.0 / normw) * w;

      /****
       * Applying the Givens rotations G_0, ..., G_i
       */
      apply_givens_rotations( i, m );

      this->setResidue( std::fabs( s[ i + 1 ] ) / normb );
      if( ! this->checkNextIteration() )
         return i;
   }

   return m;
}

template< typename Matrix >
int
GMRES< Matrix >::
orthogonalize_CWY( const int m, const RealType normb, const RealType beta )
{
   // initial binding to _M_tmp sets the correct local range, global size and
   // communicator for distributed views
   VectorViewType v_i( _M_tmp.getView() );
   VectorViewType y_i( _M_tmp.getView() );

   /***
    * z = r / | r | =  1.0 / beta * r
    */
   // TODO: investigate normalization by beta and normb
//   z = (1.0 / beta) * r;
//   z = (1.0 / normb) * r;
   z = r;

   H.setValue( 0.0 );
   s.setValue( 0.0 );
   T.setValue( 0.0 );

   // NOTE: this is unstable, s[0] is set later in hauseholder_apply_trunc
//   s[ 0 ] = beta;

   /****
    * Starting m-loop
    */
   for( int i = 0; i <= m && this->nextIteration(); i++ ) {
      /****
       * Generate new Hauseholder transformation from vector z.
       */
      y_i.bind( &Y.getData()[ i * ldSize ], size );
      hauseholder_generate( i, y_i, z );

      if( i == 0 ) {
         /****
          * s = e_1^T * P_i * z
          */
         hauseholder_apply_trunc( s, i, y_i, z );
      }
      else {
         /***
          * H_{i-1} = P_i * z
          */
         HostView h( &H.getData()[ (i - 1) * (m + 1) ], m + 1 );
         hauseholder_apply_trunc( h, i, y_i, z );
      }

      /***
       * Generate new basis vector v_i, using the compact WY representation:
       *     v_i = (I - Y_i * T_i Y_i^T) * e_i
       */
      // vectors v_i are not stored, they can be reconstructed in the update() method
//      v_i.bind( &V.getData()[ i * ldSize ], size );
      v_i.bind( V.getData(), size );
      hauseholder_cwy( v_i, i );

      if( i < m ) {
         /****
          * Solve w from M w = A v_i
          */
         preconditioned_matvec( w, v_i );

         /****
          * Apply all previous Hauseholder transformations, using the compact WY representation:
          *     z = (I - Y_i * T_i^T * Y_i^T) * w
          */
         hauseholder_cwy_transposed( z, i, w );
      }

      /****
       * Applying the Givens rotations G_0, ..., G_{i-1}
       */
      if( i > 0 )
         apply_givens_rotations( i - 1, m );

      this->setResidue( std::fabs( s[ i ] ) / normb );
      if( i > 0 && ! this->checkNextIteration() )
         return i - 1;
  }

   return m;
}

template< typename Matrix >
void
GMRES< Matrix >::
compute_residue( VectorViewType r, ConstVectorViewType x, ConstVectorViewType b )
{
   /****
    * r = M.solve(b - A * x);
    */
   if( this->preconditioner ) {
      this->matrix->vectorProduct( x, _M_tmp );
      _M_tmp = b - _M_tmp;
      this->preconditioner->solve( _M_tmp, r );
   }
   else {
      this->matrix->vectorProduct( x, r );
      r = b - r;
   }
}

template< typename Matrix >
void
GMRES< Matrix >::
preconditioned_matvec( VectorViewType w, ConstVectorViewType v )
{
   /****
    * w = M.solve(A * v_i);
    */
   if( this->preconditioner ) {
      this->matrix->vectorProduct( v, _M_tmp );
      this->preconditioner->solve( _M_tmp, w );
   }
   else
      this->matrix->vectorProduct( v, w );
}

template< typename Matrix >
void
GMRES< Matrix >::
hauseholder_generate( const int i,
                      VectorViewType y_i,
                      ConstVectorViewType z )
{
   // XXX: the upper-right triangle of Y will be full of zeros, which can be exploited for optimization
   ConstDeviceView z_local = Traits::getConstLocalView( z );
   DeviceView y_i_local = Traits::getLocalView( y_i );
   if( localOffset == 0 ) {
      TNL_ASSERT_LT( i, size, "upper-right triangle of Y is not on rank 0" );
      auto kernel_truncation = [=] __cuda_callable__ ( IndexType j ) mutable
      {
         if( j < i )
            y_i_local[ j ] = 0.0;
         else
            y_i_local[ j ] = z_local[ j ];
      };
      Algorithms::ParallelFor< DeviceType >::exec( (IndexType) 0, size, kernel_truncation );
   }
   else {
      y_i_local = z_local;
   }

   // norm of the TRUNCATED vector z
   const RealType normz = lpNorm( y_i, 2.0 );
   RealType norm_yi_squared = 0;
   if( localOffset == 0 ) {
      const RealType y_ii = y_i.getElement( i );
      if( y_ii > 0.0 )
         y_i.setElement( i, y_ii + normz );
      else
         y_i.setElement( i, y_ii - normz );

      // compute the norm of the y_i vector; equivalent to this calculation by definition:
      //       norm_yi_squared = y_i.lpNorm( 2.0 );
      //       norm_yi_squared = norm_yi_squared * norm_yi_squared
      norm_yi_squared = 2 * (normz * normz + std::fabs( y_ii ) * normz);
   }
   // no-op if the problem is not distributed
   MPI::Bcast( &norm_yi_squared, 1, 0, Traits::getCommunicator( *this->matrix ) );

   // XXX: normalization is slower, but more stable
//   y_i *= 1.0 / std::sqrt( norm_yi_squared );
//   const RealType t_i = 2.0;
   // assuming it's stable enough...
   const RealType t_i = 2.0 / norm_yi_squared;

   T[ i + i * (restarting_max + 1) ] = t_i;
   if( i > 0 ) {
      // aux = Y_{i-1}^T * y_i
      RealType aux[ i ];
      const RealType* _Y = Y.getData();
      const RealType* _y_i = Traits::getConstLocalView( y_i ).getData();
      const IndexType ldSize = this->ldSize;
      auto fetch = [_Y, _y_i, ldSize] __cuda_callable__ ( IndexType idx, int k ) { return _Y[ idx + k * ldSize ] * _y_i[ idx ]; };
      Algorithms::Multireduction< DeviceType >::reduce
               ( (RealType) 0,
                 fetch,
                 std::plus<>{},
                 size,
                 i,
                 aux );
      // no-op if the problem is not distributed
      MPI::Allreduce( aux, i, MPI_SUM, Traits::getCommunicator( *this->matrix ) );

      // [T_i]_{0..i-1} = - T_{i-1} * t_i * aux
      for( int k = 0; k < i; k++ ) {
         T[ k + i * (restarting_max + 1) ] = 0.0;
         for( int j = k; j < i; j++ )
            T[ k + i * (restarting_max + 1) ] -= T[ k + j * (restarting_max + 1) ] * (t_i * aux[ j ]);
      }
   }
}

template< typename Matrix >
void
GMRES< Matrix >::
hauseholder_apply_trunc( HostView out,
                         const int i,
                         VectorViewType y_i,
                         ConstVectorViewType z )
{
   // copy part of y_i to the YL buffer
   // The upper (m+1)x(m+1) submatrix of Y is duplicated in the YL buffer,
   // which resides on host and is broadcasted from rank 0 to all processes.
   HostView YL_i( &YL[ i * (restarting_max + 1) ], restarting_max + 1 );
   Algorithms::MultiDeviceMemoryOperations< Devices::Host, DeviceType >::copy( YL_i.getData(), Traits::getLocalView( y_i ).getData(), YL_i.getSize() );
   // no-op if the problem is not distributed
   MPI::Bcast( YL_i.getData(), YL_i.getSize(), 0, Traits::getCommunicator( *this->matrix ) );

   // NOTE: aux = t_i * (y_i, z) = 1  since  t_i = 2 / ||y_i||^2  and
   //       (y_i, z) = ||z_trunc||^2 + |z_i| ||z_trunc|| = ||y_i||^2 / 2
//   const RealType aux = T[ i + i * (restarting_max + 1) ] * (y_i, z);
   constexpr RealType aux = 1.0;
   if( localOffset == 0 ) {
      if( std::is_same< DeviceType, Devices::Host >::value ) {
         for( int k = 0; k <= i; k++ )
            out[ k ] = z[ k ] - y_i[ k ] * aux;
      }
      if( std::is_same< DeviceType, Devices::Cuda >::value ) {
         RealType host_z[ i + 1 ];
         Algorithms::MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::copy( host_z, Traits::getConstLocalView( z ).getData(), i + 1 );
         for( int k = 0; k <= i; k++ )
            out[ k ] = host_z[ k ] - YL_i[ k ] * aux;
      }
   }

   // no-op if the problem is not distributed
   MPI::Bcast( out.getData(), i + 1, 0, Traits::getCommunicator( *this->matrix ) );
}

template< typename Matrix >
void
GMRES< Matrix >::
hauseholder_cwy( VectorViewType v,
                 const int i )
{
   // aux = Y_i^T * e_i
   RealType aux[ i + 1 ];
   // the upper (m+1)x(m+1) submatrix of Y is duplicated on host
   // (faster access than from the device and it is broadcasted to all processes)
   for( int k = 0; k <= i; k++ )
      aux[ k ] = YL[ i + k * (restarting_max + 1) ];

   // aux = T_i * aux
   // Note that T_i is upper triangular, so we can overwrite the aux vector with the result in place
   for( int k = 0; k <= i; k++ ) {
      RealType aux2 = 0.0;
      for( int j = k; j <= i; j++ )
         aux2 += T[ k + j * (restarting_max + 1) ] * aux[ j ];
      aux[ k ] = aux2;
   }

   // v = e_i - Y_i * aux
   Matrices::MatrixOperations< DeviceType >::
      gemv( size, (IndexType) i + 1,
            (RealType) -1.0, Y.getData(), ldSize, aux,
            (RealType) 0.0, Traits::getLocalView( v ).getData() );
   if( localOffset == 0 )
      v.setElement( i, 1.0 + v.getElement( i ) );
}

template< typename Matrix >
void
GMRES< Matrix >::
hauseholder_cwy_transposed( VectorViewType z,
                            const int i,
                            ConstVectorViewType w )
{
   // aux = Y_i^T * w
   RealType aux[ i + 1 ];
   const RealType* _Y = Y.getData();
   const RealType* _w = Traits::getConstLocalView( w ).getData();
   const IndexType ldSize = this->ldSize;
   auto fetch = [_Y, _w, ldSize] __cuda_callable__ ( IndexType idx, int k ) { return _Y[ idx + k * ldSize ] * _w[ idx ]; };
   Algorithms::Multireduction< DeviceType >::reduce
            ( (RealType) 0,
              fetch,
              std::plus<>{},
              size,
              i + 1,
              aux );
   // no-op if the problem is not distributed
   MPI::Allreduce( aux, i + 1, MPI_SUM, Traits::getCommunicator( *this->matrix ) );

   // aux = T_i^T * aux
   // Note that T_i^T is lower triangular, so we can overwrite the aux vector with the result in place
   for( int k = i; k >= 0; k-- ) {
      RealType aux2 = 0.0;
      for( int j = 0; j <= k; j++ )
         aux2 += T[ j + k * (restarting_max + 1) ] * aux[ j ];
      aux[ k ] = aux2;
   }

   // z = w - Y_i * aux
   z = w;
   Matrices::MatrixOperations< DeviceType >::
      gemv( size, (IndexType) i + 1,
            (RealType) -1.0, Y.getData(), ldSize, aux,
            (RealType) 1.0, Traits::getLocalView( z ).getData() );
}

template< typename Matrix >
   template< typename Vector >
void
GMRES< Matrix >::
update( const int k,
        const int m,
        const HostVector& H,
        const HostVector& s,
        DeviceVector& V,
        Vector& x )
{
   RealType y[ m + 1 ];

   for( int i = 0; i <= m ; i ++ )
      y[ i ] = s[ i ];

   // Backsolve:
   for( int i = k; i >= 0; i--) {
      if( H[ i + i * ( m + 1 ) ] == 0 ) {
//         for( int _i = 0; _i <= i; _i++ ) {
//             for( int _j = 0; _j < i; _j++ )
//                std::cout << H[ _i + _j * (m+1) ] << "  ";
//            std::cout << std::endl;
//         }
         std::cerr << "H.norm = " << lpNorm( H, 2.0 ) << std::endl;
         std::cerr << "s = " << s << std::endl;
         std::cerr << "k = " << k << ", m = " << m << std::endl;
         throw 1;
      }
      y[ i ] /= H[ i + i * ( m + 1 ) ];
      for( int j = i - 1; j >= 0; j--)
         y[ j ] -= H[ j + i * ( m + 1 ) ] * y[ i ];
   }

   if( variant != Variant::CWY ) {
      // x = V * y + x
      Matrices::MatrixOperations< DeviceType >::
         gemv( size, (IndexType) k + 1,
               (RealType) 1.0, V.getData(), ldSize, y,
               (RealType) 1.0, Traits::getLocalView( x ).getData() );
   }
   else {
      // The vectors v_i are not stored, they can be reconstructed as P_0...P_j * e_j.
      // Hence, for j = 0, ... k:  x += y_j P_0...P_j e_j,
      // or equivalently: x += \sum_0^k y_j e_j - Y_k T_k \sum_0^k y_j Y_j^T e_j

      RealType aux[ k + 1 ];
      for( int j = 0; j <= k; j++ )
         aux[ j ] = 0;

      for( int j = 0; j <= k; j++ ) {
         // aux += y_j * Y_j^T * e_j
         // the upper (m+1)x(m+1) submatrix of Y is duplicated on host
         // (faster access than from the device and it is broadcasted to all processes)
         for( int i = 0; i <= j; i++ )
            aux[ i ] += y[ j ] * YL[ j + i * (restarting_max + 1) ];
      }

      // aux = T_{k+1} * aux
      // Note that T_{k+1} is upper triangular, so we can overwrite the aux vector with the result in place
      for( int i = 0; i <= k; i++ ) {
         RealType aux2 = 0.0;
         for( int j = i; j <= k; j++ )
            aux2 += T[ i + j * (restarting_max + 1) ] * aux[ j ];
         aux[ i ] = aux2;
      }

      // x -= Y_{k+1} * aux
      Matrices::MatrixOperations< DeviceType >::
         gemv( size, (IndexType) k + 1,
               (RealType) -1.0, Y.getData(), ldSize, aux,
               (RealType) 1.0, Traits::getLocalView( x ).getData() );

      // x += y
      if( localOffset == 0 )
         for( int j = 0; j <= k; j++ )
            x.setElement( j, x.getElement( j ) + y[ j ] );
   }
}

template< typename Matrix >
void
GMRES< Matrix >::
generatePlaneRotation( RealType& dx,
                       RealType& dy,
                       RealType& cs,
                       RealType& sn )
{
   if( dy == 0.0 ) {
      cs = 1.0;
      sn = 0.0;
   }
   else if( std::fabs( dy ) > std::fabs( dx ) ) {
      const RealType temp = dx / dy;
      sn = 1.0 / std::sqrt( 1.0 + temp * temp );
      cs = temp * sn;
   }
   else {
      const RealType temp = dy / dx;
      cs = 1.0 / std::sqrt( 1.0 + temp * temp );
      sn = temp * cs;
   }
}

template< typename Matrix >
void
GMRES< Matrix >::
applyPlaneRotation( RealType& dx,
                    RealType& dy,
                    RealType& cs,
                    RealType& sn )
{
   const RealType temp = cs * dx + sn * dy;
   dy = cs * dy - sn * dx;
   dx = temp;
}

template< typename Matrix >
void
GMRES< Matrix >::
apply_givens_rotations( int i, int m )
{
   for( int k = 0; k < i; k++ )
      applyPlaneRotation( H[ k     + i * (m + 1) ],
                          H[ k + 1 + i * (m + 1) ],
                          cs[ k ],
                          sn[ k ] );

   if( H[ i + 1 + i * (m + 1) ] != 0.0 ) {
      generatePlaneRotation( H[ i     + i * (m + 1) ],
                             H[ i + 1 + i * (m + 1) ],
                             cs[ i ],
                             sn[ i ]);
      applyPlaneRotation( H[ i     + i * (m + 1) ],
                          H[ i + 1 + i * (m + 1) ],
                          cs[ i ],
                          sn[ i ]);
      applyPlaneRotation( s[ i     ],
                          s[ i + 1 ],
                          cs[ i ],
                          sn[ i ] );
   }
}

template< typename Matrix >
void
GMRES< Matrix >::
setSize( const VectorViewType& x )
{
   this->size = Traits::getLocalView( x ).getSize();
   if( std::is_same< DeviceType, Devices::Cuda >::value )
      // align each column to 256 bytes - optimal for CUDA
      ldSize = roundToMultiple( size, 256 / sizeof( RealType ) );
   else
      // on the host, we add 1 to disrupt the cache false-sharing pattern
      ldSize = roundToMultiple( size, 256 / sizeof( RealType ) ) + 1;
   localOffset = getLocalOffset( *this->matrix );

   const int m = restarting_max;
   r.setLike( x );
   w.setLike( x );
   _M_tmp.setLike( x );
   cs.setSize( m + 1 );
   sn.setSize( m + 1 );
   H.setSize( ( m + 1 ) * m );
   s.setSize( m + 1 );

   // CWY-specific storage
   if( variant == Variant::CWY ) {
      z.setLike( x );
      Y.setSize( ldSize * ( m + 1 ) );
      T.setSize( (m + 1) * (m + 1) );
      YL.setSize( (m + 1) * (m + 1) );
      // vectors v_i are not stored, they can be reconstructed in the update() method
      V.setLike( x );
   }
   else {
      V.setSize( ldSize * ( m + 1 ) );
   }
}

} // namespace Linear
} // namespace Solvers
} // namespace noa::TNL
