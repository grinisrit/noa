// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <stdexcept>
#include <tuple>
#include <random>
#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/NDArray.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Math.h>

namespace noa::TNL {

// TODO: move this somewhere else
// like numpy.linspace in Python
template< typename T, typename I >
auto
linspace( T a, T b, I n )
{
   TNL::Containers::Array< T, TNL::Devices::Sequential, I > space( n );

   // avoid division by zero
   if( n == 1 ) {
      space[ 0 ] = a;
      return space;
   }

   const T dx = ( b - a ) / ( n - 1 );
   for( I i = 0; i < n; i++ )
      space[ i ] = a + dx * i;
   return space;
}

/**
 * \brief Namespace for Computational Fluid Dynamics utilities in TNL.
 */
namespace CFD {

/**
 * \brief Synthetic turbulence generator based on the code from [1, 2].  Usable
 * for initial as well as inlet boundary conditions.
 *
 * [1] http://www.tfd.chalmers.se/~lada/projects/inlet-boundary-conditions/proright.html
 * [2] Lars Davidson - Fluid mechanics, turbulent flow and turbulence modeling
 *     http://www.tfd.chalmers.se/~lada/postscript_files/solids-and-fluids_turbulent-flow_turbulence-modelling.pdf
 */
template< typename Real, typename Device, typename Index >
struct TurbulenceGenerator
{
   using xyz_permutation = std::index_sequence< 2, 1, 0 >;      // z, y, x
   using xyz_overlaps = std::index_sequence< 0, 0, 0 >;         // x, y, z
   using txyz_permutation = std::index_sequence< 0, 3, 2, 1 >;  // t, z, y, x
   using txyz_overlaps = std::index_sequence< 0, 0, 0, 0 >;     // t, x, y, z

   using Vector = TNL::Containers::Vector< Real, Device, unsigned >;
   using Array1D = TNL::Containers::Array< Real, Device, Index >;
   using Array3D = TNL::Containers::NDArray< Real,
                                             TNL::Containers::SizesHolder< Index, 0, 0, 0 >,  // t, x, y, z
                                             xyz_permutation,
                                             Device,
                                             Index,
                                             xyz_overlaps >;
   using Array4D = TNL::Containers::NDArray< Real,
                                             TNL::Containers::SizesHolder< Index, 0, 0, 0, 0 >,  // t, x, y, z
                                             txyz_permutation,
                                             Device,
                                             Index,
                                             txyz_overlaps >;

   // constant defined by eq. (27.6) in [2]
   static constexpr Real c_E = 1.452762113;

   // number of modes
   unsigned nmodes = 3000;

   // ratio of the k_e and k_min wave numbers
   Real smallestWaveNumberFactor = 5;

   // kinematic viscosity
   Real viscosity = 1.5e-5;

   // turbulent kinetic energy
   Real kineticEnergy = 1e-2;

   // integral length scale of the turbulence: assuming a boundary layer
   // thickness of delta=0.1, we set the length scale as one tenth of delta
   Real lengthScale = 0.1 * 0.1;

   // pseudo-random number generator
   std::mt19937 rng;

   TurbulenceGenerator( unsigned seed = std::random_device()() )
   {
      // seed the random number generator with given seed
      rng.seed( seed );
   }

   enum class TimeCorrelationMethod
   {
      // no correlation between time levels
      none,

      // initial time level is not correlated,
      // other time levels are correlated to the previous time level
      initialNone,

      // initial time level is correlated with zero (v(t_0 - \Delta t) = 0),
      // other time levels are correlated to the previous time level
      initialZero,

      // initial time level is correlated with the last time level of the previous series (v(t_0 - \Delta t) = v(t_n)),
      // other time levels are correlated to the previous time level
      initialWraparound,
   };

   template< typename Array1, typename Array2, typename Array3 >
   void
   generateFluctuations( Array1& u,          // x-component of velocity fluctuations vector
                         Array2& v,          // y-component of velocity fluctuations vector
                         Array3& w,          // z-component of velocity fluctuations vector
                         const Array1D& xc,  // x-coordinates of cell centers
                         const Array1D& yc,  // y-coordinates of cell centers
                         const Array1D& zc,  // z-coordinates of cell centers
                         Real minSpaceStep,
                         Real timeStep,
                         Real timeScale,  // integral time scale of the turbulence
                         TimeCorrelationMethod timeCorrelation = TimeCorrelationMethod::none )
   {
      static_assert( Array1::getDimension() == 4, "u must be a 4D array (components t,x,y,z)" );
      static_assert( Array2::getDimension() == 4, "v must be a 4D array (components t,x,y,z)" );
      static_assert( Array3::getDimension() == 4, "w must be a 4D array (components t,x,y,z)" );

      // check array sizes
      if( u.getSizes() != v.getSizes() || u.getSizes() != w.getSizes() )
         throw std::invalid_argument( "arrays u,v,w must have equal sizes" );
      if( xc.getSize() != u.template getSize< 1 >() )
         throw std::invalid_argument( "the size of the xc array does not match the x-size of the velocity arrays" );
      if( yc.getSize() != u.template getSize< 2 >() )
         throw std::invalid_argument( "the size of the yc array does not match the y-size of the velocity arrays" );
      if( zc.getSize() != u.template getSize< 3 >() )
         throw std::invalid_argument( "the size of the zc array does not match the z-size of the velocity arrays" );

      // number of time steps
      const unsigned ntimes = u.template getSize< 0 >();

      // turbulent velocity scale (RMS)
      const Real velocityScale = TNL::sqrt( 2.0 / 3.0 * kineticEnergy );

      // dissipation rate
      const Real dissipationRate = TNL::pow( kineticEnergy, 1.5 ) / lengthScale;

      // highest wave number
      const Real k_max = 2 * TNL::pi / minSpaceStep;

      // k_e (related to peak energy wave number)
      const Real k_e = 9 * TNL::pi * c_E / ( 55 * lengthScale );

      // wave number used in the viscous expression (high wave numbers) in the von Karman spectrum
      const Real k_eta = TNL::pow( dissipationRate / TNL::pow( viscosity, 3 ), 0.25 );

      // smallest wave number
      const Real k_min = k_e / smallestWaveNumberFactor;

      // wave number step
      const Real dk = ( k_max - k_min ) / nmodes;

      std::cout << "Turbulence integral length scale: " << lengthScale << std::endl;
      std::cout << "Minimal wave number: " << k_min << std::endl;
      std::cout << "Maximal wave number: " << k_max << std::endl;
      std::cout << "Wave number step: " << dk << std::endl;
      std::cout << "Number of modes: " << nmodes << std::endl;

      // wave numbers at cell centers
      const Vector k = Vector( linspace( k_min + dk / 2, k_max - dk / 2, nmodes ) );

      // random angles
      Vector phi( nmodes );
      Vector psi( nmodes );
      Vector alpha( nmodes );
      Vector theta( nmodes );

      // wave number vectors
      Vector kx;
      Vector ky;
      Vector kz;
      Vector sx;
      Vector sy;
      Vector sz;

      // time correlation factors
      const Real corr_a = TNL::exp( -timeStep / timeScale );
      const Real corr_b = TNL::sqrt( 1 - corr_a * corr_a );

      std::cout << "Number of time steps: " << ntimes << std::endl;
      std::cout << "Simulation time step: " << timeStep << std::endl;
      std::cout << "Turbulence integral time scale: " << timeScale << std::endl;
      std::cout << "Time correlation factors: a = " << corr_a << ", b = " << corr_b << std::endl;

      for( unsigned t = 0; t < ntimes; t++ ) {
         // generate random angles
         generateAngles( rng, phi, psi, alpha, theta );

         // wave number vector from random angles
         kx = TNL::sin( theta ) * TNL::cos( phi ) * k;
         ky = TNL::sin( theta ) * TNL::sin( phi ) * k;
         kz = TNL::cos( theta ) * k;

         // sigma (s=sigma) from random angles. sigma is the unit direction which
         // gives the direction of the synthetic velocity vector (u, v, w)
         sx = TNL::cos( phi ) * TNL::cos( theta ) * TNL::cos( alpha ) - TNL::sin( phi ) * TNL::sin( alpha );
         sy = TNL::sin( phi ) * TNL::cos( theta ) * TNL::cos( alpha ) + TNL::cos( phi ) * TNL::sin( alpha );
         sz = -TNL::sin( theta ) * TNL::cos( alpha );

         // extract views for capturing in the lambda function
         const auto _k = k.getConstView();
         auto _sx = sx.getView();
         auto _sy = sy.getView();
         auto _sz = sz.getView();

         // loop over all wave numbers
         TNL::Algorithms::ParallelFor< Device >::exec(
            unsigned( 0 ),
            nmodes,
            [ velocityScale, dk, k_e, k_eta, _k, _sx, _sy, _sz ] __cuda_callable__( unsigned m ) mutable
            {
               const Real kappa = _k[ m ];

               // von Karman spectrum
               const Real E =
                  ( c_E / k_e * TNL::pow( kappa / k_e, 4 ) / TNL::pow( 1 + TNL::pow( kappa / k_e, 2 ), Real( 17.0 / 6.0 ) )
                    * TNL::exp( -2 * TNL::pow( kappa / k_eta, 2 ) ) );

               const Real utn = velocityScale * TNL::sqrt( E * dk );

               // multiply sigma with the spectrum amplitude (see eq (27.8) in [2])
               _sx[ m ] *= 2 * utn;
               _sy[ m ] *= 2 * utn;
               _sz[ m ] *= 2 * utn;
            } );

         // extract further variables and views for capturing in the lambda function
         const auto nmodes = this->nmodes;
         const auto _psi = psi.getConstView();
         const auto _xc = xc.getConstView();
         const auto _yc = yc.getConstView();
         const auto _zc = zc.getConstView();
         const auto _kx = kx.getConstView();
         const auto _ky = ky.getConstView();
         const auto _kz = kz.getConstView();
         auto _u = u.getView();
         auto _v = v.getView();
         auto _w = w.getView();

         // loop through the grid
         const auto subarray = u.template getSubarrayView< 1, 2, 3 >( t, 0, 0, 0 );
         subarray.forAll(
            [ = ] __cuda_callable__( Index x, Index y, Index z ) mutable
            {
               // initialize velocity fluctuations to zero
               Real ut = 0;
               Real vt = 0;
               Real wt = 0;

               // loop over all wave numbers
               for( unsigned m = 0; m < nmodes; m++ ) {
                  const Real beta = _kx[ m ] * _xc[ x ] + _ky[ m ] * _yc[ y ] + _kz[ m ] * _zc[ z ] + _psi[ m ];
                  const Real cosBeta = TNL::cos( beta );

                  // update synthetic velocity field - see eq (27.8) in [2]
                  ut += cosBeta * _sx[ m ];
                  vt += cosBeta * _sy[ m ];
                  wt += cosBeta * _sz[ m ];
               }

               // introduce time correlation
               if( t == 0 ) {
                  switch( timeCorrelation ) {
                     case TimeCorrelationMethod::initialZero:
                        ut = corr_a * 0 + corr_b * ut;
                        vt = corr_a * 0 + corr_b * vt;
                        wt = corr_a * 0 + corr_b * wt;
                        break;
                     case TimeCorrelationMethod::initialWraparound:
                        ut = corr_a * _u( ntimes - 1, x, y, z ) + corr_b * ut;
                        vt = corr_a * _v( ntimes - 1, x, y, z ) + corr_b * vt;
                        wt = corr_a * _w( ntimes - 1, x, y, z ) + corr_b * wt;
                        break;
                     default:
                        break;
                  }
               }
               else if( timeCorrelation != TimeCorrelationMethod::none ) {
                  ut = corr_a * _u( t - 1, x, y, z ) + corr_b * ut;
                  vt = corr_a * _v( t - 1, x, y, z ) + corr_b * vt;
                  wt = corr_a * _w( t - 1, x, y, z ) + corr_b * wt;
               }

               // store the fluctuations in the output arrays
               _u( t, x, y, z ) = ut;
               _v( t, x, y, z ) = vt;
               _w( t, x, y, z ) = wt;
            } );
      }
   }

   std::tuple< Array4D, Array4D, Array4D >
   getFluctuations( const Array1D& xc,  // x-coordinates of cell centers
                    const Array1D& yc,  // y-coordinates of cell centers
                    const Array1D& zc,  // z-coordinates of cell centers
                    Real minSpaceStep,
                    unsigned ntimes = 1,  // number of time steps
                    Real timeStep = 1,
                    Real timeScale = 1,  // integral time scale of the turbulence
                    TimeCorrelationMethod timeCorrelation = TimeCorrelationMethod::none )
   {
      if( timeCorrelation == TimeCorrelationMethod::initialWraparound )
         throw std::logic_error( "TimeCorrelationMethod::initialWraparound cannot be used in the getFluctuations method, "
                                 "it would mean using uninitialized values." );

      // grid sizes
      const Index Nx = xc.getSize();
      const Index Ny = yc.getSize();
      const Index Nz = zc.getSize();

      // output arrays for velocity fluctuations
      Array4D u;
      Array4D v;
      Array4D w;
      u.setSizes( ntimes, Nx, Ny, Nz );
      v.setSizes( ntimes, Nx, Ny, Nz );
      w.setSizes( ntimes, Nx, Ny, Nz );

      generateFluctuations( u, v, w, xc, yc, zc, minSpaceStep, timeStep, timeScale, timeCorrelation );

      return std::make_tuple( u, v, w );
   }

private:
   // specialization for host devices
   template< typename RNG, typename Array >
   static std::enable_if_t< ! std::is_same< typename Array::DeviceType, TNL::Devices::Cuda >::value >
   generateAngles( RNG&& rng, Array& phi, Array& psi, Array& alpha, Array& theta )
   {
      using Value = typename Array::ValueType;

      std::uniform_real_distribution< Value > dis_phi( 0, 2 * TNL::pi );
      std::uniform_real_distribution< Value > dis_psi( 0, 2 * TNL::pi );
      std::uniform_real_distribution< Value > dis_alpha( 0, 2 * TNL::pi );
      std::uniform_real_distribution< Value > dis_ang( 0, 1 );

      // TODO: assert that all arrays have equal sizes

      for( unsigned m = 0; m < phi.getSize(); m++ ) {
         phi[ m ] = dis_phi( rng );
         psi[ m ] = dis_psi( rng );
         alpha[ m ] = dis_alpha( rng );
         const Value ang = dis_ang( rng );
         theta[ m ] = std::acos( Value( 1.0 ) - ang / Value( 0.5 ) );
      }
   }

   // specialization for CUDA
   template< typename RNG, typename Array >
   static std::enable_if_t< std::is_same< typename Array::DeviceType, TNL::Devices::Cuda >::value >
   generateAngles( RNG&& rng, Array& phi, Array& psi, Array& alpha, Array& theta )
   {
      // create auxiliary host arrays
      using HostArray = typename Array::template Self< typename Array::ValueType, TNL::Devices::Sequential >;
      HostArray _phi( phi.getSize() );
      HostArray _psi( psi.getSize() );
      HostArray _alpha( alpha.getSize() );
      HostArray _theta( theta.getSize() );

      // generate on the host
      generateAngles( std::forward< RNG >( rng ), _phi, _psi, _alpha, _theta );

      // copy results to the device
      phi = _phi;
      psi = _psi;
      alpha = _alpha;
      theta = _theta;
   }
};

}  // namespace CFD
}  // namespace noa::TNL
