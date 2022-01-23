// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Operators {   

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >,
                 PressureGradient > :: LaxFridrichs()
: regularizeEps( 0.0 ),
  viscosityCoefficient( 1.0 ),
  mesh( 0 ),
  pressureGradient( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  >::bindMesh( const MeshType& mesh )
{
   this->mesh = &mesh;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
const typename LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  >::MeshType&
   LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  >::getMesh() const
{
   return * this->mesh;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: setRegularization( const RealType& epsilon )
{
   this->regularizeEps = epsilon;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: setViscosityCoefficient( const RealType& v )
{
   this->viscosityCoefficient = v;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: setRho( Vector& rho )
{
   this->rho.bind( rho );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: setRhoU1( Vector& rho_u1 )
{
   this->rho_u1. bind( rho_u1 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: setRhoU2( Vector& rho_u2 )
{
   this->rho_u2. bind( rho_u2 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: setE( Vector& e )
{
   this->e.bind( e );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: setPressureGradient( Vector& grad_p )
{
   this->pressureGradient = &grad_p;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: getExplicitUpdate( const IndexType centralVolume,
                                                                                                              RealType& rho_t,
                                                                                                              RealType& rho_u1_t,
                                                                                                              RealType& rho_u2_t,
                                                                                                              const RealType& tau ) const
{
   TNL_ASSERT_TRUE( mesh, "No mesh has been binded with the Lax-Fridrichs scheme." );
   TNL_ASSERT_TRUE( pressureGradient, "No pressure gradient was set in the the Lax-Fridrichs scheme." )

   const IndexType& c = centralVolume;
   const IndexType e = this->mesh -> getElementNeighbor( centralVolume,  1,  0 );
   const IndexType w = this->mesh -> getElementNeighbor( centralVolume, -1,  0 );
   const IndexType n = this->mesh -> getElementNeighbor( centralVolume,  0,  1 );
   const IndexType s = this->mesh -> getElementNeighbor( centralVolume,  0, -1 );

   const RealType u1_e = rho_u1[ e ] / regularize( rho[ e ] );
   const RealType u1_w = rho_u1[ w ] / regularize( rho[ w ] );
   const RealType u2_n = rho_u2[ n ] / regularize( rho[ n ] );
   const RealType u2_s = rho_u2[ s ] / regularize( rho[ s ] );
   const RealType u1_c = rho_u1[ c ] / regularize( rho[ c ] );
   const RealType u1_n = rho_u1[ n ] / regularize( rho[ n ] );
   const RealType u1_s = rho_u1[ s ] / regularize( rho[ s ] );
   const RealType u2_c = rho_u2[ c ] / regularize( rho[ c ] );
   const RealType u2_e = rho_u2[ e ] / regularize( rho[ e ] );
   const RealType u2_w = rho_u2[ w ] / regularize( rho[ w ] );

   /****
    * Get the central volume and its neighbors (east, north, west, south) coordinates
    */
   CoordinatesType c_coordinates, e_coordinates, n_coordinates, w_coordinates, s_coordinates;
   this->mesh -> getElementCoordinates( c, c_coordinates );
   e_coordinates = n_coordinates = w_coordinates = s_coordinates = c_coordinates;
   e_coordinates. x() ++;
   w_coordinates. x() --;
   n_coordinates. y() ++;
   s_coordinates. y() --;

   /****
    * Get the volumes measure
    */
   const RealType mu_D_c = this->mesh -> getElementMeasure( c_coordinates );
   const RealType mu_D_e = this->mesh -> getElementMeasure( e_coordinates );
   const RealType mu_D_n = this->mesh -> getElementMeasure( n_coordinates );
   const RealType mu_D_w = this->mesh -> getElementMeasure( w_coordinates );
   const RealType mu_D_s = this->mesh -> getElementMeasure( s_coordinates );

   /****
    * Get the edge normals
    */
   PointType e_normal, w_normal, n_normal, s_normal;
   this->mesh -> template getEdgeNormal<  1,  0 >( c_coordinates, e_normal );
   this->mesh -> template getEdgeNormal< -1,  0 >( c_coordinates, w_normal );
   this->mesh -> template getEdgeNormal<  0,  1 >( c_coordinates, n_normal );
   this->mesh -> template getEdgeNormal<  0, -1 >( c_coordinates, s_normal );

   /****
    * Compute the fluxes
    */
   const RealType rho_f_e = 0.5 * ( rho_u1[ c ] + rho_u1[ e ] );
   const RealType rho_f_w = 0.5 * ( rho_u1[ c ] + rho_u1[ w ] );
   const RealType rho_f_n = 0.5 * ( rho_u1[ c ] + rho_u1[ n ] );
   const RealType rho_f_s = 0.5 * ( rho_u1[ c ] + rho_u1[ s ] );
   const RealType rho_g_e = 0.5 * ( rho_u2[ c ] + rho_u2[ e ] );
   const RealType rho_g_w = 0.5 * ( rho_u2[ c ] + rho_u2[ w ] );
   const RealType rho_g_n = 0.5 * ( rho_u2[ c ] + rho_u2[ n ] );
   const RealType rho_g_s = 0.5 * ( rho_u2[ c ] + rho_u2[ s ] );

   const RealType rho_u1_f_e = 0.5 * ( rho_u1[ c ] * u1_c + rho_u1[ e ] * u1_e );
   const RealType rho_u1_f_w = 0.5 * ( rho_u1[ c ] * u1_c + rho_u1[ w ] * u1_w );
   const RealType rho_u1_f_n = 0.5 * ( rho_u1[ c ] * u1_c + rho_u1[ n ] * u1_n );
   const RealType rho_u1_f_s = 0.5 * ( rho_u1[ c ] * u1_c + rho_u1[ s ] * u1_s );
   const RealType rho_u1_g_e = 0.5 * ( rho_u1[ c ] * u2_c + rho_u1[ e ] * u2_e );
   const RealType rho_u1_g_w = 0.5 * ( rho_u1[ c ] * u2_c + rho_u1[ w ] * u2_w );
   const RealType rho_u1_g_n = 0.5 * ( rho_u1[ c ] * u2_c + rho_u1[ n ] * u2_n );
   const RealType rho_u1_g_s = 0.5 * ( rho_u1[ c ] * u2_c + rho_u1[ s ] * u2_s );

   const RealType rho_u2_f_e = 0.5 * ( rho_u2[ c ] * u1_c + rho_u2[ e ] * u1_e );
   const RealType rho_u2_f_w = 0.5 * ( rho_u2[ c ] * u1_c + rho_u2[ w ] * u1_w );
   const RealType rho_u2_f_n = 0.5 * ( rho_u2[ c ] * u1_c + rho_u2[ n ] * u1_n );
   const RealType rho_u2_f_s = 0.5 * ( rho_u2[ c ] * u1_c + rho_u2[ s ] * u1_s );
   const RealType rho_u2_g_e = 0.5 * ( rho_u2[ c ] * u2_c + rho_u2[ e ] * u2_e );
   const RealType rho_u2_g_w = 0.5 * ( rho_u2[ c ] * u2_c + rho_u2[ w ] * u2_w );
   const RealType rho_u2_g_n = 0.5 * ( rho_u2[ c ] * u2_c + rho_u2[ n ] * u2_n );
   const RealType rho_u2_g_s = 0.5 * ( rho_u2[ c ] * u2_c + rho_u2[ s ] * u2_s );

   /****
    * Compute the pressure gradient
    */
   PointType grad_p;
   pressureGradient -> getGradient( c, grad_p );

   /****
    * rho_t + ( rho u_1 )_x + ( rho u_2 )_y =  0
    */
   rho_t = - 1.0 / mu_D_c * ( rho_f_e * e_normal. x() + rho_g_e * e_normal. y() +
                              rho_f_n * n_normal. x() + rho_g_n * n_normal. y() +
                              rho_f_w * w_normal. x() + rho_g_w * w_normal. y() +
                              rho_f_s * s_normal. x() + rho_g_s * s_normal. y() )
           + this->viscosityCoefficient * 1.0 / ( 8.0 * tau ) * mu_D_c *
                            ( ( mu_D_c + mu_D_e ) * ( rho[ e ] - rho[ c ] ) +
                              ( mu_D_c + mu_D_n ) * ( rho[ n ] - rho[ c ] ) +
                              ( mu_D_c + mu_D_w ) * ( rho[ w ] - rho[ c ] ) +
                              ( mu_D_c + mu_D_s ) * ( rho[ s ] - rho[ c ] ) );

   /****
    * ( rho * u1 )_t + ( rho * u1 * u1 )_x + ( rho * u1 * u2 )_y - p_x =  0
    */
   rho_u1_t = - 1.0 / mu_D_c * ( rho_u1_f_e * e_normal. x() + rho_u1_g_e * e_normal. y() +
                                 rho_u1_f_n * n_normal. x() + rho_u1_g_n * n_normal. y() +
                                 rho_u1_f_w * w_normal. x() + rho_u1_g_w * w_normal. y() +
                                 rho_u1_f_s * s_normal. x() + rho_u1_g_s * s_normal. y() )
              + this->viscosityCoefficient * 1.0 / ( 8.0 * tau ) * mu_D_c *
                               ( ( mu_D_c + mu_D_e ) * ( rho_u1[ e ] - rho_u1[ c ] ) +
                                 ( mu_D_c + mu_D_n ) * ( rho_u1[ n ] - rho_u1[ c ] ) +
                                 ( mu_D_c + mu_D_w ) * ( rho_u1[ w ] - rho_u1[ c ] ) +
                                 ( mu_D_c + mu_D_s ) * ( rho_u1[ s ] - rho_u1[ c ] ) )
                                 - grad_p. x();

   /****
    * ( rho * u2 )_t + ( rho * u2 * u1 )_x + ( rho * u2 * u2 )_y - p_y =  0
    */
   rho_u2_t = - 1.0 / mu_D_c * ( rho_u2_f_e * e_normal. x() + rho_u2_g_e * e_normal. y() +
                                 rho_u2_f_n * n_normal. x() + rho_u2_g_n * n_normal. y() +
                                 rho_u2_f_w * w_normal. x() + rho_u2_g_w * w_normal. y() +
                                 rho_u2_f_s * s_normal. x() + rho_u2_g_s * s_normal. y() )
              + this->viscosityCoefficient * 1.0 / ( 8.0 * tau ) * mu_D_c *
                               ( ( mu_D_c + mu_D_e ) * ( rho_u2[ e ] - rho_u2[ c ] ) +
                                 ( mu_D_c + mu_D_n ) * ( rho_u2[ n ] - rho_u2[ c ] ) +
                                 ( mu_D_c + mu_D_w ) * ( rho_u2[ w ] - rho_u2[ c ] ) +
                                 ( mu_D_c + mu_D_s ) * ( rho_u2[ s ] - rho_u2[ c ] ) )
                                 - grad_p. y();
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename > class GridGeometry >
Real LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient  > :: regularize( const Real& r ) const
{
   return r + ( ( r >= 0 ) - ( r < 0 ) ) * this->regularizeEps;
}

/****
 * Specialization for the grids with no deformations (Identical grid geometry)
 */

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >,
                 PressureGradient > :: LaxFridrichs()
: regularizeEps( 1.0e-5 ),
  viscosityCoefficient( 1.0 ),
  mesh( 0 ),
  pressureGradient( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: bindMesh( const MeshType& mesh )
{
   this->mesh = &mesh;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setRegularization( const RealType& epsilon )
{
   this->regularizeEps = epsilon;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setViscosityCoefficient( const RealType& v )
{
   this->viscosityCoefficient = v;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setRho( Vector& rho )
{
   this->rho. bind( rho );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setRhoU1( Vector& rho_u1 )
{
   this->rho_u1. bind( rho_u1 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setRhoU2( Vector& rho_u2 )
{
   this->rho_u2. bind( rho_u2 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setE( Vector& e )
{
   this->energy.bind( e );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setP( Vector& p )
{
   this->p.bind( p );
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
   template< typename Vector >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: setPressureGradient( Vector& grad_p )
{
   this->pressureGradient = &grad_p;
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: getExplicitUpdate( const IndexType centralVolume,
                                                                                                                          RealType& rho_t,
                                                                                                                          RealType& rho_u1_t,
                                                                                                                          RealType& rho_u2_t,
                                                                                                                          const RealType& tau ) const
{
   TNL_ASSERT_TRUE( mesh, "No mesh has been binded with the Lax-Fridrichs scheme." );
   TNL_ASSERT_TRUE( pressureGradient, "No pressure gradient was set in the the Lax-Fridrichs scheme." )

   const IndexType& xSize = this->mesh -> getDimensions(). x();
   const IndexType& ySize = this->mesh -> getDimensions(). y();
   const RealType hx = this->mesh -> getParametricStep(). x();
   const RealType hy = this->mesh -> getParametricStep(). y();

   const IndexType& c = centralVolume;
   const IndexType e = this->mesh -> getElementNeighbor( centralVolume,  1,  0 );
   const IndexType w = this->mesh -> getElementNeighbor( centralVolume, -1,  0 );
   const IndexType n = this->mesh -> getElementNeighbor( centralVolume,  0,  1 );
   const IndexType s = this->mesh -> getElementNeighbor( centralVolume,  0, -1 );

   /****
    * rho_t + ( rho u_1 )_x + ( rho u_2 )_y =  0
    */
   const RealType u1_e = rho_u1[ e ] / regularize( rho[ e ] );
   const RealType u1_w = rho_u1[ w ] / regularize( rho[ w ] );
   const RealType u2_n = rho_u2[ n ] / regularize( rho[ n ] );
   const RealType u2_s = rho_u2[ s ] / regularize( rho[ s ] );
   rho_t = this->viscosityCoefficient / tau * 0.25 * ( rho[ e ] + rho[ w ] + rho[ s ] + rho[ n ] - 4.0 * rho[ c ] )
               - ( rho[ e ] * u1_e - rho[ w ] * u1_w ) / ( 2.0 * hx )
               - ( rho[ n ] * u2_n - rho[ s ] * u2_s ) / ( 2.0 * hy );

   /****
    * Compute the pressure gradient
    */
   PointType grad_p;
   pressureGradient -> getGradient( c, grad_p );

   /****
    * ( rho * u1 )_t + ( rho * u1 * u1 )_x + ( rho * u1 * u2 )_y - p_x =  0
    */
   rho_u1_t = this->viscosityCoefficient / tau * 0.25 * ( rho_u1[ e ] + rho_u1[ w ] + rho_u1[ s ] + rho_u1[ n ] - 4.0 * rho_u1[ c ] )
                   - ( rho_u1[ e ] * u1_e - rho_u1[ w ] * u1_w ) / ( 2.0 * hx )
                   - ( rho_u1[ n ] * u2_n - rho_u1[ s ] * u2_s ) / ( 2.0 * hy )
                   - grad_p. x();
   /****
    * ( rho * u2 )_t + ( rho * u2 * u1 )_x + ( rho * u2 * u2 )_y - p_y =  0
    */
   rho_u2_t = this->viscosityCoefficient / tau * 0.25 * ( rho_u2[ e ] + rho_u2[ w ] + rho_u2[ s ] + rho_u2[ n ] - 4.0 * rho_u2[ c ] )
                   - ( rho_u2[ e ] * u1_e - rho_u2[ w ] * u1_w ) / ( 2.0 * hx )
                   - ( rho_u2[ n ] * u2_n - rho_u2[ s ] * u2_s ) / ( 2.0 * hy )
                   - grad_p. y();
}

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
void LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: getExplicitUpdate( const IndexType centralVolume,
                                                                                                                          RealType& rho_t,
                                                                                                                          RealType& rho_u1_t,
                                                                                                                          RealType& rho_u2_t,
                                                                                                                          RealType& e_t,
                                                                                                                          const RealType& tau ) const
{
   TNL_ASSERT_TRUE( mesh, "No mesh has been binded with the Lax-Fridrichs scheme." );
   TNL_ASSERT_TRUE( pressureGradient, "No pressure gradient was set in the the Lax-Fridrichs scheme." )

   const IndexType& xSize = this->mesh -> getDimensions(). x();
   const IndexType& ySize = this->mesh -> getDimensions(). y();
   const RealType hx = this->mesh -> getParametricStep(). x();
   const RealType hy = this->mesh -> getParametricStep(). y();

   const IndexType& c = centralVolume;
   const IndexType e = this->mesh -> getElementNeighbor( centralVolume,  1,  0 );
   const IndexType w = this->mesh -> getElementNeighbor( centralVolume, -1,  0 );
   const IndexType n = this->mesh -> getElementNeighbor( centralVolume,  0,  1 );
   const IndexType s = this->mesh -> getElementNeighbor( centralVolume,  0, -1 );

   /****
    * rho_t + ( rho u_1 )_x + ( rho u_2 )_y =  0
    */
   const RealType u1_e = rho_u1[ e ] / regularize( rho[ e ] );
   const RealType u1_w = rho_u1[ w ] / regularize( rho[ w ] );
   const RealType u2_n = rho_u2[ n ] / regularize( rho[ n ] );
   const RealType u2_s = rho_u2[ s ] / regularize( rho[ s ] );
   rho_t = this->viscosityCoefficient / tau * 0.25 * ( rho[ e ] + rho[ w ] + rho[ s ] + rho[ n ] - 4.0 * rho[ c ] )
               - ( rho[ e ] * u1_e - rho[ w ] * u1_w ) / ( 2.0 * hx )
               - ( rho[ n ] * u2_n - rho[ s ] * u2_s ) / ( 2.0 * hy );

   /****
    * Compute the pressure gradient
    */
   PointType grad_p;
   pressureGradient -> getGradient( c, grad_p );

   /****
    * ( rho * u1 )_t + ( rho * u1 * u1 )_x + ( rho * u1 * u2 )_y - p_x =  0
    */
   rho_u1_t = this->viscosityCoefficient / tau * 0.25 * ( rho_u1[ e ] + rho_u1[ w ] + rho_u1[ s ] + rho_u1[ n ] - 4.0 * rho_u1[ c ] )
                   - ( rho_u1[ e ] * u1_e - rho_u1[ w ] * u1_w ) / ( 2.0 * hx )
                   - ( rho_u1[ n ] * u2_n - rho_u1[ s ] * u2_s ) / ( 2.0 * hy )
                   - grad_p. x();
   /****
    * ( rho * u2 )_t + ( rho * u2 * u1 )_x + ( rho * u2 * u2 )_y - p_y =  0
    */
   rho_u2_t = this->viscosityCoefficient / tau * 0.25 * ( rho_u2[ e ] + rho_u2[ w ] + rho_u2[ s ] + rho_u2[ n ] - 4.0 * rho_u2[ c ] )
                   - ( rho_u2[ e ] * u1_e - rho_u2[ w ] * u1_w ) / ( 2.0 * hx )
                   - ( rho_u2[ n ] * u2_n - rho_u2[ s ] * u2_s ) / ( 2.0 * hy )
                   - grad_p. y();

   /****
    * e_t + ( ( e + p ) * u )_x + ( ( e + p ) * v )_y = 0
    */
   e_t = this->viscosityCoefficient / tau * 0.25 * ( energy[ e ] + energy[ w ] + energy[ s ] + energy[ n ] - 4.0 * energy[ c ] )
              - ( ( energy[ e ] + p[ e ] ) * u1_e - ( energy[ w ] + p[ w ] ) * u1_w ) / ( 2.0 * hx )
              - ( ( energy[ n ] + p[ n ] ) * u2_n - ( energy[ s ] + p[ s ] ) * u2_s ) / ( 2.0 * hy );
}


template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient >
Real LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient  > :: regularize( const Real& r ) const
{
   return r + ( ( r >= 0 ) - ( r < 0 ) ) * this->regularizeEps;
}

} // namespace Operators
} // namespace noa::TNL
