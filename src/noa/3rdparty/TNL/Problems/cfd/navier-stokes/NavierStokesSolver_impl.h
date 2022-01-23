// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Solvers/cfd/navier-stokes/NavierStokesSolver.h>

namespace noa::TNL {

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::NavierStokesSolver()
: advection( 0 ),
  u1Viscosity( 0 ),
  u2Viscosity( 0 ),
  energyViscosity( 0 ),
  mu( 0.0 ),
  gravity( 0.0 ),
  R( 0.0 ),
  T( 0.0 )
{
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setAdvectionScheme( AdvectionSchemeType& advection )
{
   this->advection = &advection;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setDiffusionScheme( DiffusionSchemeType& u1Viscosity,
                                                                                                        DiffusionSchemeType& u2Viscosity,
                                                                                                        DiffusionSchemeType& energyViscosity )
{
   this->u1Viscosity = &u1Viscosity;
   this->u2Viscosity = &u2Viscosity;
   this->energyViscosity = &energyViscosity;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setBoundaryConditions( BoundaryConditionsType& boundaryConditions )
{
   this->boundaryConditions = &boundaryConditions;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setMesh( MeshType& mesh )
{
   this->mesh = &mesh;
   this->rho.setSize( this->mesh->getDofs() );
   this->u1.setSize(  this->mesh->getDofs() );
   this->u2.setSize(  this->mesh->getDofs() );
   this->u.setSize(  this->mesh->getDofs() );
   this->p.setSize(   this->mesh->getDofs() );
   this->energy.setSize(   this->mesh->getDofs() );
   this->rhsDofVector.setSize(  this->getDofs() );
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setMu( const RealType& mu )
{
   this->mu = mu;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::RealType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getMu() const
{
   return this->mu;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setR( const RealType& R )
{
   this->R = R;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::RealType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getR() const
{
   return this->R;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setT( const RealType& T )
{
   this->T = T;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::RealType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getT() const
{
   return this->T;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setHeatCapacityRatio( const RealType& gamma )
{
   this->gamma = gamma;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::RealType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getHeatCapacityRatio() const
{
   return this->gamma;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::setGravity( const RealType& gravity )
{
   this->gravity = gravity;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::RealType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getGravity() const
{
   return this->gravity;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getRho()
{
   return this->rho;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getRho() const
{
   return this->rho;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU1()
{
   return this->u1;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU1() const
{
   return this->u1;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU2()
{
   return this->u2;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU2() const
{
   return this->u2;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU()
{
   return this->u;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getU() const
{
   return this->u;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getPressure()
{
   return this->p;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getPressure() const
{
   return this->p;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getEnergy()
{
   return this->energy;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
const typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::VectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getEnergy() const
{
   return this->energy;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::IndexType
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getDofs() const
{
   return 4*this->mesh->getDofs();
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
void NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::bindDofVector( RealType* data )
{
   this->dofVector.bind( data, this->getDofs() );
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::DofVectorType&
   NavierStokesSolver< AdvectionScheme, DiffusionScheme, BoundaryConditions >::getDofVector()
{
   return this->dofVector;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
   template< typename Vector >
void NavierStokesSolver< AdvectionScheme,
                      DiffusionScheme,
                      BoundaryConditions > :: updatePhysicalQuantities( const Vector& dofs_rho,
                                                                        const Vector& dofs_rho_u1,
                                                                        const Vector& dofs_rho_u2,
                                                                        const Vector& dofs_e )
{
   if( DeviceType :: getDevice() == Devices::HostDevice )
   {
      const IndexType size = dofs_rho.getSize();

      #ifdef HAVE_OPENMP
      #pragma omp parallel for, if( Devices::Host::isOMPEnabled() )
      #endif
      for( IndexType c = 0; c < size; c++ )
         {
            this->rho[ c ] = dofs_rho[ c ];
            const RealType u1 = this->u1[ c ] = dofs_rho_u1[ c ] / dofs_rho[ c ];
            const RealType u2 = this->u2[ c ] = dofs_rho_u2[ c ] / dofs_rho[ c ];
            this->u[ c ] = ::sqrt( u1*u1 + u2*u2 );
            //this->p[ c ] = dofs_rho[ c ] * this->R * this->T;
            this->p[ c ] = ( this->gamma - 1.0 ) *
                           ( dofs_e[ c ] - 0.5 * this->rho[ c ] * ( this->u1[ c ] * this->u1[ c ] + this->u2[ c ] * this->u2[ c ] ) );
            this->energy[ c ] = dofs_e[ c ];
            //this->temperature[ c ] = this->p[ c ] / ( this->rho[ c ] * this->R );
         }
   }
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
   template< typename SolverVectorType >
void NavierStokesSolver< AdvectionScheme,
                      DiffusionScheme,
                      BoundaryConditions >::getExplicitUpdate( const RealType& time,
                                                            const RealType& tau,
                                                            SolverVectorType& u,
                                                            SolverVectorType& fu )
{
   TNL_ASSERT_TRUE( this->advection, "advection scheme was not set" );
   TNL_ASSERT_TRUE( this->u1Viscosity, "diffusion scheme was not set" );
   TNL_ASSERT_TRUE( this->u2Viscosity, "diffusion scheme was not set" );
   TNL_ASSERT_TRUE( this->boundaryConditions, "boundary conditions were not set" );

   SharedVector< RealType, DeviceType, IndexType > dofs_rho, dofs_rho_u1, dofs_rho_u2, dofs_e,
                                                      rho_t, rho_u1_t, rho_u2_t, e_t;

   const IndexType& dofs = this->mesh->getDofs();
   dofs_rho. bind( & u. getData()[ 0 ], dofs );
   dofs_rho_u1. bind( & u. getData()[ dofs ], dofs );
   dofs_rho_u2. bind( & u. getData()[ 2 * dofs ], dofs );
   dofs_e. bind( & u. getData()[ 3 * dofs ], dofs );

   this->advection->setRho( dofs_rho );
   this->advection->setRhoU1( dofs_rho_u1 );
   this->advection->setRhoU2( dofs_rho_u2 );
   this->advection->setE( dofs_e );
   this->advection->setP( this->p );
   this->energyViscosity->setFunction( this->energy );

   rho_t.bind( & fu. getData()[ 0 ], dofs );
   rho_u1_t.bind( & fu. getData()[ dofs ], dofs );
   rho_u2_t.bind( & fu. getData()[ 2 * dofs ], dofs );
   e_t.bind( & fu. getData()[ 3 * dofs ], dofs );

   updatePhysicalQuantities( dofs_rho, dofs_rho_u1, dofs_rho_u2, dofs_e );


   this->boundaryConditions->apply( time, tau, this->rho, this->u1, this->u2, this->energy );

   const IndexType& xSize = this->mesh->getDimensions().x();
   const IndexType& ySize = this->mesh->getDimensions().y();

   if( DeviceType::getDevice() == Devices::HostDevice )
   {
      for( IndexType i = 0; i < xSize; i ++ )
      {
         const IndexType c1 = mesh->getElementIndex( i, 0 );
         const IndexType c2 = mesh->getElementIndex( i, 1 );
         const IndexType c3 = mesh->getElementIndex( i, ySize - 1 );
         const IndexType c4 = mesh->getElementIndex( i, ySize - 2 );

         dofs_rho[ c1 ]    = this->rho[ c1 ];
         dofs_rho_u1[ c1 ] = this->rho[ c1 ] * this->u1[ c1 ];
         dofs_rho_u2[ c1 ] = this->rho[ c1 ] * this->u2[ c1 ];
         dofs_e[ c1 ] = this->energy[ c1 ];
         /*dofs_e[ c1 ]      = this->computeEnergy( this->rho[ c1 ],
                                                  this->temperature[ c1 ],
                                                  this->gamma,
                                                  this->u1[ c1 ],
                                                  this->u2[ c1 ] );*/

         dofs_rho[ c3 ]    = this->rho[ c3 ];
         dofs_rho_u1[ c3 ] = this->rho[ c3 ] * this->u1[ c3 ];
         dofs_rho_u2[ c3 ] = this->rho[ c3 ] * this->u2[ c3 ];
         dofs_e[ c3 ] = this->energy[ c3 ];
         /*dofs_e[ c3 ]      = this->computeEnergy( this->rho[ c3 ],
                                                  this->temperature[ c3 ],
                                                  this->gamma,
                                                  this->u1[ c3 ],
                                                  this->u2[ c3 ] );*/
      }
      for( IndexType j = 0; j < ySize; j ++ )
      {
         const IndexType c1 = mesh->getElementIndex( 0, j );
         const IndexType c2 = mesh->getElementIndex( 1, j );
         const IndexType c3 = mesh->getElementIndex( xSize - 1, j );
         const IndexType c4 = mesh->getElementIndex( xSize - 2, j );

         dofs_rho[ c1 ]    = this->rho[ c1 ];
         dofs_rho_u1[ c1 ] = this->rho[ c1 ] * this->u1[ c1 ];
         dofs_rho_u2[ c1 ] = this->rho[ c1 ] * this->u2[ c1 ];
         dofs_e[ c1 ] = this->energy[ c1 ];
         /*dofs_e[ c1 ]      = this->computeEnergy( this->rho[ c1 ],
                                                  this->temperature[ c1 ],
                                                  this->gamma,
                                                  this->u1[ c1 ],
                                                  this->u2[ c1 ] );*/


         dofs_rho[ c3 ]    = this->rho[ c3 ];
         dofs_rho_u1[ c3 ] = this->rho[ c3 ] * this->u1[ c3 ];
         dofs_rho_u2[ c3 ] = this->rho[ c3 ] * this->u2[ c3 ];
         dofs_e[ c3 ] = this->energy[ c3 ];
         /*dofs_e[ c3 ] = this->computeEnergy( this->rho[ c3 ],
                                                  this->temperature[ c3 ],
                                                  this->gamma,
                                                  this->u1[ c3 ],
                                                  this->u2[ c3 ] );*/

      }
   }
   writePhysicalVariables( time, -4 );

#ifdef HAVE_OPENMP
  #pragma omp parallel for, if( Devices::Host::isOMPEnabled() )
  #endif
  for( IndexType j = 0; j < ySize; j ++ )
     for( IndexType i = 0; i < xSize; i ++ )
     {
        IndexType c = this->mesh->getElementIndex( i, j );
        if( i == 0 || j == 0 ||
            i == xSize - 1 || j == ySize - 1 )
        {
           rho_t[ c ] = rho_u1_t[ c ] = rho_u2_t[ c ] = e_t[ c ] = 0.0;
           continue;
        }

        this->advection->getExplicitUpdate( c,
                                         rho_t[ c ],
                                         rho_u1_t[ c ],
                                         rho_u2_t[ c ],
                                         e_t[ c ],
                                         tau );

        //rho_u1_t[ c ] += ;
        //rho_u2_t[ c ] -= startUpCoefficient * this->gravity * this->rho[ c ];

        /***
         * Add the viscosity term
         */
        /*rho_u1_t[ c ] += this->mu*( u1Viscosity->getDiffusion( c, 4.0/3.0, 1.0, 0.0 ) +
                                    u2Viscosity->getDiffusion( c, 0.0, 0.0, 1.0/3.0 ) );
        rho_u2_t[ c ] += this->mu*( u2Viscosity->getDiffusion( c, 1.0, 4.0/3.0, 0.0 ) +
                                    u1Viscosity->getDiffusion( c, 0.0, 0.0, 1.0/3.0 ) );


        RealType k = 2.495*pow( 400.0, 1.5 ) / ( 400.0 + 194.0 );
        e_t[ c ] += this->mu*( u1Viscosity->getDiffusion( c,
                                                          this->u1, this->u1, this->u2,
                                                          4.0/3.0, 1.0, -2.0/3.0 ) +
                               u1Viscosity->getDiffusion( c,
                                                          this->u1, this->u1, this->u2,
                                                          0.0, 0.0, 1.0 ) +
                               u2Viscosity->getDiffusion( c,
                                                          this->u2, this->u2, this->u1,
                                                          1.0, 4.0/3.0, -2.0/3.0 ) +
                               u2Viscosity->getDiffusion( c,
                                                          this->u2, this->u2, this->u1,
                                                          0.0, 0.0, 1.0 ) +
                               k * energyViscosity->getDiffusion( c, 1.0, 1.0, 0.0 ) );
        */
        rho_u1_t[ c ] += this->mu*( u1Viscosity->getDiffusion( c, 1.0, 1.0, 0.0 ) );
        rho_u2_t[ c ] += this->mu*( u2Viscosity->getDiffusion( c, 1.0, 1.0, 0.0 ) );


        RealType k = 2.495*pow( 400.0, 1.5 ) / ( 400.0 + 194.0 );
        //cout << k << std::endl;
        /*e_t[ c ] += this->mu*( u1Viscosity->getDiffusion( c,
                                                          this->u1, this->u1, this->u2,
                                                          1.0, 1.0, 0.0 ) +
                               u2Viscosity->getDiffusion( c,
                                                          this->u2, this->u2, this->u1,
                                                          1.0, 1.0, -0.0 ) );*/
                               //energyViscosity->getDiffusion( c, 1.0, 1.0, 0.0 ) );


        //e_t[ c ] = 0.0;
     }

}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
bool NavierStokesSolver< AdvectionScheme,
                      DiffusionScheme,
                      BoundaryConditions >::writePhysicalVariables( const RealType& t,
                                                                    const IndexType step )
{
   SharedVector< RealType, DeviceType, IndexType > dofs_rho, dofs_rho_u1, dofs_rho_u2, dofs_e;
   const IndexType& dofs = mesh->getDofs();
   dofs_rho.    bind( & dofVector.getData()[ 0        ], dofs );
   dofs_rho_u1. bind( & dofVector.getData()[     dofs ], dofs );
   dofs_rho_u2. bind( & dofVector.getData()[ 2 * dofs ], dofs );
   dofs_e.      bind( & dofVector.getData()[ 3 * dofs ], dofs );

   this->updatePhysicalQuantities( dofs_rho, dofs_rho_u1, dofs_rho_u2, dofs_e );
   Vector< StaticVector< 2, RealType >, DeviceType, IndexType > u;
   u. setLike( u1 );
   String fileName;

   for( IndexType i = 0; i < this->u1. getSize(); i ++ )
      u[ i ] = StaticVector< 2, RealType >( this->u1[ i ], this->u2[ i ] );
   FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   if( ! u.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! this->rho.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "p-", step, 5, ".tnl", fileName );
   if( ! this->p.save( fileName ) )
      return false;
   FileNameBaseNumberEnding( "e-", step, 5, ".tnl", fileName );
   if( ! this->energy.save( fileName ) )
      return false;
   return true;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
bool NavierStokesSolver< AdvectionScheme,
                      DiffusionScheme,
                      BoundaryConditions >::writeConservativeVariables( const RealType& t,
                                                                        const IndexType step )
{
   SharedVector< RealType, DeviceType, IndexType > dofs_rho, dofs_rho_u1, dofs_rho_u2, dofs_e;

   const IndexType& dofs = mesh->getDofs();
   dofs_rho.    bind( & dofVector.getData()[ 0        ], dofs );
   dofs_rho_u1. bind( & dofVector.getData()[     dofs ], dofs );
   dofs_rho_u2. bind( & dofVector.getData()[ 2 * dofs ], dofs );
   dofs_e.      bind( & dofVector.getData()[ 3 * dofs ], dofs );

   String fileName;
   FileNameBaseNumberEnding( "rho-", step, 5, ".tnl", fileName );
   if( ! dofs_rho. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u1-", step, 5, ".tnl", fileName );
   if( ! dofs_rho_u1. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u2-", step, 5, ".tnl", fileName );
   if( ! dofs_rho_u2. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "e-", step, 5, ".tnl", fileName );
   if( ! dofs_e. save( fileName ) )
      return false;
   return true;
}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
typename NavierStokesSolver< AdvectionScheme,
                                DiffusionScheme,
                                BoundaryConditions >::RealType
   NavierStokesSolver< AdvectionScheme,
                          DiffusionScheme,
                          BoundaryConditions >::computeEnergy( const RealType& rho,
                                                               const RealType& pressure,
                                                               const RealType& gamma,
                                                               const RealType& u1,
                                                               const RealType& u2 ) const
{
   /*return rho * this->R * temperature / ( gamma - 1.0 ) +
                  0.5 * rho * ( u1*u1 + u2*u2 );*/
   return pressure / ( gamma - 1.0 ) +
                  0.5 * rho * ( u1*u1 + u2*u2 );

}

template< typename AdvectionScheme,
          typename DiffusionScheme,
          typename BoundaryConditions >
   template< typename DofVector >
bool NavierStokesSolver< AdvectionScheme,
                      DiffusionScheme,
                      BoundaryConditions >::writeExplicitRhs( const RealType& t,
                                                              const IndexType step,
                                                              DofVector& rhs )
{
   SharedVector< RealType, DeviceType, IndexType > dofs_rho, dofs_rho_u1, dofs_rho_u2, dofs_e;

   const IndexType& dofs = mesh->getDofs();
   dofs_rho.    bind( & rhs.getData()[ 0        ], dofs );
   dofs_rho_u1. bind( & rhs.getData()[     dofs ], dofs );
   dofs_rho_u2. bind( & rhs.getData()[ 2 * dofs ], dofs );
   dofs_e.      bind( & rhs.getData()[ 3 * dofs ], dofs );

   String fileName;
   FileNameBaseNumberEnding( "rho-t-", step, 5, ".tnl", fileName );
   if( ! dofs_rho. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u1-t-", step, 5, ".tnl", fileName );
   if( ! dofs_rho_u1. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "rho-u2-t-", step, 5, ".tnl", fileName );
   if( ! dofs_rho_u2. save( fileName ) )
      return false;

   FileNameBaseNumberEnding( "e-t-", step, 5, ".tnl", fileName );
   if( ! dofs_e. save( fileName ) )
      return false;
   return true;
}

} // namespace noa::TNL
