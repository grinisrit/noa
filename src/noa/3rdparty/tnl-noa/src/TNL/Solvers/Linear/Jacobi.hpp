// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Functional.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/Jacobi.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
void
Jacobi< Matrix >::configSetup( Config::ConfigDescription& config, const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "jacobi-omega", "Relaxation parameter of the weighted/damped Jacobi method.", 1.0 );
   config.addEntry< int >(
      prefix + "residue-period", "Number of iterations between subsequent recomputations of the residue.", 4 );
}

template< typename Matrix >
bool
Jacobi< Matrix >::setup( const Config::ParameterContainer& parameters, const String& prefix )
{
   if( parameters.checkParameter( prefix + "jacobi-omega" ) )
      this->setOmega( parameters.getParameter< double >( prefix + "jacobi-omega" ) );
   if( this->omega <= 0.0 || this->omega > 2.0 ) {
      std::cerr << "Warning: The Jacobi method parameter omega is out of interval (0,2). The value is " << this->omega
                << " the method will not converge." << std::endl;
   }
   if( parameters.checkParameter( prefix + "residue-period" ) )
      this->setResiduePeriod( parameters.getParameter< int >( prefix + "residue-period" ) );
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
void
Jacobi< Matrix >::setOmega( RealType omega )
{
   this->omega = omega;
}

template< typename Matrix >
auto
Jacobi< Matrix >::getOmega() const -> RealType
{
   return omega;
}

template< typename Matrix >
void
Jacobi< Matrix >::setResiduePeriod( IndexType period )
{
   this->residuePeriod = period;
}

template< typename Matrix >
auto
Jacobi< Matrix >::getResiduePerid() const -> IndexType
{
   return this->residuePeriod;
}

template< typename Matrix >
bool
Jacobi< Matrix >::solve( ConstVectorViewType b, VectorViewType x )
{
   VectorType aux;
   aux.setLike( x );

   /////
   // Fetch diagonal elements
   this->diagonal.setLike( x );
   auto diagonalView = this->diagonal.getView();
   auto fetch_diagonal =
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, const IndexType& columnIdx, const RealType& value ) mutable
   {
      if( columnIdx == rowIdx )
         diagonalView[ rowIdx ] = value;
   };
   this->matrix->forAllElements( fetch_diagonal );

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   auto bView = b.getView();
   auto xView = x.getView();
   auto auxView = aux.getView();
   RealType bNorm = lpNorm( b, (RealType) 2.0 );
   aux = x;
   while( this->nextIteration() ) {
      this->performIteration( bView, diagonalView, xView, auxView );
      if( this->getIterations() % this->residuePeriod == 0 )
         this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
      this->currentIteration++;
      this->performIteration( bView, diagonalView, auxView, xView );
      if( ( this->getIterations() ) % this->residuePeriod == 0 )
         this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   }
   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   return this->checkConvergence();
}

template< typename Matrix >
void
Jacobi< Matrix >::performIteration( const ConstVectorViewType& b,
                                    const ConstVectorViewType& diagonalView,
                                    const ConstVectorViewType& in,
                                    VectorViewType& out ) const
{
   const RealType omega_ = this->omega;
   auto fetch = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value )
   {
      return value * in[ columnIdx ];
   };
   auto keep = [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
   {
      out[ rowIdx ] = in[ rowIdx ] + omega_ / diagonalView[ rowIdx ] * ( b[ rowIdx ] - value );
   };
   this->matrix->reduceAllRows( fetch, TNL::Plus{}, keep, 0.0 );
}

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL
