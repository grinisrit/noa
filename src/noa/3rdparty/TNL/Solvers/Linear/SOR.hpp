// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Functional.h>
#include <noa/3rdparty/TNL/Algorithms/AtomicOperations.h>
#include <noa/3rdparty/TNL/Solvers/Linear/SOR.h>
#include <noa/3rdparty/TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
void
SOR< Matrix >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "sor-omega", "Relaxation parameter of the SOR method.", 1.0 );
   config.addEntry< int >( prefix + "residue-period", "Says after how many iterations the reside is recomputed.", 4 );
}

template< typename Matrix >
bool
SOR< Matrix >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( parameters.checkParameter( prefix + "sor-omega" ) )
      this->setOmega( parameters.getParameter< double >( prefix + "sor-omega" ) );
   if( this->omega <= 0.0 || this->omega > 2.0 )
   {
      std::cerr << "Warning: The SOR method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << std::endl;
   }
   if( parameters.checkParameter( prefix + "residue-period" ) )
      this->setResiduePeriod( parameters.getParameter< int >( prefix + "residue-period" ) );
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
void SOR< Matrix > :: setOmega( const RealType& omega )
{
   this->omega = omega;
}

template< typename Matrix >
const typename SOR< Matrix > :: RealType& SOR< Matrix > :: getOmega( ) const
{
   return this->omega;
}

template< typename Matrix >
void
SOR< Matrix >::
setResiduePeriod( IndexType period )
{
   this->residuePeriod = period;
}

template< typename Matrix >
auto
SOR< Matrix >::
getResiduePerid() const -> IndexType
{
   return this->residuePeriod;
}

template< typename Matrix >
bool SOR< Matrix > :: solve( ConstVectorViewType b, VectorViewType x )
{
   /////
   // Fetch diagonal elements
   this->diagonal.setLike( x );
   auto diagonalView = this->diagonal.getView();
   auto fetch_diagonal = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, const IndexType& columnIdx, const RealType& value ) mutable {
      if( columnIdx == rowIdx ) diagonalView[ rowIdx ] = value;
   };
   this->matrix->forAllElements( fetch_diagonal );

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   auto bView = b.getView();
   auto xView = x.getView();
   RealType bNorm = lpNorm( b, ( RealType ) 2.0 );
   while( this->nextIteration() )
   {
      this->performIteration( bView, diagonalView, xView );
      if( this->getIterations() % this->residuePeriod == 0 )
         this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   }
   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   return this->checkConvergence();
};

template< typename Matrix >
void
SOR< Matrix >::
performIteration( const ConstVectorViewType& b,
                  const ConstVectorViewType& diagonalView,
                  VectorViewType& x ) const
{
   const RealType omega_ = this->omega;
   auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, const RealType& value ) {
         return value * x[ columnIdx ];
   };
   auto keep = [=] __cuda_callable__ ( IndexType rowIdx, const RealType& value ) mutable {
      Algorithms::AtomicOperations< DeviceType >::add( x[ rowIdx ], omega_ / diagonalView[ rowIdx ] * ( b[ rowIdx ] - value ) );
   };
   this->matrix->reduceAllRows( fetch, noa::TNL::Plus{}, keep, 0.0 );
}

} // namespace Linear
} // namespace Solvers
} // namespace noa::TNL
