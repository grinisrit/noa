// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Functions/MeshFunction.h>

namespace noaTNL {
namespace Operators {

template< typename MeshFunction >
class IdentityOperator
   : public Domain< MeshFunction::getMeshDimension(), MeshFunction::getDomainType() >
{
   public:
 
      typedef typename MeshFunction::MeshType MeshType;
      typedef typename MeshFunction::RealType RealType;
      typedef typename MeshFunction::IndexType IndexType;
 
      OperatorComposition( const MeshFunction& meshFunction )
      : meshFunction( meshFunction ) {};
 
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshFunction& function,
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         static_assert( MeshFunction::getMeshDimension() == InnerOperator::getDimension(),
            "Mesh function and operator have both different number of dimensions." );
         return this->meshFunction( meshEntity, time );
      }
 
 
   protected:
 
      const MeshFunction& meshFunction;
};

} // namespace Operators
} // namespace noaTNL

