// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/String.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Operators/Operator.h>

namespace TNL {
namespace Operators {

template< typename OperatorT >
class FunctionInverseOperator
: public Operator< typename OperatorT::MeshType,
                      OperatorT::getDomainType(),
                      OperatorT::getPreimageEntitiesDimension(),
                      OperatorT::getImageEntitiesDimension(),
                      typename OperatorT::RealType,
                      typename OperatorT::IndexType >
{
   public:
 
      typedef OperatorT OperatorType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::IndexType IndexType;
      typedef FunctionInverseOperator ExactOperatorType;
 
      FunctionInverseOperator( const OperatorType& operator_ )
      : operator_( operator_ ) {};
 
      const OperatorType& getOperator() const { return this->operator_; }
 
      template< typename MeshFunction,
                typename MeshEntity >
      __cuda_callable__
      typename MeshFunction::RealType
      operator()( const MeshFunction& u,
                  const MeshEntity& entity,
                  const typename MeshFunction::RealType& time = 0.0 ) const
      {
         return 1.0 / operator_( u, entity, time );
      }
 
   protected:
 
      const OperatorType& operator_;
};

} // namespace Operators
} // namespace TNL

