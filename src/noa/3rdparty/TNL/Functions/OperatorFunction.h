// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <noa/3rdparty/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/TNL/Functions/MeshFunction.h>
#include <noa/3rdparty/TNL/Solvers/PDE/BoundaryConditionsSetter.h>

namespace noa::TNL {
namespace Functions {   

/***
 * This class evaluates given operator on given preimageFunction. If the flag
 * EvaluateOnFly is set on true, the values on particular mesh entities
 * are computed just  when operator() is called. If the EvaluateOnFly flag
 * is 'false', values on all mesh entities are evaluated by calling a method
 * refresh() they are stores in internal mesh preimageFunction and the operator()
 * just returns precomputed values. If BoundaryConditions are void then the
 * values on the boundary mesh entities are undefined. In this case, the mesh
 * preimageFunction evaluator evaluates this preimageFunction only on the INTERIOR mesh entities.
 */
   
template< typename Operator,
          typename Function = typename Operator::FunctionType,
          typename BoundaryConditions = void,
          bool EvaluateOnFly = false,
          bool IsAnalytic = ( Function::getDomainType() == SpaceDomain || Function::getDomainType() == NonspaceDomain ) >
class OperatorFunction{};

/****
 * Specialization for 'On the fly' evaluation with the boundary conditions does not make sense.
 */
template< typename Operator,
          typename MeshFunctionT,
          typename BoundaryConditions,
          bool IsAnalytic >
class OperatorFunction< Operator, MeshFunctionT, BoundaryConditions, true, IsAnalytic >
 : public Domain< Operator::getDimension(), MeshDomain >
{
};

/****
 * Specialization for 'On the fly' evaluation and no boundary conditions.
 */
template< typename Operator,
          typename MeshFunctionT,
          bool IsAnalytic >
class OperatorFunction< Operator, MeshFunctionT, void, true, IsAnalytic >
 : public Domain< Operator::getDomainDimension(), Operator::getDomainType() >
{
   public:
 
      static_assert( MeshFunctionT::getDomainType() == MeshDomain ||
                     MeshFunctionT::getDomainType() == MeshInteriorDomain ||
                     MeshFunctionT::getDomainType() == MeshBoundaryDomain,
         "Only mesh preimageFnctions may be used in the operator preimageFunction. Use ExactOperatorFunction instead of OperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename MeshFunctionT::MeshType >::value,
          "Both, operator and mesh preimageFunction must be defined on the same mesh." );
 
      typedef Operator OperatorType;
      typedef MeshFunctionT FunctionType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef typename OperatorType::ExactOperatorType ExactOperatorType;
      typedef MeshFunction< MeshType, OperatorType::getPreimageEntitiesDimension() > PreimageFunctionType;
      typedef Pointers::SharedPointer<  MeshType, DeviceType > MeshPointer;
      
      static constexpr int getEntitiesDimension() { return OperatorType::getImageEntitiesDimension(); };     
      
      OperatorFunction( const OperatorType& operator_ )
      :  operator_( operator_ ), preimageFunction( 0 ){};
 
      OperatorFunction( const OperatorType& operator_,
                           const FunctionType& preimageFunction )
      :  operator_( operator_ ), preimageFunction( &preimageFunction ){};
 
      const MeshType& getMesh() const
      {
         TNL_ASSERT_TRUE( this->preimageFunction, "The preimage function was not set." );
         return this->preimageFunction->getMesh();
      };
      
      const MeshPointer& getMeshPointer() const
      { 
         TNL_ASSERT_TRUE( this->preimageFunction, "The preimage function was not set." );
         return this->preimageFunction->getMeshPointer(); 
      };

      
      void setPreimageFunction( const FunctionType& preimageFunction ) { this->preimageFunction = &preimageFunction; }
 
      Operator& getOperator() { return this->operator_; }
 
      const Operator& getOperator() const { return this->operator_; }
 
      bool refresh( const RealType& time = 0.0 ) { return true; };
 
      bool deepRefresh( const RealType& time = 0.0 ) { return true; };
 
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0.0 ) const
      {
         TNL_ASSERT_TRUE( this->preimageFunction, "The preimage function was not set." );
         return operator_( *preimageFunction, meshEntity, time );
      }
 
   protected:
 
      const Operator& operator_;
 
      const FunctionType* preimageFunction;
 
      template< typename, typename > friend class MeshFunctionEvaluator;
};

/****
 * Specialization for precomputed evaluation and no boundary conditions.
 */
template< typename Operator,
          typename PreimageFunction,
          bool IsAnalytic >
class OperatorFunction< Operator, PreimageFunction, void, false, IsAnalytic >
 : public Domain< Operator::getDomainDimension(), Operator::getDomainType() >
{
   public:
 
      static_assert( PreimageFunction::getDomainType() == MeshDomain ||
                     PreimageFunction::getDomainType() == MeshInteriorDomain ||
                     PreimageFunction::getDomainType() == MeshBoundaryDomain,
         "Only mesh preimageFunctions may be used in the operator preimageFunction. Use ExactOperatorFunction instead of OperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename PreimageFunction::MeshType >::value,
          "Both, operator and mesh preimageFunction must be defined on the same mesh." );
 
      typedef Operator OperatorType;
      typedef typename OperatorType::MeshType MeshType;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef PreimageFunction PreimageFunctionType;
      typedef Functions::MeshFunction< MeshType, Operator::getImageEntitiesDimension() > ImageFunctionType;
      typedef OperatorFunction< Operator, PreimageFunction, void, true > OperatorFunctionType;
      typedef typename OperatorType::ExactOperatorType ExactOperatorType;
      typedef Pointers::SharedPointer<  MeshType, DeviceType > MeshPointer;
      
      static constexpr int getEntitiesDimension() { return OperatorType::getImageEntitiesDimension(); };     
      
      OperatorFunction( OperatorType& operator_,
                           const MeshPointer& mesh )
      :  operator_( operator_ ), imageFunction( mesh )
      {};
 
      OperatorFunction( OperatorType& operator_,
                           PreimageFunctionType& preimageFunction )
      :  operator_( operator_ ), imageFunction( preimageFunction.getMeshPointer() ), preimageFunction( &preimageFunction )
      {};
 
      const MeshType& getMesh() const { return this->imageFunction.getMesh(); };
      
      const MeshPointer& getMeshPointer() const { return this->imageFunction.getMeshPointer(); };
      
      ImageFunctionType& getImageFunction() { return this->imageFunction; };
 
      const ImageFunctionType& getImageFunction() const { return this->imageFunction; };
 
      void setPreimageFunction( PreimageFunctionType& preimageFunction )
      {
         this->preimageFunction = &preimageFunction;
         this->imageFunction.setMesh( preimageFunction.getMeshPointer() );
      };
 
      const PreimageFunctionType& getPreimageFunction() const { return *this->preimageFunction; };
 
      Operator& getOperator() { return this->operator_; }
 
      const Operator& getOperator() const { return this->operator_; }

      bool refresh( const RealType& time = 0.0 )
      {
         OperatorFunction operatorFunction( this->operator_, *preimageFunction );
         this->operator_.setPreimageFunction( *this->preimageFunction );
         if( ! this->operator_.refresh( time ) ||
             ! operatorFunction.refresh( time )  )
             return false;
         this->imageFunction = operatorFunction;
         return true;
      };
 
      bool deepRefresh( const RealType& time = 0.0 )
      {
         if( ! this->preimageFunction->deepRefresh( time ) )
            return false;
         return this->refresh( time );
      };
 
      template< typename MeshEntity >
      __cuda_callable__
      RealType operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return imageFunction[ meshEntity.getIndex() ];
      }
 
      __cuda_callable__
      RealType operator[]( const IndexType& index ) const
      {
         return imageFunction[ index ];
      }
 
   protected:
 
      Operator& operator_;
 
      PreimageFunctionType* preimageFunction;
 
      ImageFunctionType imageFunction;
 
      template< typename, typename > friend class MeshFunctionEvaluator;
};

/****
 * Specialization for precomputed evaluation and with boundary conditions.
 */
template< typename Operator,
          typename Function,
          typename BoundaryConditions,
          bool IsAnalytic >
class OperatorFunction< Operator, Function, BoundaryConditions, false, IsAnalytic >
  : public Domain< Operator::getDimension(), MeshDomain >
{
   public:
 
      static_assert( Function::getDomainType() == MeshDomain ||
                     Function::getDomainType() == MeshInteriorDomain ||
                     Function::getDomainType() == MeshBoundaryDomain,
         "Only mesh preimageFunctions may be used in the operator preimageFunction. Use ExactOperatorFunction instead of OperatorFunction." );
      static_assert( std::is_same< typename Operator::MeshType, typename Function::MeshType >::value,
          "Both, operator and mesh preimageFunction must be defined on the same mesh." );
      static_assert( std::is_same< typename BoundaryConditions::MeshType, typename Operator::MeshType >::value,
         "The operator and the boundary conditions are defined on different mesh types." );
 
      typedef Operator OperatorType;
      typedef typename OperatorType::MeshType MeshType;
      typedef Pointers::SharedPointer<  MeshType > MeshPointer;
      typedef typename OperatorType::RealType RealType;
      typedef typename OperatorType::DeviceType DeviceType;
      typedef typename OperatorType::IndexType IndexType;
      typedef Function PreimageFunctionType;
      typedef Functions::MeshFunction< MeshType, Operator::getImageEntitiesDimension() > ImageFunctionType;
      typedef BoundaryConditions BoundaryConditionsType;
      typedef OperatorFunction< Operator, Function, void, true > OperatorFunctionType;
      typedef typename OperatorType::ExactOperatorType ExactOperatorType;
 
      static constexpr int getEntitiesDimension() { return OperatorType::getImageEntitiesDimension(); };
 
      OperatorFunction( OperatorType& operator_,
                           const BoundaryConditionsType& boundaryConditions,
                           const MeshPointer& meshPointer )
      :  operator_( operator_ ),
         boundaryConditions( boundaryConditions ),
         imageFunction( meshPointer )//,
         //preimageFunction( 0 )
      {
         this->preimageFunction = NULL;
      };
      
      OperatorFunction( OperatorType& operator_,
                           const BoundaryConditionsType& boundaryConditions,
                           const PreimageFunctionType& preimageFunction )
      :  operator_( operator_ ),
         boundaryConditions( boundaryConditions ),
         imageFunction( preimageFunction.getMeshPointer() ),
         preimageFunction( &preimageFunction )
      {};
 
      const MeshType& getMesh() const { return imageFunction.getMesh(); };
      
      const MeshPointer& getMeshPointer() const { return imageFunction.getMeshPointer(); };
      
      void setPreimageFunction( const PreimageFunctionType& preimageFunction )
      {
         this->preimageFunction = &preimageFunction;
      }
 
      const PreimageFunctionType& getPreimageFunction() const
      {
         TNL_ASSERT_TRUE( this->preimageFunction, "The preimage function was not set." );
         return *this->preimageFunction;
      };
 
      PreimageFunctionType& getPreimageFunction()
      {
         TNL_ASSERT_TRUE( this->preimageFunction, "The preimage function was not set." );
         return *this->preimageFunction;
      };
 
      const ImageFunctionType& getImageFunction() const { return this->imageFunction; };
 
      ImageFunctionType& getImageFunction() { return this->imageFunction; };
 
      Operator& getOperator() { return this->operator_; }
 
      const Operator& getOperator() const { return this->operator_; }

      bool refresh( const RealType& time = 0.0 )
      {
         OperatorFunctionType operatorFunction( this->operator_, *this->preimageFunction );
         this->operator_.setPreimageFunction( *this->preimageFunction );
         if( ! this->operator_.refresh( time ) ||
             ! operatorFunction.refresh( time )  )
             return false;
         this->imageFunction = operatorFunction;
         Solvers::PDE::BoundaryConditionsSetter< ImageFunctionType, BoundaryConditionsType >::apply( this->boundaryConditions, time, this->imageFunction );
         return true;
      };
 
      bool deepRefresh( const RealType& time = 0.0 )
      {
         return preimageFunction->deepRefresh( time ) &&
                this->refresh( time );
      };
 
      template< typename MeshEntity >
      __cuda_callable__
      const RealType& operator()(
         const MeshEntity& meshEntity,
         const RealType& time = 0 ) const
      {
         return imageFunction[ meshEntity.getIndex() ];
      }
 
      __cuda_callable__
      const RealType& operator[]( const IndexType& index ) const
      {
         return imageFunction[ index ];
      }
 
   protected:
 
      Operator& operator_;
 
      const PreimageFunctionType* preimageFunction;
 
      ImageFunctionType imageFunction;
 
      const BoundaryConditionsType& boundaryConditions;
 
      template< typename, typename > friend class MeshFunctionEvaluator;
};

/****
 * Specialization for precomputed evaluation and with boundary conditions.
 */
template< typename Operator,
          typename Function >
class OperatorFunction< Operator, Function, void, false, true >
  : public Domain< Function::getDomainDimension(), Function::getDomainType() >
{
   public:
      
      typedef Function FunctionType;
      typedef typename FunctionType::RealType RealType;
      typedef typename FunctionType::PointType PointType;
      typedef Operator OperatorType;
      
      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return( this->function.setup( parameters, prefix) &&
                 this->operator_.setup( parameters, prefix ) );
      }
      
      __cuda_callable__
      FunctionType& getFunction()
      {
         return this->function;
      }
      
      __cuda_callable__
      const FunctionType& getFunction() const
      {
         return this->function;
      }
      
      __cuda_callable__
      OperatorType& getOperator()
      {
         return this->operator_;
      }
      
      __cuda_callable__
      const OperatorType& getOperator() const
      {
         return this->operator_;
      }
      
      __cuda_callable__
      RealType operator()( const PointType& v,
                           const RealType& time = 0.0 ) const
      {
         return this->operator_( this->function, v, time );
      }
      
   protected:
      
      Function function;
      
      Operator operator_;
 
};

} // namespace Functions
} // namespace noa::TNL

