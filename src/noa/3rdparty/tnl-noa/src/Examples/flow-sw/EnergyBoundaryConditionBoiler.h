#pragma once

#include <TNL/Functions/FunctionAdapter.h>
#include "CompressibleConservativeVariables.h"

namespace TNL {
namespace Operators {

template< typename Mesh,
          typename Function,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType >
class EnergyBoundaryConditionsBoiler
{

};

/****
 * Base
 */
template< typename Function>
class EnergyBoundaryConditionsBoilerBase
{
   public:
      
      typedef Function FunctionType;
      
      static void configSetup( const Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         Function::configSetup( config, prefix );
      }
      
      template< typename MeshPointer >
      bool setup( const MeshPointer& meshPointer, 
                  const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
         return Functions::FunctionAdapter< typename MeshPointer::ObjectType, FunctionType >::setup( this->function, meshPointer, parameters, prefix );
      }

      static void configSetup( Config::ConfigDescription& config,
                               const String& prefix = "" )
      {
         Function::configSetup( config, prefix );
      };

      bool setup( const Config::ParameterContainer& parameters,
                  const String& prefix = "" )
      {
          return this->function.setup( parameters, prefix );
      };

      void setFunction( const FunctionType& function )
      {
         this->function = function;
      };
      
      FunctionType& getFunction()
      {
         return this->function;
      }

      const FunctionType& getFunction() const
      {
         return this->function;
      };

   protected:

      FunctionType function;


};

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class EnergyBoundaryConditionsBoiler< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public EnergyBoundaryConditionsBoilerBase< Function >,
     public Operator< Meshes::Grid< 1, MeshReal, Device, MeshIndex >,
                         Functions::MeshBoundaryDomain,
                         1, 1,
                         Real,
                         Index >
{
   public:

   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   typedef Function FunctionType;
   typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
   typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
   typedef Containers::StaticVector< 1, RealType > PointType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef EnergyBoundaryConditionsBoilerBase< Function > BaseType;
   typedef CompressibleConservativeVariables< MeshType > CompressibleConservativeVariablesType;
   typedef Pointers::SharedPointer< CompressibleConservativeVariablesType > CompressibleConservativeVariablesPointer;
   typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;

   template< typename EntityType,
             typename MeshFunction >
   __cuda_callable__
   const RealType operator()( const MeshFunction& u,
                              const EntityType& entity,   
                              const RealType& time = 0 ) const
   {
      const MeshType& mesh = entity.getMesh();
      const auto& neighborEntities = entity.getNeighborEntities();
      const IndexType& index = entity.getIndex();
      if( entity.getCoordinates().x() == 0 )
         return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0 >()]
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0 >()]
                  + this->timestep
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0 >()]
                  + this->timestep
                  );
      else
         return u[ neighborEntities.template getEntityIndex< -1 >() ];   

   }


   template< typename EntityType >
   __cuda_callable__
   Index getLinearSystemRowLength( const MeshType& mesh,
                                   const IndexType& index,
                                   const EntityType& entity ) const
   {
      return 2;
   }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const
      {
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index, 1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() * 
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         else
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< -1 >(), -1.0 );
            matrixRow.setElement( 1, index, 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }

      void setTimestep(const RealType timestep )
      {
         this->timestep = timestep;
      }

      void setGamma(const RealType gamma )
      {
         this->gamma = gamma;
      }

      void setCompressibleConservativeVariables(const CompressibleConservativeVariablesPointer& compressibleConservativeVariables)
      {
         this->compressibleConservativeVariables = compressibleConservativeVariables;
      }

      void setPressure(const MeshFunctionPointer& pressure)
      {
         this->pressure = pressure;
      }

      void setCavitySpeed(const RealType cavitySpeed)
      {
         this->cavitySpeed = cavitySpeed;
      }

   private:
      CompressibleConservativeVariablesPointer compressibleConservativeVariables;
      RealType timestep;
      RealType cavitySpeed;
      RealType gamma;
      MeshFunctionPointer pressure;
};

/****
 * 2D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class EnergyBoundaryConditionsBoiler< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public EnergyBoundaryConditionsBoilerBase< Function >,
     public Operator< Meshes::Grid< 2, MeshReal, Device, MeshIndex >,
                         Functions::MeshBoundaryDomain,
                         2, 2,
                         Real,
                         Index >

{
   public:

      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      typedef Function FunctionType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef Containers::StaticVector< 2, RealType > PointType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef EnergyBoundaryConditionsBoilerBase< Function > BaseType;
      typedef CompressibleConservativeVariables< MeshType > CompressibleConservativeVariablesType;
      typedef Pointers::SharedPointer< CompressibleConservativeVariablesType > CompressibleConservativeVariablesPointer;
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;


      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,                            
                                 const RealType& time = 0 ) const
      {
         const MeshType& mesh = entity.getMesh();
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         if( entity.getCoordinates().x() == 0 )
         {
            if( ( entity.getCoordinates().y() < 0.20 * ( entity.getMesh().getDimensions().y() - 1 ) ) && ( entity.getCoordinates().y() > 0.19 * ( entity.getMesh().getDimensions().y() - 1 ) ) )
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                * (
                  this->cavitySpeed
                * 
                  this->cavitySpeed
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                  + 0
                  )
                );
            else 
               return u[ neighborEntities.template getEntityIndex< 0, 0 >() ];
                    /*( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0 >() ]
                      / ( this->gamma - 1 )
                      )
                     + 0.5
                     * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                     * (
                         ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 1, 0 >()]
                         / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 1, 0 >()]
                         + 0
                       )
                       * 
                       ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 1, 0 >()]
                         /  (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 1, 0 >()]
                         + 0
                       )
                     );*/
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            if( ( entity.getCoordinates().y() < 0.20 * ( entity.getMesh().getDimensions().y() - 1 ) ) && ( entity.getCoordinates().y() > 0.19 * ( entity.getMesh().getDimensions().y() - 1 ) ) ) 
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                * (
                  this->cavitySpeed * ( -1 )
                * 
                  this->cavitySpeed * ( -1 )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                  + 0
                  )
                );
            else if( entity.getCoordinates().y() > 0.8 * ( entity.getMesh().getDimensions().y() - 1 ) && false )
               return u[ neighborEntities.template getEntityIndex< 0, 0 >() ];
            else
               return u[ neighborEntities.template getEntityIndex< 0, 0 >() ];
                    /*( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0 >() ]
                    / ( this->gamma - 1 )
                    )
                    + 0.5
                    * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                    * (
                        ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< -1, 0 >()]
                        / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< -1, 0 >()]
                        + 0
                      )
                      * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< -1, 0 >()]
                        / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< -1, 0 >()]
                        + 0
                        )
                );*/
         }
         if( entity.getCoordinates().y() == 0 )
         {
            if( ( entity.getCoordinates().x() < 0.6 * ( entity.getMesh().getDimensions().x() - 1 ) ) && ( entity.getCoordinates().x() > 0.4 * ( entity.getMesh().getDimensions().x() - 1 ) ) ) 
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                * (
                  ( 
                  (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                  + 0
                  )
                * ( 
                  (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                  + 0
                  )
                +
                  this->cavitySpeed
                * 
                  this->cavitySpeed
                );
            else 
               return u[ neighborEntities.template getEntityIndex< 0, 0 >() ];
                    /*( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0 >() ]
                    / ( this->gamma - 1 )
                    )
                   + 0.5
                   * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                   * (
                       ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 1 >()]
                       / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 1 >()]
                       + 0
                     )
                   * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 1 >()]
                     / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 1 >()]
                     + 0
                     )
                );*/
         }
         // The following line is commented to avoid compiler warning
         //if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, -1 >() ];
            /*return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0 >()]
                * (
                    ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, -1 >()]
                    / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, -1 >()]
                    + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, -1 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, -1 >()]
                  + 0
                  )
                );*/
         }         
      }

      template< typename EntityType >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity ) const
      {
         return 2;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      void setMatrixElements( const PreimageFunction& u,
                              const MeshEntity& entity,
                              const RealType& time,
                              const RealType& tau,
                              Matrix& matrix,
                              Vector& b ) const
      {
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index,                                                1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 1, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< -1, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            matrixRow.setElement( 0, index,                                                1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 0, 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, -1 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                 1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }         
      }

      void setTimestep(const RealType timestep )
      {
         this->timestep = timestep;
      }

      void setGamma(const RealType gamma )
      {
         this->gamma = gamma;
      }

      void setCompressibleConservativeVariables(const CompressibleConservativeVariablesPointer& compressibleConservativeVariables)
      {
         this->compressibleConservativeVariables = compressibleConservativeVariables;
      }

      void setPressure(const MeshFunctionPointer& pressure)
      {
         this->pressure = pressure;
      }

      void setCavitySpeed(const RealType cavitySpeed)
      {
         this->cavitySpeed = cavitySpeed;
      }

   private:
      CompressibleConservativeVariablesPointer compressibleConservativeVariables;
      RealType timestep;
      RealType cavitySpeed;
      RealType gamma;
      MeshFunctionPointer pressure;
};

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
class EnergyBoundaryConditionsBoiler< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >
   : public EnergyBoundaryConditionsBoilerBase< Function >,
     public Operator< Meshes::Grid< 3, MeshReal, Device, MeshIndex >,
                         Functions::MeshBoundaryDomain,
                         3, 3,
                         Real,
                         Index >
{
   public:

      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;

      typedef Function FunctionType;
      typedef Functions::MeshFunctionView< MeshType > MeshFunctionType;
      typedef Containers::Vector< RealType, DeviceType, IndexType> DofVectorType;
      typedef Containers::StaticVector< 3, RealType > PointType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      typedef EnergyBoundaryConditionsBoilerBase< Function > BaseType;  
      typedef CompressibleConservativeVariables< MeshType > CompressibleConservativeVariablesType;
      typedef Pointers::SharedPointer< CompressibleConservativeVariablesType > CompressibleConservativeVariablesPointer; 
      typedef Pointers::SharedPointer< MeshFunctionType, DeviceType > MeshFunctionPointer;

      template< typename EntityType,
                typename MeshFunction >
      __cuda_callable__
      const RealType operator()( const MeshFunction& u,
                                 const EntityType& entity,
                                 const RealType& time = 0 ) const
      {
         const MeshType& mesh = entity.getMesh();
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         if( entity.getCoordinates().x() == 0 )
         {
            if( ( entity.getCoordinates().y() < 0.59 * ( entity.getMesh().getDimensions().y() - 1 ) ) && ( entity.getCoordinates().y() > 0.39 * ( entity.getMesh().getDimensions().y() - 1 ) )
              &&( entity.getCoordinates().z() < 0.59 * ( entity.getMesh().getDimensions().z() - 1 ) ) && ( entity.getCoordinates().z() > 0.39 * ( entity.getMesh().getDimensions().z() - 1 ) ) )
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                * (
                  this->cavitySpeed
                * 
                  this->cavitySpeed
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                );
            else 
               return u[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ];
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            if( ( entity.getCoordinates().y() < 0.59 * ( entity.getMesh().getDimensions().y() - 1 ) ) && ( entity.getCoordinates().y() > 0.39 * ( entity.getMesh().getDimensions().y() - 1 ) )
              &&( entity.getCoordinates().z() < 0.59 * ( entity.getMesh().getDimensions().z() - 1 ) ) && ( entity.getCoordinates().z() > 0.39 * ( entity.getMesh().getDimensions().z() - 1 ) ) )
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                * (
                  this->cavitySpeed * ( -1 )
                * 
                  this->cavitySpeed * ( -1 )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                );
            else if( entity.getCoordinates().y() >0.8 * ( entity.getMesh().getDimensions().y() - 1 ) )
               return u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ];
            else
               return u[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ];
         }
         if( entity.getCoordinates().y() == 0 )
         {
            if( ( entity.getCoordinates().x() < 0.59 * ( entity.getMesh().getDimensions().x() - 1 ) ) && ( entity.getCoordinates().x() > 0.39 * ( entity.getMesh().getDimensions().x() - 1 ) )
              &&( entity.getCoordinates().z() < 0.59 * ( entity.getMesh().getDimensions().z() - 1 ) ) && ( entity.getCoordinates().z() > 0.39 * ( entity.getMesh().getDimensions().z() - 1 ) ) )
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                * (
                  this->cavitySpeed
                * 
                  this->cavitySpeed
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                );
            else 
               return u[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ];
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            if( ( entity.getCoordinates().x() < 0.59 * ( entity.getMesh().getDimensions().x() - 1 ) ) && ( entity.getCoordinates().x() > 0.39 * ( entity.getMesh().getDimensions().x() - 1 ) )
              &&( entity.getCoordinates().z() < 0.59 * ( entity.getMesh().getDimensions().z() - 1 ) ) && ( entity.getCoordinates().z() > 0.39 * ( entity.getMesh().getDimensions().z() - 1 ) ) )
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                * (
                  this->cavitySpeed * ( -1 )
                * 
                  this->cavitySpeed * ( -1 )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 2 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                );
            else 
               return u[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ];
         }
         if( entity.getCoordinates().z() == 0 )
{
            if( ( entity.getCoordinates().x() < 0.6 * ( entity.getMesh().getDimensions().y() - 1 ) ) && ( entity.getCoordinates().y() > 0.4 * ( entity.getMesh().getDimensions().y() - 1 ) )
              &&( entity.getCoordinates().y() < 0.6 * ( entity.getMesh().getDimensions().z() - 1 ) ) && ( entity.getCoordinates().z() > 0.4 * ( entity.getMesh().getDimensions().z() - 1 ) ) )
               return ( (* this->pressure)[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ]
                / ( this->gamma - 1 )
                )
                + 0.5
                * (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                * 
                (
                  ( 
                    (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 0 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                +
                  ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                * ( (* (* this->compressibleConservativeVariables->getMomentum())[ 1 ])[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  / (* this->compressibleConservativeVariables->getDensity())[neighborEntities.template getEntityIndex< 0, 0, 0 >()]
                  + 0
                  )
                +
                  (
                  this->cavitySpeed
                * 
                  this->cavitySpeed
                  )
                );
               
            else 
               return u[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ];
         }
         // The following line is commented to avoid compiler warning
         //if( entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1 )
         {
            return u[ neighborEntities.template getEntityIndex< 0, 0, 0 >() ];
         }   
      }


      template< typename EntityType >
      __cuda_callable__
      Index getLinearSystemRowLength( const MeshType& mesh,
                                      const IndexType& index,
                                      const EntityType& entity ) const
      {
         return 2;
      }

      template< typename PreimageFunction,
                typename MeshEntity,
                typename Matrix,
                typename Vector >
      __cuda_callable__
      void setMatrixElements( const PreimageFunction& u,
                                     const MeshEntity& entity,
                                     const RealType& time,
                                     const RealType& tau,
                                     Matrix& matrix,
                                     Vector& b ) const
      {
         const auto& neighborEntities = entity.getNeighborEntities();
         const IndexType& index = entity.getIndex();
         typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
         if( entity.getCoordinates().x() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 1, 0, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().x() == entity.getMesh().getDimensions().x() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< -1, 0, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().x() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 0, 1, 0 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() * 
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().y() == entity.getMesh().getDimensions().y() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, -1, 0 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().y() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == 0 )
         {
            matrixRow.setElement( 0, index,                                                   1.0 );
            matrixRow.setElement( 1, neighborEntities.template getEntityIndex< 0, 0, 1 >(), -1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().z() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
         if( entity.getCoordinates().z() == entity.getMesh().getDimensions().z() - 1 )
         {
            matrixRow.setElement( 0, neighborEntities.template getEntityIndex< 0, 0, -1 >(), -1.0 );
            matrixRow.setElement( 1, index,                                                    1.0 );
            b[ index ] = entity.getMesh().getSpaceSteps().z() *
               Functions::FunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
         }
      }

      void setTimestep(const RealType timestep )
      {
         this->timestep = timestep;
      }

      void setGamma(const RealType gamma )
      {
         this->gamma = gamma;
      }

      void setCompressibleConservativeVariables(const CompressibleConservativeVariablesPointer& compressibleConservativeVariables)
      {
         this->compressibleConservativeVariables = compressibleConservativeVariables;
      }

      void setPressure(const MeshFunctionPointer& pressure)
      {
         this->pressure = pressure;
      }

      void setCavitySpeed(const RealType cavitySpeed)
      {
         this->cavitySpeed = cavitySpeed;
      }

   private:
      CompressibleConservativeVariablesPointer compressibleConservativeVariables;
      RealType timestep;
      RealType cavitySpeed;
      RealType gamma;
      MeshFunctionPointer pressure;
};

template< typename Mesh,
          typename Function,
          typename Real,
          typename Index >
std::ostream& operator << ( std::ostream& str, const EnergyBoundaryConditionsBoiler< Mesh, Function, Real, Index >& bc )
{
   str << "Neumann boundary ConditionsBoiler: function = " << bc.getFunction();
   return str;
}

} // namespace Operators
} // namespace TNL

