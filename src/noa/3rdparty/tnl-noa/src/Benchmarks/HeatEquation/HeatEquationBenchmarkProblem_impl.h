#ifndef HeatEquationBenchmarkPROBLEM_IMPL_H_
#define HeatEquationBenchmarkPROBLEM_IMPL_H_

#include <TNL/FileName.h>
#include <TNL/Matrices/MatrixSetter.h>
#include <TNL/Solvers/PDE/LinearSystemAssembler.h>
#include <TNL/Solvers/PDE/BackwardTimeDiscretisation.h>
#include <TNL/Solvers/PDE/ExplicitUpdater.h>
#include "TestGridEntity.h"
#include "Tuning/tunning.h"
#include "Tuning/SimpleCell.h"
#include "Tuning/GridTraverser.h"

//#define WITH_TNL  // In the 'tunning' part, this serves for comparison of performance
                  // when using common TNL structures compared to the benchmark ones



template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
HeatEquationBenchmarkProblem()
: cudaMesh( 0 ),
  cudaBoundaryConditions( 0 ),
  cudaRightHandSide( 0 ),
  cudaDifferentialOperator( 0 )
{
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
String
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getPrologHeader() const
{
   if( this->cudaKernelType == "pure-c" )
      return "Heat Equation Benchmark PURE-C test";
   if( this->cudaKernelType == "templated" )
      return "Heat Equation Benchmark TEMPLATED test";
   if( this->cudaKernelType == "templated-compact" )
      return "Heat Equation Benchmark TEMPLATED COMPACT test";
   if( this->cudaKernelType == "tunning" )
      return "Heat Equation Benchmark TUNNIG test";
   return "";
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
   /****
    * Add data you want to have in the computation report (log) as follows:
    * logger.writeParameter< double >( "Parameter description", parameter );
    */
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( ! this->boundaryConditionPointer->setup( this->getMesh(), parameters, "boundary-conditions-" ) ||
       ! this->rightHandSidePointer->setup( parameters, "right-hand-side-" ) )
      return false;
   this->cudaKernelType = parameters.getParameter< String >( "cuda-kernel-type" );

   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      this->cudaBoundaryConditions = Cuda::passToDevice( *this->boundaryConditionPointer );
      this->cudaRightHandSide = Cuda::passToDevice( *this->rightHandSidePointer );
      this->cudaDifferentialOperator = Cuda::passToDevice( *this->differentialOperatorPointer );
   }
   this->explicitUpdater.setDifferentialOperator( this->differentialOperatorPointer );
   this->explicitUpdater.setBoundaryConditions( this->boundaryConditionPointer );
   this->explicitUpdater.setRightHandSide( this->rightHandSidePointer );
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
typename HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::IndexType
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getDofs() const
{
   /****
    * Return number of  DOFs (degrees of freedom) i.e. number
    * of unknowns to be resolved by the main solver.
    */
   return this->getMesh()->template getEntitiesCount< typename MeshType::Cell >();
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
bindDofs( DofVectorPointer& dofsPointer )
{
   this->u->bind( this->getMesh(), *dofsPointer );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setInitialCondition( const Config::ParameterContainer& parameters,
                     DofVectorPointer& dofsPointer )
{
   const String& initialConditionFile = parameters.getParameter< String >( "initial-condition" );
   MeshFunctionViewType u;
   u.bind( this->getMesh(), *dofsPointer );
   if( ! Functions::readMeshFunction( u, "u", initialConditionFile ) )
   {
      std::cerr << "I am not able to load the initial condition from the file " << initialConditionFile << "." << std::endl;
      return false;
   }
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename Matrix >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
setupLinearSystem( Matrix& matrix )
{
   const IndexType dofs = this->getDofs();
   using RowsCapacitiesTypeType = typename Matrix::RowsCapacitiesType;
   RowsCapacitiesTypeType rowLengths;
   if( ! rowLengths.setSize( dofs ) )
      return false;
   Matrices::MatrixSetter< MeshType, DifferentialOperator, BoundaryCondition, RowsCapacitiesTypeType > matrixSetter;
   matrixSetter.template getCompressedRowLengths< typename Mesh::Cell >( this->getMesh(),
                                                                          differentialOperatorPointer,
                                                                          boundaryConditionPointer,
                                                                          rowLengths );
   matrix.setDimensions( dofs, dofs );
   if( ! matrix.setCompressedRowLengths( rowLengths ) )
      return false;
   return true;
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
bool
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
makeSnapshot( const RealType& time,
              const IndexType& step,
              DofVectorPointer& dofsPointer )
{
   std::cout << std::endl << "Writing output at time " << time << " step " << step << "." << std::endl;
   this->bindDofs( dofsPointer );
   MeshFunctionViewType u;
   u.bind( this->getMesh(), *dofsPointer );

   FileName fileName;
   fileName.setFileNameBase( "u-" );
   fileName.setExtension( "vti" );
   fileName.setIndex( step );

   //FileNameBaseNumberEnding( "u-", step, 5, ".tnl", fileName );
   u.write( "u", fileName.getFileName() );
   return true;
}

#ifdef HAVE_CUDA

template< typename Real, typename Index >
__global__ void boundaryConditionsKernel( Real* u,
                                          Real* fu,
                                          const Index gridXSize, const Index gridYSize )
{
   const Index i = ( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index j = ( blockIdx.y ) * blockDim.y + threadIdx.y;
   if( i == 0 && j < gridYSize )
   {
      fu[ j * gridXSize ] = 0.0;
      u[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
   }
   if( i == gridXSize - 1 && j < gridYSize )
   {
      fu[ j * gridXSize + gridYSize - 1 ] = 0.0;
      u[ j * gridXSize + gridYSize - 1 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];
   }
   if( j == 0 && i > 0 && i < gridXSize - 1 )
   {
      fu[ i ] = 0.0; //u[ j * gridXSize + 1 ];
      u[ i ] = 0.0; //u[ j * gridXSize + 1 ];
   }
   if( j == gridYSize -1  && i > 0 && i < gridXSize - 1 )
   {
      fu[ j * gridXSize + i ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];
      u[ j * gridXSize + i ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];
   }
}


template< typename Real, typename Index >
__global__ void heatEquationKernel( const Real* u,
                                    Real* fu,
                                    const Real tau,
                                    const Real hx_inv,
                                    const Real hy_inv,
                                    const Index gridXSize,
                                    const Index gridYSize )
{
   const Index i = blockIdx.x * blockDim.x + threadIdx.x;
   const Index j = blockIdx.y * blockDim.y + threadIdx.y;
   if( i > 0 && i < gridXSize - 1 &&
       j > 0 && j < gridYSize - 1 )
   {
      const Index c = j * gridXSize + i;
      fu[ c ] = ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv;
   }
}

template< typename GridType,
          typename GridEntity,
          typename BoundaryConditions,
          typename MeshFunction >
__global__ void
boundaryConditionsTemplatedCompact( const GridType* grid,
                                    const BoundaryConditions* boundaryConditions,
                                    MeshFunction* u,
                                    const typename GridType::RealType time,
                                    const typename GridEntity::CoordinatesType begin,
                                    const typename GridEntity::CoordinatesType end,
                                    const typename GridEntity::EntityOrientationType entityOrientation,
                                    const typename GridEntity::EntityBasisType entityBasis,
                                    const typename GridType::IndexType gridXIdx,
                                    const typename GridType::IndexType gridYIdx )
{
   typename GridType::CoordinatesType coordinates;

   coordinates.x() = begin.x() + ( gridXIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;

   if( coordinates.x() < end.x() &&
       coordinates.y() < end.y() )
   {
      GridEntity entity( *grid, coordinates, entityOrientation, entityBasis );
      entity.refresh();
      if( entity.isBoundaryEntity() )
      {
         ( *u )( entity ) = ( *boundaryConditions )( *u, entity, time );
      }
   }
}

/*template< typename EntityType, int Dimension >
struct EntityPointer : public EntityPointer< EntityType, Dimension - 1 >
{
   __device__ EntityPointer( const EntityType* ptr )
      : EntityPointer< EntityType, Dimension - 1 >( ptr ), pointer( ptr )
   {
   }

   const EntityType* pointer;
};

template< typename EntityType >
struct EntityPointer< EntityType, 0 >
{
   __device__ inline EntityPointer( const EntityType* ptr )
   :pointer( ptr )
   {
   }


   const EntityType* pointer;
};*/

/*template< typename GridType >
struct TestEntity
{
   typedef typename GridType::Cell::CoordinatesType CoordinatesType;

   __device__ inline TestEntity( const GridType& grid,
               const typename GridType::Cell::CoordinatesType& coordinates,
               const typename GridType::Cell::EntityOrientationType& orientation,
               const typename GridType::Cell::EntityBasisType& basis )
   : grid( grid ),
      coordinates( coordinates ),
      orientation( orientation ),
      basis( basis ),
      entityIndex( 0 ),
      ptr( &grid )
   {
   };

   const GridType& grid;

   EntityPointer< GridType, 2 > ptr;
   //TestEntity< GridType > *entity1, *entity2, *entity3;

   typename GridType::IndexType entityIndex;

   const typename GridType::Cell::CoordinatesType coordinates;
   const typename GridType::Cell::EntityOrientationType orientation;
   const typename GridType::Cell::EntityBasisType basis;
};*/

template< typename GridType,
          typename GridEntity,
          typename DifferentialOperator,
          typename RightHandSide,
          typename MeshFunction >
__global__ void
heatEquationTemplatedCompact( const GridType* grid,
                              const DifferentialOperator* differentialOperator,
                              const RightHandSide* rightHandSide,
                              MeshFunction* _u,
                              MeshFunction* _fu,
                              const typename GridType::RealType time,
                              const typename GridEntity::CoordinatesType begin,
                              const typename GridEntity::CoordinatesType end,
                              const typename GridEntity::EntityOrientationType entityOrientation,
                              const typename GridEntity::EntityBasisType entityBasis,
                              const typename GridType::IndexType gridXIdx,
                              const typename GridType::IndexType gridYIdx )
{
   typename GridType::CoordinatesType coordinates;
   typedef typename GridType::IndexType IndexType;
   typedef typename GridType::RealType RealType;

   coordinates.x() = begin.x() + ( gridXIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   coordinates.y() = begin.y() + ( gridYIdx * Cuda::getMaxGridSize() + blockIdx.y ) * blockDim.y + threadIdx.y;

   MeshFunction& u = *_u;
   MeshFunction& fu = *_fu;

   if( coordinates.x() < end.x() &&
       coordinates.y() < end.y() )
   {
      GridEntity entity( *grid, coordinates, entityOrientation, entityBasis );

      entity.refresh();
      if( ! entity.isBoundaryEntity() )
      {
         fu( entity ) =
            ( *differentialOperator )( u, entity, time );

         typedef Functions::FunctionAdapter< GridType, RightHandSide > FunctionAdapter;
         fu( entity ) +=  FunctionAdapter::getValue( *rightHandSide, entity, time );
      }
   }
}
#endif



template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
getExplicitUpdate( const RealType& time,
                   const RealType& tau,
                   DofVectorPointer& uDofs,
                   DofVectorPointer& fuDofs )
{
   /****
    * If you use an explicit solver like Euler or Merson, you
    * need to implement this method. Compute the right-hand side of
    *
    *   d/dt u(x) = fu( x, u )
    *
    * You may use supporting mesh dependent data if you need.
    */
   const MeshPointer& mesh = this->getMesh();
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      const IndexType gridXSize = mesh->getDimensions().x();
      const IndexType gridYSize = mesh->getDimensions().y();
      const RealType& hx_inv = mesh->template getSpaceStepsProducts< -2,  0 >();
      const RealType& hy_inv = mesh->template getSpaceStepsProducts<  0, -2 >();
      RealType* u = uDofs->getData();
      RealType* fu = fuDofs->getData();
      for( IndexType j = 0; j < gridYSize; j++ )
      {
         fu[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
         fu[ j * gridXSize + gridXSize - 2 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];
      }
      for( IndexType i = 0; i < gridXSize; i++ )
      {
         fu[ i ] = 0.0; //u[ gridXSize + i ];
         fu[ ( gridYSize - 1 ) * gridXSize + i ] = 0.0; //u[ ( gridYSize - 2 ) * gridXSize + i ];
      }

      /*typedef typename MeshType::Cell CellType;
      typedef typename CellType::CoordinatesType CoordinatesType;
      CoordinatesType coordinates( 0, 0 ), entityOrientation( 0,0 ), entityBasis( 0, 0 );*/

      //CellType entity( mesh, coordinates, entityOrientation, entityBasis );

      for( IndexType j = 1; j < gridYSize - 1; j++ )
         for( IndexType i = 1; i < gridXSize - 1; i++ )
         {
            const IndexType c = j * gridXSize + i;
            fu[ c ] = tau * ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                              ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
         }
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
      #ifdef HAVE_CUDA
      if( this->cudaKernelType == "pure-c" )
      {
         const IndexType gridXSize = mesh->getDimensions().x();
         const IndexType gridYSize = mesh->getDimensions().y();
         const RealType& hx_inv = mesh->template getSpaceStepsProducts< -2,  0 >();
         const RealType& hy_inv = mesh->template getSpaceStepsProducts<  0, -2 >();

         dim3 cudaBlockSize( 16, 16 );
         dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                            gridYSize / 16 + ( gridYSize % 16 != 0 ) );

         int cudaErr;

         heatEquationKernel<<< cudaGridSize, cudaBlockSize >>>
            ( uDofs->getData(), fuDofs->getData(), tau, hx_inv, hy_inv, gridXSize, gridYSize );
         if( cudaGetLastError() != cudaSuccess )
         {
            std::cerr << "Laplace operator failed." << std::endl;
            return;
         }

         boundaryConditionsKernel<<< cudaGridSize, cudaBlockSize >>>( uDofs->getData(), fuDofs->getData(), gridXSize, gridYSize );
         if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
         {
            std::cerr << "Setting of boundary conditions failed. " << cudaErr << std::endl;
            return;
         }

      }
      if( this->cudaKernelType == "templated-compact" )
      {
         typedef typename MeshType::EntityType< 2 > CellType;
         //typedef typename MeshType::Cell CellType;
         //std::cerr << "Size of entity is ... " << sizeof( TestEntity< MeshType > ) << " vs. " << sizeof( CellType ) << std::endl;
         typedef typename CellType::CoordinatesType CoordinatesType;
         u->bind( mesh, *uDofs );
         fu->bind( mesh, *fuDofs );
         fu->getData().setValue( 1.0 );
         const CoordinatesType begin( 0,0 );
         const CoordinatesType& end = mesh->getDimensions();
         CellType cell( mesh.template getData< DeviceType >() );
         dim3 cudaBlockSize( 16, 16 );
         dim3 cudaBlocks;
         cudaBlocks.x = Cuda::getNumberOfBlocks( end.x() - begin.x() + 1, cudaBlockSize.x );
         cudaBlocks.y = Cuda::getNumberOfBlocks( end.y() - begin.y() + 1, cudaBlockSize.y );
         const IndexType cudaXGrids = Cuda::getNumberOfGrids( cudaBlocks.x );
         const IndexType cudaYGrids = Cuda::getNumberOfGrids( cudaBlocks.y );

         //std::cerr << "Setting boundary conditions..." << std::endl;

         Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
         for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
            for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
               boundaryConditionsTemplatedCompact< MeshType, CellType, BoundaryCondition, MeshFunctionViewType >
                  <<< cudaBlocks, cudaBlockSize >>>
                  ( &mesh.template getData< Devices::Cuda >(),
                    &boundaryConditionPointer.template getData< Devices::Cuda >(),
                    &u.template modifyData< Devices::Cuda >(),
                    time,
                    begin,
                    end,
                    cell.getOrientation(),
                    cell.getBasis(),
                    gridXIdx,
                    gridYIdx );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;

         //std::cerr << "Computing the heat equation ..." << std::endl;
         for( IndexType gridYIdx = 0; gridYIdx < cudaYGrids; gridYIdx ++ )
            for( IndexType gridXIdx = 0; gridXIdx < cudaXGrids; gridXIdx ++ )
               heatEquationTemplatedCompact< MeshType, CellType, DifferentialOperator, RightHandSide, MeshFunctionViewType >
                  <<< cudaBlocks, cudaBlockSize >>>
                  ( &mesh.template getData< DeviceType >(),
                    &differentialOperatorPointer.template getData< DeviceType >(),
                    &rightHandSidePointer.template getData< DeviceType >(),
                    &u.template modifyData< DeviceType >(),
                    &fu.template modifyData< DeviceType >(),
                    time,
                    begin,
                    end,
                    cell.getOrientation(),
                    cell.getBasis(),
                    gridXIdx,
                    gridYIdx );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }
      if( this->cudaKernelType == "templated" )
      {
         //if( !this->cudaMesh )
         //   this->cudaMesh = tnlCuda::passToDevice( &mesh );
         this->u->bind( mesh, *uDofs );
         this->fu->bind( mesh, *fuDofs );
         //explicitUpdater.setGPUTransferTimer( this->gpuTransferTimer );
         this->explicitUpdater.template update< typename Mesh::Cell >( time, tau, mesh, this->u, this->fu );
      }
      if( this->cudaKernelType == "tunning" )
      {
         if( std::is_same< DeviceType, Devices::Cuda >::value )
         {
            this->u->bind( mesh, *uDofs );
            this->fu->bind( mesh, *fuDofs );


            /*this->explicitUpdater.template update< typename Mesh::Cell >( time, tau, mesh, this->u, this->fu );
            return;*/

#ifdef WITH_TNL
            using ExplicitUpdaterType = TNL::Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionViewType, DifferentialOperator, BoundaryCondition, RightHandSide >;
            using Cell = typename MeshType::Cell;
            using MeshTraverserType = Meshes::Traverser< MeshType, Cell >;
            using UserData = TNL::Solvers::PDE::ExplicitUpdaterTraverserUserData< RealType,
               MeshFunctionViewType,
               DifferentialOperator,
               BoundaryCondition,
               RightHandSide >;

#else
            //using CellConfig = Meshes::GridEntityNoStencilStorage;
            using CellConfig = Meshes::GridEntityCrossStencilStorage< 1 >;
            using ExplicitUpdaterType = ExplicitUpdater< Mesh, MeshFunctionViewType, DifferentialOperator, BoundaryCondition, RightHandSide >;
            using Cell = typename MeshType::Cell;
            //using Cell = SimpleCell< Mesh, CellConfig >;
            using MeshTraverserType = Traverser< MeshType, Cell >;
            using UserData = ExplicitUpdaterTraverserUserData< RealType,
               MeshFunctionViewType,
               DifferentialOperator,
               BoundaryCondition,
               RightHandSide >;
#endif

            using InteriorEntitiesProcessor = typename ExplicitUpdaterType::TraverserInteriorEntitiesProcessor;
            using BoundaryEntitiesProcessor = typename ExplicitUpdaterType::TraverserBoundaryEntitiesProcessor;

            UserData userData;
            userData.time = time;
            userData.differentialOperator = &this->differentialOperatorPointer.template getData< Devices::Cuda >();
            userData.boundaryConditions = &this->boundaryConditionPointer.template getData< Devices::Cuda >();
            userData.rightHandSide = &this->rightHandSidePointer.template getData< Devices::Cuda >();
            userData.u = &this->u.template modifyData< Devices::Cuda >(); //uDofs->getData();
            userData.fu = &this->fu.template modifyData< Devices::Cuda >(); //fuDofs->getData();
#ifndef WITH_TNL
            userData.real_u = uDofs->getData();
            userData.real_fu = fuDofs->getData();
#endif
            /*
            const IndexType gridXSize = mesh->getDimensions().x();
            const IndexType gridYSize = mesh->getDimensions().y();
            dim3 cudaBlockSize( 16, 16 );
            dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                               gridYSize / 16 + ( gridYSize % 16 != 0 ) );
            */

            Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
            int cudaErr;
            Meshes::Traverser< MeshType, Cell > meshTraverser;
            meshTraverser.template processInteriorEntities< InteriorEntitiesProcessor >
                                                          ( mesh,
                                                            userData );
             // */
            /*_heatEquationKernel< InteriorEntitiesProcessor, UserData, MeshType, RealType, IndexType >
            <<< cudaGridSize, cudaBlockSize >>>
               ( &mesh.template getData< Devices::Cuda >(),
                userData );
                //&userDataPtr.template modifyData< Devices::Cuda >() );*/
            if( cudaGetLastError() != cudaSuccess )
            {
               std::cerr << "Laplace operator failed." << std::endl;
               return;
            }

            meshTraverser.template processBoundaryEntities< BoundaryEntitiesProcessor >
                                                          ( mesh,
                                                            userData );
            // */
           /*_boundaryConditionsKernel< BoundaryEntitiesProcessor, UserData, MeshType, RealType, IndexType >
            <<< cudaGridSize, cudaBlockSize >>>
               ( &mesh.template getData< Devices::Cuda >(),
                userData );
                //&userDataPtr.template modifyData< Devices::Cuda >() );
            // */
            if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
            {
               std::cerr << "Setting of boundary conditions failed. " << cudaErr << std::endl;
               return;
            }



         }
      }
      #endif
   }
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
applyBoundaryConditions( const RealType& time,
                            DofVectorPointer& uDofs )
{
   if( this->cudaKernelType == "templated" )
   {
      this->bindDofs( uDofs );
      this->explicitUpdater.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time, this->u );
   }
   if( this->cudaKernelType == "tunning" )
   {
      /*
      return;
      this->bindDofs( uDofs );
      this->explicitUpdater.template applyBoundaryConditions< typename Mesh::Cell >( this->getMesh(), time, this->u );
      return;*/

#ifdef HAVE_CUDA
/*
#ifdef WITH_TNL
      using ExplicitUpdaterType = TNL::Solvers::PDE::ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide >;
      using Cell = typename MeshType::Cell;
      using MeshTraverserType = Meshes::Traverser< MeshType, Cell >;
      using UserData = TNL::Solvers::PDE::ExplicitUpdaterTraverserUserData< RealType,
         MeshFunctionType,
         DifferentialOperator,
         BoundaryCondition,
         RightHandSide >;

#else
      //using CellConfig = Meshes::GridEntityNoStencilStorage;
      using CellConfig = Meshes::GridEntityCrossStencilStorage< 1 >;
      using ExplicitUpdaterType = ExplicitUpdater< Mesh, MeshFunctionType, DifferentialOperator, BoundaryCondition, RightHandSide >;
      //using Cell = typename MeshType::Cell;
      using Cell = SimpleCell< Mesh, CellConfig >;
      using MeshTraverserType = Traverser< MeshType, Cell >;
      using UserData = ExplicitUpdaterTraverserUserData< RealType,
         MeshFunctionType,
         DifferentialOperator,
         BoundaryCondition,
         RightHandSide >;
#endif
         using InteriorEntitiesProcessor = typename ExplicitUpdaterType::TraverserInteriorEntitiesProcessor;
         using BoundaryEntitiesProcessor = typename ExplicitUpdaterType::TraverserBoundaryEntitiesProcessor;

         UserData userData;
         userData.time = time;
         userData.differentialOperator = &this->differentialOperatorPointer.template getData< Devices::Cuda >();
         userData.rightHandSide = &this->rightHandSidePointer.template getData< Devices::Cuda >();
         userData.u = &this->u.template modifyData< Devices::Cuda >(); //uDofs->getData();
#ifndef WITH_TNL
         userData.real_u = uDofs->getData();
#endif
      userData.boundaryConditions = &this->boundaryConditionPointer.template getData< Devices::Cuda >();
      Meshes::Traverser< MeshType, Cell > meshTraverser;
      const MeshPointer& mesh = this->getMesh();
      // */
      /*meshTraverser.template processBoundaryEntities< BoundaryEntitiesProcessor >
                                                    ( mesh,
                                                      userData );*/

      /*_boundaryConditionsKernel< BoundaryEntitiesProcessor, UserData, MeshType, RealType, IndexType >
      <<< cudaGridSize, cudaBlockSize >>>
         ( &mesh.template getData< Devices::Cuda >(),
          userData );
          //&userDataPtr.template modifyData< Devices::Cuda >() );
      // */
      int cudaErr;
      if( ( cudaErr = ::cudaGetLastError() ) != cudaSuccess )
      {
         std::cerr << "Setting of boundary conditions failed. " << cudaErr << std::endl;
         return;
      }
#endif
   }
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
   template< typename MatrixPointer >
void
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
assemblyLinearSystem( const RealType& time,
                      const RealType& tau,
                      DofVectorPointer& _u,
                      MatrixPointer& matrix,
                      DofVectorPointer& b )
{
   // TODO: the instance should be "cached" like this->explicitUpdater, but there is a problem with MatrixPointer
   Solvers::PDE::LinearSystemAssembler< Mesh,
                             MeshFunctionViewType,
                             DifferentialOperator,
                             BoundaryCondition,
                             RightHandSide,
                             Solvers::PDE::BackwardTimeDiscretisation,
                             typename DofVectorPointer::ObjectType > systemAssembler;

   MeshFunctionViewPointer u;
   u->bind( this->getMesh(), *_u );
   systemAssembler.setDifferentialOperator( this->differentialOperator );
   systemAssembler.setBoundaryConditions( this->boundaryCondition );
   systemAssembler.setRightHandSide( this->rightHandSide );
   systemAssembler.template assembly< typename Mesh::Cell >( time, tau, this->getMesh(), u, matrix, b );
}

template< typename Mesh,
          typename BoundaryCondition,
          typename RightHandSide,
          typename DifferentialOperator >
HeatEquationBenchmarkProblem< Mesh, BoundaryCondition, RightHandSide, DifferentialOperator >::
~HeatEquationBenchmarkProblem()
{
   if( this->cudaMesh ) Cuda::freeFromDevice( this->cudaMesh );
   if( this->cudaBoundaryConditions )  Cuda::freeFromDevice( this->cudaBoundaryConditions );
   if( this->cudaRightHandSide ) Cuda::freeFromDevice( this->cudaRightHandSide );
   if( this->cudaDifferentialOperator ) Cuda::freeFromDevice( this->cudaDifferentialOperator );
}


#endif /* HeatEquationBenchmarkPROBLEM_IMPL_H_ */
