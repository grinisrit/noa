#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#ifdef HAVE_MPI

#include <experimental/filesystem>

#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/Writers/PVTIWriter.h>
#include <TNL/Meshes/Readers/PVTIReader.h>

#include "../../Functions/Functions.h"

namespace fs = std::experimental::filesystem;

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Functions;
using namespace TNL::Devices;


template<typename DofType>
void setDof_1D( DofType &dof, typename DofType::RealType value )
{
   for( int i = 0; i < dof.getSize(); i++ )
      dof[ i ] = value;
}

template<typename DofType>
void check_Boundary_1D(int rank, int nproc, const DofType& dof, typename DofType::RealType expectedValue)
{
    if(rank==0)//Left
    {
        EXPECT_EQ( dof[0], expectedValue) << "Left boundary test failed";
        return;
    }

    if(rank==(nproc-1))//Right
    {
        EXPECT_EQ( dof[dof.getSize()-1], expectedValue) << "Right boundary test failed";
        return;
    }
}

template<typename DofType>
void check_Overlap_1D(int rank, int nproc, const DofType& dof, typename DofType::RealType expectedValue)
{
    if( rank == 0 )//Left
    {
        EXPECT_EQ( dof[dof.getSize()-1], expectedValue) << "Left boundary node overlap test failed";
        return;
    }

    if( rank == ( nproc - 1 ) )
    {
        EXPECT_EQ( dof[0], expectedValue) << "Right boundary node overlap test failed";
        return;
    }

    EXPECT_EQ( dof[0], expectedValue) << "left overlap test failed";
    EXPECT_EQ( dof[dof.getSize()-1], expectedValue)<< "right overlap test failed";
}

template<typename DofType>
void check_Inner_1D(int rank, int nproc, const DofType& dof, typename DofType::RealType expectedValue)
{
    for( int i = 1; i < ( dof.getSize()-2 ); i++ )
        EXPECT_EQ( dof[i], expectedValue) << "i = " << i;
}

/*
 * Light check of 1D distributed grid and its synchronization.
 * Number of process is not limited.
 * Overlap is limited to 1
 * Only double is tested as dof Real type -- it may be changed, extend test
 * Global size is hardcoded as 10 -- it can be changed, extend test
 */

typedef Grid<1,double,Host,int> GridType;
typedef MeshFunctionView< GridType > MeshFunctionType;
typedef MeshFunctionView< GridType, GridType::getMeshDimension(), bool > MaskType;
typedef Vector< double,Host,int> DofType;
typedef Vector< bool, Host, int > MaskDofType;
typedef typename GridType::Cell Cell;
typedef typename GridType::IndexType IndexType;
typedef typename GridType::PointType PointType;
typedef DistributedMesh<GridType> DistributedGridType;
using Synchronizer = DistributedMeshSynchronizer< DistributedGridType >;

class DistributedGridTest_1D : public ::testing::Test
{
   protected:

      DistributedMesh< GridType > *distributedGrid;
      DofType dof;
      MaskDofType maskDofs;

      Pointers::SharedPointer< GridType > localGrid;
      Pointers::SharedPointer< MeshFunctionType > meshFunctionPtr;
      Pointers::SharedPointer< MaskType > maskPointer;

      MeshFunctionEvaluator< MeshFunctionType, ConstFunction< double, 1 > > constFunctionEvaluator;
      Pointers::SharedPointer< ConstFunction< double, 1 >, Host > constFunctionPtr;

      MeshFunctionEvaluator< MeshFunctionType, LinearFunction< double, 1 > > linearFunctionEvaluator;
      Pointers::SharedPointer< LinearFunction< double, 1 >, Host > linearFunctionPtr;

      int rank;
      int nproc;

      void SetUp()
      {
         int size=10;
         rank=TNL::MPI::GetRank();
         nproc=TNL::MPI::GetSize();

         PointType globalOrigin;
         PointType globalProportions;
         GridType globalGrid;

         globalOrigin.x()=-0.5;
         globalProportions.x()=size;

         globalGrid.setDimensions(size);
         globalGrid.setDomain(globalOrigin,globalProportions);

         distributedGrid=new DistributedGridType();
         typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
         distributedGrid->setGlobalGrid( globalGrid );
         SubdomainOverlapsGetter< GridType >::
            getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1 );
         distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

         // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
         // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
         *localGrid = distributedGrid->getLocalMesh();

         dof.setSize( localGrid->template getEntitiesCount< Cell >() );

         meshFunctionPtr->bind( localGrid, dof );

         constFunctionPtr->Number=rank;
      }

      void SetUpPeriodicBoundaries()
      {
         typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
         SubdomainOverlapsGetter< GridType >::
            getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1 );
         distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

         // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
         // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
         *localGrid = distributedGrid->getLocalMesh();
      }

      void TearDown()
      {
         delete distributedGrid;
      }
};

TEST_F( DistributedGridTest_1D, isBoundaryDomainTest )
{
   if( rank == 0 || rank == nproc - 1 )
      EXPECT_TRUE( distributedGrid->isBoundarySubdomain() );
   else
      EXPECT_FALSE( distributedGrid->isBoundarySubdomain() );
}

TEST_F(DistributedGridTest_1D, evaluateAllEntities)
{
   //Check Traversars
   //All entities, without overlap
   setDof_1D( dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   check_Boundary_1D(rank, nproc, dof, rank);
   check_Overlap_1D(rank, nproc, dof, -1);
   check_Inner_1D(rank, nproc, dof, rank);
}

TEST_F(DistributedGridTest_1D, evaluateBoundaryEntities)
{
   //Boundary entities, without overlap
   setDof_1D(dof,-1);
   constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionPtr , constFunctionPtr );
   check_Boundary_1D(rank, nproc, dof, rank);
   check_Overlap_1D(rank, nproc, dof, -1);
   check_Inner_1D(rank, nproc, dof, -1);
}

TEST_F(DistributedGridTest_1D, evaluateInteriorEntities)
{
   //Inner entities, without overlap
   setDof_1D(dof,-1);
   constFunctionEvaluator.evaluateInteriorEntities( meshFunctionPtr , constFunctionPtr );
   check_Boundary_1D(rank, nproc, dof, -1);
   check_Overlap_1D(rank, nproc, dof, -1);
   check_Inner_1D(rank, nproc, dof, rank);
}

TEST_F(DistributedGridTest_1D, SynchronizerNeighborsTest )
{
   setDof_1D(dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   Synchronizer synchronizer;
   synchronizer.setDistributedGrid( distributedGrid );
   synchronizer.synchronize( *meshFunctionPtr );

   if(rank!=0) {
      EXPECT_EQ((dof)[0],rank-1)<< "Left Overlap was filled by wrong process.";
   }
   if(rank!=nproc-1) {
      EXPECT_EQ((dof)[dof.getSize()-1],rank+1)<< "Right Overlap was filled by wrong process.";
   }
}

TEST_F(DistributedGridTest_1D, EvaluateLinearFunction )
{
   //fill mesh function with linear function (physical center of cell corresponds with its coordinates in grid)
   setDof_1D(dof,-1);
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionPtr, linearFunctionPtr);
   Synchronizer synchronizer;
   synchronizer.setDistributedGrid( distributedGrid );
   synchronizer.synchronize( *meshFunctionPtr );

   auto entity = localGrid->template getEntity< Cell >(0);
   entity.refresh();
   EXPECT_EQ(meshFunctionPtr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function Overlap error on left Edge.";

   auto entity2= localGrid->template getEntity< Cell >((dof).getSize()-1);
   entity2.refresh();
   EXPECT_EQ(meshFunctionPtr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function Overlap error on right Edge.";
}

TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithoutMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   *localGrid = distributedGrid->getLocalMesh();

   dof.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );

   setDof_1D( dof, -rank-1 );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   Synchronizer synchronizer;
   synchronizer.setDistributedGrid( distributedGrid );
   synchronizer.synchronize( *meshFunctionPtr, true );

   if( rank == 0 ) {
      EXPECT_EQ( dof[ 0 ], -nproc ) << "Left Overlap was filled by wrong process.";
   }
   if( rank == nproc-1 ) {
      EXPECT_EQ( dof[ dof.getSize() - 1 ], -1 )<< "Right Overlap was filled by wrong process.";
   }
}

TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithActiveMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   *localGrid = distributedGrid->getLocalMesh();

   dof.setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   setDof_1D( dof, -rank-1 );
   maskDofs.setValue( true );
   //constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr, constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   Synchronizer synchronizer;
   synchronizer.setDistributedGrid( distributedGrid );
   synchronizer.synchronize( *meshFunctionPtr, true, maskPointer );
   if( rank == 0 ) {
      EXPECT_EQ( dof[ 0 ], -nproc ) << "Left Overlap was filled by wrong process.";
   }
   if( rank == nproc-1 ) {
      EXPECT_EQ( dof[ dof.getSize() - 1 ], -1 )<< "Right Overlap was filled by wrong process.";
   }
}

// TODO: Fix tests with overlap-to-boundary direction and masks
/*
TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithInactiveMaskOnLeft )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   *localGrid = distributedGrid->getLocalMesh();

   dof.setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   setDof_1D( dof, -rank-1 );
   maskDofs.setValue( true );
   maskDofs.setElement( 1, false );
   //constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   TNL_MPI_PRINT( "#### " << dof );
   meshFunctionPtr->synchronize( true, maskPointer );
   TNL_MPI_PRINT( ">>> " << dof );

   if( rank == 0 )
      EXPECT_EQ( dof[ 0 ], 0 ) << "Left Overlap was filled by wrong process.";
   if( rank == nproc-1 )
      EXPECT_EQ( dof[ dof.getSize() - 1 ], -1 )<< "Right Overlap was filled by wrong process.";
}

TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithInactiveMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   *localGrid = distributedGrid->getLocalMesh();

   dof.setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   setDof_1D( dof, -rank-1 );
   maskDofs.setValue( true );
   maskDofs.setElement( 1, false );
   maskDofs.setElement( dof.getSize() - 2, false );
   //constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->synchronize( true, maskPointer );

   if( rank == 0 )
      EXPECT_EQ( dof[ 0 ], 0 ) << "Left Overlap was filled by wrong process.";
   if( rank == nproc-1 )
      EXPECT_EQ( dof[ dof.getSize() - 1 ], nproc - 1 )<< "Right Overlap was filled by wrong process.";

}
*/

TEST_F(DistributedGridTest_1D, SynchronizePeriodicBoundariesLinearTest )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   *localGrid = distributedGrid->getLocalMesh();

   dof.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );

   setDof_1D(dof, -rank-1 );
   linearFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , linearFunctionPtr );

   Synchronizer synchronizer;
   synchronizer.setDistributedGrid( distributedGrid );
   synchronizer.synchronize( *meshFunctionPtr, true );

   auto entity = localGrid->template getEntity< Cell >( 0 );
   auto entity2= localGrid->template getEntity< Cell >( (dof).getSize() - 1 );
   entity.refresh();
   entity2.refresh();

   if( rank == 0 ) {
      EXPECT_EQ( meshFunctionPtr->getValue(entity), 9 ) << "Linear function Overlap error on left Edge.";
   }
   if( rank == nproc - 1 ) {
      EXPECT_EQ( meshFunctionPtr->getValue(entity2), 0 ) << "Linear function Overlap error on right Edge.";
   }
}

TEST_F(DistributedGridTest_1D, PVTIWriterReader)
{
   // create a .pvti file (only rank 0 actually writes to the file)
   const std::string baseName = "DistributedGridTest_1D_" + std::to_string(nproc) + "proc";
   const std::string mainFilePath = baseName + ".pvti";
   std::string subfilePath;
   {
      std::ofstream file;
      if( TNL::MPI::GetRank() == 0 )
         file.open( mainFilePath );
      using PVTI = Meshes::Writers::PVTIWriter< GridType >;
      PVTI pvti( file );
      pvti.writeImageData( *distributedGrid );
      // TODO
//      if( mesh.getGhostLevels() > 0 ) {
//         pvti.template writePPointData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
//         pvti.template writePCellData< std::uint8_t >( Meshes::VTK::ghostArrayName() );
//      }
      subfilePath = pvti.addPiece( mainFilePath, *distributedGrid );

      // create a .vti file for local data
      using Writer = Meshes::Writers::VTIWriter< GridType >;
      std::ofstream subfile( subfilePath );
      Writer writer( subfile );
      writer.writeImageData( *localGrid );
      // TODO
//      if( mesh.getGhostLevels() > 0 ) {
//         writer.writePointData( mesh.vtkPointGhostTypes(), Meshes::VTK::ghostArrayName() );
//         writer.writeCellData( mesh.vtkCellGhostTypes(), Meshes::VTK::ghostArrayName() );
//      }

      // end of scope closes the files
   }

   // load and test
   TNL::MPI::Barrier();
   Readers::PVTIReader reader( mainFilePath );
   reader.detectMesh();
   EXPECT_EQ( reader.getMeshType(), "Meshes::DistributedGrid" );
   DistributedMesh< GridType > loadedGrid;
   reader.loadMesh( loadedGrid );
   EXPECT_EQ( loadedGrid, *distributedGrid );

   // cleanup
   EXPECT_EQ( fs::remove( subfilePath ), true );
   TNL::MPI::Barrier();
   if( TNL::MPI::GetRank() == 0 ) {
      EXPECT_EQ( fs::remove( mainFilePath ), true );
      EXPECT_EQ( fs::remove( baseName ), true );
   }
}

TEST_F(DistributedGridTest_1D, readDistributedMeshFunction)
{
   const std::string baseName = "DistributedGridTest_MeshFunction_1D_" + std::to_string(nproc) + "proc.pvti";
   const std::string mainFilePath = baseName + ".pvti";

   // evaluate a function
   dof.setValue( -1 );
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr, constFunctionPtr );

   // write the mesh function into a .pvti file
   EXPECT_TRUE( writeDistributedMeshFunction( *distributedGrid, *meshFunctionPtr, "foo", mainFilePath ) );

   // wait for rank 0 to write the main .pvti file
   TNL::MPI::Barrier();

   // load the mesh function from the .pvti file
   DofType loadedDof;
   loadedDof.setLike( dof );
   loadedDof.setValue( -2 );
   MeshFunctionType loadedMeshFunction;
   loadedMeshFunction.bind( localGrid, loadedDof );
   EXPECT_TRUE( readDistributedMeshFunction( *distributedGrid, loadedMeshFunction, "foo", mainFilePath ) );

   // compare the dofs (MeshFunction and MeshFunctionView do not have operator==)
//   EXPECT_EQ( loadedMeshFunction, *meshFunctionPtr );
   EXPECT_EQ( loadedDof, dof );

   // cleanup
   TNL::MPI::Barrier();
   if( TNL::MPI::GetRank() == 0 ) {
      EXPECT_TRUE( fs::remove( mainFilePath ) );
      EXPECT_GT( fs::remove_all( baseName ), 1u );
   }
}
#endif

#endif

#include "../../main_mpi.h"
