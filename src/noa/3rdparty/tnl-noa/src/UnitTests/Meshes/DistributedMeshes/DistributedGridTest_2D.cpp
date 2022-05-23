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
void setDof_2D( DofType &dof, typename DofType::RealType value )
{
   for( int i = 0; i < dof.getSize(); i++ )
      dof[ i ] = value;
}

template<typename DofType,typename GridType>
void checkLeftEdge( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxy;
    if( !with_first ) begin++;
    if( !with_last ) end--;

    for( int i=begin;i<end;i++ )
            EXPECT_EQ( dof[maxx*i], expectedValue) << "Left Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkRightEdge(const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxy;
    if( !with_first ) begin++;
    if( !with_last ) end--;

    for( int i = begin; i < end; i++ )
            EXPECT_EQ( dof[maxx*i+(maxx-1)], expectedValue) << "Right Edge test failed " << i <<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkUpEdge( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxx;
    if( !with_first ) begin++;
    if( !with_last ) end--;

    for( int i=begin; i<end; i++ )
            EXPECT_EQ( dof[i], expectedValue) << "Up Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkDownEdge( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
    int maxx = grid.getDimensions().x();
    int maxy = grid.getDimensions().y();
    int begin = 0;
    int end = maxx;
    if( !with_first ) begin++;
    if( !with_last ) end--;

    for( int i=begin; i<end; i++ )
            EXPECT_EQ( dof[maxx*(maxy-1)+i], expectedValue) << "Down Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkLeftBoundary( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxy - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;

   for( int i=begin;i<end;i++ )
      EXPECT_EQ( dof[ maxx * i + 1 ], expectedValue) << "Left Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkRightBoundary(const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue)
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxy - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;

   for( int i = begin; i < end; i++ )
     EXPECT_EQ( dof[ maxx * i + ( maxx - 2 ) ], expectedValue) << "Right Edge test failed " << i <<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkUpBoundary( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxx - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;

   for( int i=begin; i<end; i++ )
      EXPECT_EQ( dof[ maxx + i ], expectedValue) << "Up Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkDownBoundary( const GridType &grid, const DofType &dof, bool with_first, bool with_last, typename DofType::RealType expectedValue )
{
   int maxx = grid.getDimensions().x();
   int maxy = grid.getDimensions().y();
   int begin = 1;
   int end = maxx - 1;
   if( !with_first ) begin++;
   if( !with_last ) end--;

   for( int i=begin; i<end; i++ )
      EXPECT_EQ( dof[ maxx * ( maxy-2 ) + i ], expectedValue) << "Down Edge test failed " << i<<" " << maxx << " "<< maxy;
}

template<typename DofType,typename GridType>
void checkCorner(const GridType &grid, const DofType &dof, bool up, bool left, typename DofType::RealType expectedValue )
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    if(up&&left)
    {
        EXPECT_EQ( dof[0], expectedValue) << "Up Left Conner test failed ";
    }
    if(up && !left)
    {
        EXPECT_EQ( dof[maxx-1], expectedValue) << "Up Right Conner test failed ";
    }
    if(!up && left)
    {
        EXPECT_EQ( dof[(maxy-1)*maxx], expectedValue) << "Down Left Conner test failed ";
    }
    if(!up && !left)
    {
        EXPECT_EQ( dof[(maxy-1)*maxx+maxx-1], expectedValue) << "Down right Conner test failed ";
    }
}


/*expecting 9 processes*/
template<typename DofType,typename GridType>
void check_Boundary_2D(int rank, const GridType &grid, const DofType &dof, typename DofType::RealType expectedValue)
{
    if(rank==0)//Up Left
    {
        checkUpEdge(grid,dof,true,false,expectedValue);//posledni je overlap
        checkLeftEdge(grid,dof,true,false,expectedValue);//posledni je overlap
    }

    if(rank==1)//Up Center
    {
        checkUpEdge(grid,dof,false,false,expectedValue);//prvni a posledni je overlap
    }

    if(rank==2)//Up Right
    {
        checkUpEdge(grid,dof,false,true,expectedValue);//prvni je overlap
        checkRightEdge(grid,dof,true,false,expectedValue);//posledni je overlap
    }

    if(rank==3)//Center Left
    {
        checkLeftEdge(grid,dof,false,false,expectedValue);//prvni a posledni je overlap
    }

    if(rank==4)//Center Center
    {
        //No boundary
    }

    if(rank==5)//Center Right
    {
        checkRightEdge(grid,dof,false,false,expectedValue);
    }

    if(rank==6)//Down Left
    {
        checkDownEdge(grid,dof,true,false,expectedValue);
        checkLeftEdge(grid,dof,false,true,expectedValue);
    }

    if(rank==7) //Down Center
    {
        checkDownEdge(grid,dof,false,false,expectedValue);
    }

    if(rank==8) //Down Right
    {
        checkDownEdge(grid,dof,false,true,expectedValue);
        checkRightEdge(grid,dof,false,true,expectedValue);
    }
}

/*expecting 9 processes
 * Known BUG of Traversars: Process boundary is writing over overlap.
 * it should be true, true, every where, but we dont check boundary overalp on boundary
 * so boundary overlap is not checked (it is filled incorectly by boundary condition).
 */
template<typename DofType,typename GridType>
void check_Overlap_2D(int rank, const GridType &grid, const DofType &dof, typename DofType::RealType expectedValue)
{
    if(rank==0)//Up Left
    {
        checkRightEdge(grid,dof,false,true,expectedValue);
        checkDownEdge(grid,dof,false,true,expectedValue);
    }

    if(rank==1)//Up Center
    {
        checkDownEdge(grid,dof,true,true,expectedValue);
        checkLeftEdge(grid,dof,false,true,expectedValue);
        checkRightEdge(grid,dof,false,true,expectedValue);
    }

    if(rank==2)//Up Right
    {
        checkDownEdge(grid,dof,true,false,expectedValue);//prvni je overlap
        checkLeftEdge(grid,dof,false,true,expectedValue);
    }

    if(rank==3)//Center Left
    {
        checkUpEdge(grid,dof,false,true,expectedValue);
        checkDownEdge(grid,dof,false,true,expectedValue);
        checkRightEdge(grid,dof,true,true,expectedValue);
    }

    if(rank==4)//Center Center
    {
        checkUpEdge(grid,dof,true,true,expectedValue);
        checkDownEdge(grid,dof,true,true,expectedValue);
        checkRightEdge(grid,dof,true,true,expectedValue);
        checkLeftEdge(grid,dof,true,true,expectedValue);
    }

    if(rank==5)//Center Right
    {
        checkUpEdge(grid,dof,true,false,expectedValue);
        checkDownEdge(grid,dof,true,false,expectedValue);
        checkLeftEdge(grid,dof,true,true,expectedValue);
    }

    if(rank==6)//Down Left
    {
        checkUpEdge(grid,dof,false,true,expectedValue);
        checkRightEdge(grid,dof,true,false,expectedValue);
    }

    if(rank==7) //Down Center
    {
        checkUpEdge(grid,dof,true,true,expectedValue);
        checkLeftEdge(grid,dof,true,false,expectedValue);
        checkRightEdge(grid,dof,true,false,expectedValue);
    }

    if(rank==8) //Down Right
    {
        checkUpEdge(grid,dof,true,false,expectedValue);
        checkLeftEdge(grid,dof,true,false,expectedValue);
    }
}



template<typename DofType,typename GridType>
void check_Inner_2D(int rank, const GridType& grid, const DofType& dof, typename DofType::RealType expectedValue)
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    for(int j=1;j<maxy-1;j++)//prvni a posledni jsou buď hranice, nebo overlap
        for(int i=1;i<maxx-1;i++) //buď je vlevo hranice, nebo overlap
            EXPECT_EQ( dof[j*maxx+i], expectedValue) << " "<< j<<" "<<i << " " << maxx << " " << maxy;
}

/*
 * Light check of 2D distributed grid and its synchronization.
 * expected 9 processes
 */
typedef Grid<2,double,Host,int> GridType;
typedef MeshFunctionView<GridType> MeshFunctionType;
typedef MeshFunctionView< GridType, GridType::getMeshDimension(), bool > MaskType;
typedef Vector<double,Host,int> DofType;
typedef Vector< bool, Host, int > MaskDofType;
typedef typename GridType::Cell Cell;
typedef typename GridType::IndexType IndexType;
typedef typename GridType::PointType PointType;
typedef DistributedMesh<GridType> DistributedGridType;
using Synchronizer = DistributedMeshSynchronizer< DistributedGridType >;

class DistributedGridTest_2D : public ::testing::Test
{
   public:

      using CoordinatesType = typename GridType::CoordinatesType;

      DistributedGridType *distributedGrid;
      DofType dof;
      MaskDofType maskDofs;

      Pointers::SharedPointer<GridType> localGrid;
      Pointers::SharedPointer<MeshFunctionType> meshFunctionPtr;
      Pointers::SharedPointer< MaskType > maskPointer;

      MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,2> > constFunctionEvaluator;
      Pointers::SharedPointer< ConstFunction<double,2>, Host > constFunctionPtr;

      MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,2> > linearFunctionEvaluator;
      Pointers::SharedPointer< LinearFunction<double,2>, Host > linearFunctionPtr;

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
         globalOrigin.y()=-0.5;
         globalProportions.x()=size;
         globalProportions.y()=size;

         globalGrid.setDimensions(size,size);
         globalGrid.setDomain(globalOrigin,globalProportions);

         distributedGrid=new DistributedGridType();
         distributedGrid->setDomainDecomposition( typename DistributedGridType::CoordinatesType( 3, 3 ) );
         distributedGrid->setGlobalGrid( globalGrid );
         typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
         SubdomainOverlapsGetter< GridType >::
            getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1 );
         distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

         // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
         // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
         *localGrid = distributedGrid->getLocalMesh();

         dof.setSize( localGrid->template getEntitiesCount< Cell >() );

         meshFunctionPtr->bind(localGrid,dof);

         constFunctionPtr->Number=rank;
      }

      void TearDown()
      {
         delete distributedGrid;
      }
};

TEST_F(DistributedGridTest_2D, evaluateAllEntities)
{
   //Check Traversars
   //All entities, without overlap
   setDof_2D(dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   check_Boundary_2D(rank, *localGrid, dof, rank);
   check_Overlap_2D(rank, *localGrid, dof, -1);
   check_Inner_2D(rank, *localGrid, dof, rank);
}


TEST_F(DistributedGridTest_2D, evaluateBoundaryEntities)
{
    //Boundary entities, without overlap
    setDof_2D(dof,-1);
    constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionPtr , constFunctionPtr );
    check_Boundary_2D(rank, *localGrid, dof, rank);
    check_Overlap_2D(rank, *localGrid, dof, -1);
    check_Inner_2D(rank, *localGrid, dof, -1);
}

TEST_F(DistributedGridTest_2D, evaluateInteriorEntities)
{
    //Inner entities, without overlap
    setDof_2D(dof,-1);
    constFunctionEvaluator.evaluateInteriorEntities( meshFunctionPtr , constFunctionPtr );
    check_Boundary_2D(rank, *localGrid, dof, -1);
    check_Overlap_2D(rank, *localGrid, dof, -1);
    check_Inner_2D(rank, *localGrid, dof, rank);
}

TEST_F(DistributedGridTest_2D, LinearFunctionTest)
{
    //fill meshfunction with linear function (physical center of cell corresponds with its coordinates in grid)
    setDof_2D(dof,-1);
    linearFunctionEvaluator.evaluateAllEntities(meshFunctionPtr, linearFunctionPtr);
    Synchronizer synchronizer;
    synchronizer.setDistributedGrid( distributedGrid );
    synchronizer.synchronize( *meshFunctionPtr );

    int count =localGrid->template getEntitiesCount< Cell >();
    for(int i=0;i<count;i++)
    {
            auto entity= localGrid->template getEntity< Cell >(i);
            entity.refresh();
            EXPECT_EQ(meshFunctionPtr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function doesnt fit recievd data. " << entity.getCoordinates().x() << " "<<entity.getCoordinates().y() << " "<< localGrid->getDimensions().x() <<" "<<localGrid->getDimensions().y();
    }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborTest )
{
   //Expect 9 processes
   setDof_2D(dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   Synchronizer synchronizer;
   synchronizer.setDistributedGrid( distributedGrid );
   synchronizer.synchronize( *meshFunctionPtr );

   // checkNeighbor_2D(rank, *localGrid, dof);

    if(rank==0)//Up Left
    {
        checkRightEdge(*localGrid, dof, true,  false, 1 );
        checkDownEdge( *localGrid, dof, true,  false, 3 );
        checkCorner(   *localGrid, dof, false, false, 4 );
    }

    if(rank==1)//Up Center
    {
        checkLeftEdge( *localGrid, dof, true,  false, 0 );
        checkRightEdge(*localGrid, dof, true,  false, 2 );
        checkCorner(   *localGrid, dof, false, true,  3 );
        checkDownEdge( *localGrid, dof, false, false, 4 );
        checkCorner(   *localGrid, dof, false, false, 5 );
    }

    if(rank==2)//Up Right
    {
        checkLeftEdge( *localGrid, dof, true,  false, 1 );
        checkCorner(   *localGrid, dof, false, true,  4 );
        checkDownEdge( *localGrid, dof, false, true,  5 );
    }

    if(rank==3)//Center Left
    {
        checkUpEdge(    *localGrid, dof, true,  false, 0 );
        checkCorner(    *localGrid, dof, true,  false, 1 );
        checkRightEdge( *localGrid, dof, false, false, 4 );
        checkDownEdge(  *localGrid, dof, true,  false, 6 );
        checkCorner(    *localGrid, dof, false, false, 7 );
    }

    if(rank==4)//Center Center
    {
        checkCorner(    *localGrid, dof, true,  true,  0 );
        checkUpEdge(    *localGrid, dof, false, false, 1 );
        checkCorner(    *localGrid, dof, true,  false, 2 );
        checkLeftEdge(  *localGrid, dof, false, false, 3 );
        checkRightEdge( *localGrid, dof, false, false, 5 );
        checkCorner(    *localGrid, dof, false, true,  6 );
        checkDownEdge(  *localGrid, dof, false, false, 7 );
        checkCorner(    *localGrid, dof, false, false, 8 );
    }

    if(rank==5)//Center Right
    {
        checkCorner(   *localGrid, dof, true,  true,  1 );
        checkUpEdge(   *localGrid, dof, false, true,  2 );
        checkLeftEdge( *localGrid, dof, false, false, 4 );
        checkCorner(   *localGrid, dof, false, true,  7 );
        checkDownEdge( *localGrid, dof, false, true,  8 );
    }

    if(rank==6)//Down Left
    {
        checkUpEdge(    *localGrid, dof, true,  false, 3 );
        checkCorner(    *localGrid, dof, true,  false, 4 );
        checkRightEdge( *localGrid, dof, false, true,  7 );
    }

    if(rank==7) //Down Center
    {
        checkCorner(    *localGrid, dof, true,  true,  3 );
        checkUpEdge(    *localGrid, dof, false, false, 4 );
        checkCorner(    *localGrid, dof, true,  false, 5 );
        checkLeftEdge(  *localGrid, dof, false, true,  6 );
        checkRightEdge( *localGrid, dof, false, true,  8 );
    }

    if(rank==8) //Down Right
    {
        checkCorner(   *localGrid, dof, true,  true, 4 );
        checkUpEdge(   *localGrid, dof, false, true, 5 );
        checkLeftEdge( *localGrid, dof, false, true, 7 );
    }
}

// TODO: Fix tests for periodic BC -
// checkLeftBoundary -> checkLeft Overlap etc. for direction BoundaryToOverlap
// Fix the tests with mask to work with the direction OverlapToBoundary
/*
TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithoutMask )
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

   dof->setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );

   //Expecting 9 processes
   setDof_2D(dof, -rank-1 );
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->synchronize( true );

   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *localGrid, dof, false,  true, -3 );
      checkUpBoundary(   *localGrid, dof, false,  true, -7 );
   }

   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *localGrid, dof, true, true, -8 );
   }

   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *localGrid, dof, false, true, -1 );
      checkUpBoundary(    *localGrid, dof, true, false, -9 );
   }

   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *localGrid, dof, true, true, -6 );
   }

   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *localGrid, dof, true, true, -4 );
   }

   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *localGrid, dof, false,  true, -1 );
      checkLeftBoundary( *localGrid, dof, true,  false,  -9 );
   }

   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *localGrid, dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *localGrid, dof, true, false, -3 );
      checkRightBoundary( *localGrid, dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithActiveMask )
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

   dof->setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   //Expecting 9 processes
   setDof_2D(dof, -rank-1 );
   maskDofs.setValue( true );
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->synchronize( true, maskPointer );

   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *localGrid, dof, false,  true, -3 );
      checkUpBoundary(   *localGrid, dof, false,  true, -7 );
   }

   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *localGrid, dof, true, true, -8 );
   }

   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *localGrid, dof, false, true, -1 );
      checkUpBoundary(    *localGrid, dof, true, false, -9 );
   }

   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *localGrid, dof, true, true, -6 );
   }

   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *localGrid, dof, true, true, -4 );
   }

   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *localGrid, dof, false,  true, -1 );
      checkLeftBoundary( *localGrid, dof, true,  false,  -9 );
   }

   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *localGrid, dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *localGrid, dof, true, false, -3 );
      checkRightBoundary( *localGrid, dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInactiveMaskOnLeft )
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

   dof->setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   //Expecting 9 processes
   setDof_2D(dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYzXm ] == -1 )
   {
      for( IndexType i = 0; i < localGrid->getDimensions().y(); i++ )
      {
         typename GridType::Cell cell( *localGrid );
         cell.getCoordinates() = CoordinatesType( 1, i );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->synchronize( true, maskPointer );

   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *localGrid, dof, false,  true, 0 );
      checkUpBoundary(   *localGrid, dof, false,  true, -7 );
   }

   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *localGrid, dof, true, true, -8 );
   }

   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *localGrid, dof, false, true, -1 );
      checkUpBoundary(    *localGrid, dof, true, false, -9 );
   }

   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *localGrid, dof, true, true, 3 );
   }

   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *localGrid, dof, true, true, -4 );
   }

   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *localGrid, dof, false,  true, -1 );
      checkLeftBoundary( *localGrid, dof, true,  false,  6 );
   }

   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *localGrid, dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *localGrid, dof, true, false, -3 );
      checkRightBoundary( *localGrid, dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInActiveMaskOnRight )
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

   dof->setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   //Expecting 9 processes
   setDof_2D(dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYzXp ] == -1 )
   {
      for( IndexType i = 0; i < localGrid->getDimensions().y(); i++ )
      {
         typename GridType::Cell cell( *localGrid );
         cell.getCoordinates() = CoordinatesType( localGrid->getDimensions().x() - 2, i );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->synchronize( true, maskPointer );

   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *localGrid, dof, false,  true, -3 );
      checkUpBoundary(   *localGrid, dof, false,  true, -7 );
   }

   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *localGrid, dof, true, true, -8 );
   }

   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *localGrid, dof, false, true, 2 );
      checkUpBoundary(    *localGrid, dof, true, false, -9 );
   }

   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *localGrid, dof, true, true, -6 );
   }

   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *localGrid, dof, true, true, 5 );
   }

   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *localGrid, dof, false,  true, -1 );
      checkLeftBoundary( *localGrid, dof, true,  false,  -9 );
   }

   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *localGrid, dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *localGrid, dof, true, false, -3 );
      checkRightBoundary( *localGrid, dof, true, false, 8 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInActiveMaskUp )
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

   dof->setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   //Expecting 9 processes
   setDof_2D(dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYmXz ] == -1 )
   {
      for( IndexType i = 0; i < localGrid->getDimensions().x(); i++ )
      {
         typename GridType::Cell cell( *localGrid );
         cell.getCoordinates() = CoordinatesType( i, 1 );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->synchronize( true, maskPointer );

   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *localGrid, dof, false,  true, -3 );
      checkUpBoundary(   *localGrid, dof, false,  true, 0 );
   }

   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *localGrid, dof, true, true, 1 );
   }

   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *localGrid, dof, false, true, -1 );
      checkUpBoundary(    *localGrid, dof, true, false, 2 );
   }

   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *localGrid, dof, true, true, -6 );
   }

   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *localGrid, dof, true, true, -4 );
   }

   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *localGrid, dof, false,  true, -1 );
      checkLeftBoundary( *localGrid, dof, true,  false,  -9 );
   }

   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *localGrid, dof, true, true, -2 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *localGrid, dof, true, false, -3 );
      checkRightBoundary( *localGrid, dof, true, false, -7 );
   }
}

TEST_F(DistributedGridTest_2D, SynchronizerNeighborPeriodicBoundariesWithInActiveMaskDown )
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

   dof->setSize( localGrid->template getEntitiesCount< Cell >() );
   maskDofs.setSize( localGrid->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( localGrid, dof );
   maskPointer->bind( localGrid, maskDofs );

   //Expecting 9 processes
   setDof_2D(dof, -rank-1 );
   maskDofs.setValue( true );
   if( distributedGrid->getNeighbors()[ ZzYpXz ] == -1 )
   {
      for( IndexType i = 0; i < localGrid->getDimensions().x(); i++ )
      {
         typename GridType::Cell cell( *localGrid );
         cell.getCoordinates() = CoordinatesType( i, localGrid->getDimensions().y() - 2 );
         cell.refresh();
         maskPointer->getData().setElement( cell.getIndex(), false );
      }
   }
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->synchronize( true, maskPointer );

   if( rank == 0 )
   {
      SCOPED_TRACE( "Up Left" );
      checkLeftBoundary( *localGrid, dof, false,  true, -3 );
      checkUpBoundary(   *localGrid, dof, false,  true, -7 );
   }

   if( rank == 1 )
   {
      SCOPED_TRACE( "Up Center" );
      checkUpBoundary( *localGrid, dof, true, true, -8 );
   }

   if( rank == 2 )
   {
      SCOPED_TRACE( "Up Right" );
      checkRightBoundary( *localGrid, dof, false, true, -1 );
      checkUpBoundary(    *localGrid, dof, true, false, -9 );
   }

   if( rank == 3 )
   {
      SCOPED_TRACE( "Center Left" );
      checkLeftBoundary( *localGrid, dof, true, true, -6 );
   }

   if( rank == 5 )
   {
      SCOPED_TRACE( "Center Right" );
      checkRightBoundary( *localGrid, dof, true, true, -4 );
   }

   if( rank == 6 )
   {
      SCOPED_TRACE( "Down Left" );
      checkDownBoundary( *localGrid, dof, false,  true, 6 );
      checkLeftBoundary( *localGrid, dof, true,  false,  -9 );
   }

   if( rank == 7 )
   {
      SCOPED_TRACE( "Down Center" );
      checkDownBoundary( *localGrid, dof, true, true, 7 );
   }

   if( rank == 8 )
   {
      SCOPED_TRACE( "Down Right" );
      checkDownBoundary(  *localGrid, dof, true, false, 8 );
      checkRightBoundary( *localGrid, dof, true, false, -7 );
   }
}
*/

TEST_F(DistributedGridTest_2D, PVTIWriterReader)
{
   // create a .pvti file (only rank 0 actually writes to the file)
   const std::string baseName = "DistributedGridTest_2D_" + std::to_string(nproc) + "proc";
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

TEST_F(DistributedGridTest_2D, readDistributedMeshFunction)
{
   const std::string baseName = "DistributedGridTest_MeshFunction_2D_" + std::to_string(nproc) + "proc.pvti";
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
