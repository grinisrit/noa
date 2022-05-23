#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#ifdef HAVE_MPI

#include <TNL/Devices/Host.h>
#include <TNL/Functions/CutMeshFunction.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

#include "../../Functions/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Devices;

static const char* TEST_FILE_NAME = "test_CutDistributedMeshFunctionTest.tnl";

//======================================DATA===============================================================
TEST(CutDistributedMeshFunction, 2D_Data)
{
   typedef Grid<2, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,2> LinearFunctionType;

   MeshType globalOriginalGrid;
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   typename DistributedMeshType::CoordinatesType overlap;

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 3, 4 ) );
   distributedGrid.setGlobalGrid(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   Pointers::SharedPointer<MeshType> originalGrid;
   *originalGrid = globalOriginalGrid->getLocalMesh();

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0);

   Pointers::SharedPointer<MeshFunctionView<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunctionView<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   DistributedMeshSynchronizer< DistributedMeshType > synchronizer;
   synchronizer.setDistributedGrid( &distributedGrid );
   synchronizer.synchronize( *meshFunctionptr );

   //Prepare Mesh Function parts for Cut
   CutDistributedMeshType cutDistributedGrid;
   Pointers::SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<MeshFunctionView<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,typename CutMeshType::IndexType>(5) );

   if(inCut)
   {
       MeshFunctionView<CutMeshType> cutMeshFunction;
       cutMeshFunction.bind(cutGrid,cutDof);

        for(int i=0;i<originalGrid->getDimensions().y();i++)
        {
               typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
               typename CutMeshType::Cell outEntity(*cutGrid);

                fromEntity.getCoordinates().x()=5-distributedGrid.getGlobalBegin().x();
                fromEntity.getCoordinates().y()=i;
                outEntity.getCoordinates().x()=i;

                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<i  <<" Chyba";
        }

    }
}

TEST(CutDistributedMeshFunction, 3D_1_Data)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid;
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2, 2, 3 ) );
   distributedGrid.setGlobalGrid(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   Pointers::SharedPointer<MeshType> originalGrid;
   *originalGrid = globalOriginalGrid->getLocalMesh();

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0);

   Pointers::SharedPointer<MeshFunctionView<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunctionView<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   DistributedMeshSynchronizer< DistributedMeshType > synchronizer;
   synchronizer.setDistributedGrid( &distributedGrid );
   synchronizer.synchronize( *meshFunctionptr );

   //Prepare Mesh Function parts for Cut
   CutDistributedMeshType cutDistributedGrid;
   Pointers::SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<MeshFunctionView<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof,
            StaticVector<1,int>(2),
            StaticVector<2,int>(1,0),
            StaticVector<2,typename CutMeshType::IndexType>(3,4) );

   if(inCut)
   {
       MeshFunctionView<CutMeshType> cutMeshFunction;
       cutMeshFunction.bind(cutGrid,cutDof);

        for(int i=0;i<originalGrid->getDimensions().z();i++)
        {
               typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
               typename CutMeshType::Cell outEntity(*cutGrid);

                fromEntity.getCoordinates().x()=4-distributedGrid.getGlobalBegin().x();
                fromEntity.getCoordinates().y()=3-distributedGrid.getGlobalBegin().y();
                fromEntity.getCoordinates().z()=i;
                outEntity.getCoordinates().x()=i;

                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<i  <<" Chyba";
        }
    }
}

TEST(CutDistributedMeshFunction, 3D_2_Data)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<2, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid;
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2, 2, 3 ) );
   distributedGrid.setGlobalGrid(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   Pointers::SharedPointer<MeshType> originalGrid;
   *originalGrid = globalOriginalGrid->getLocalMesh();

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0);

   Pointers::SharedPointer<MeshFunctionView<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunctionView<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   DistributedMeshSynchronizer< DistributedMeshType > synchronizer;
   synchronizer.setDistributedGrid( &distributedGrid );
   synchronizer.synchronize( *meshFunctionptr );

   //Prepare Mesh Function parts for Cut
   CutDistributedMeshType cutDistributedGrid;
   Pointers::SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<MeshFunctionView<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof,
            StaticVector<2,int>(0,2),
            StaticVector<1,int>(1),
            StaticVector<1,typename CutMeshType::IndexType>(4) );

   if(inCut)
   {
       MeshFunctionView<CutMeshType> cutMeshFunction;
       cutMeshFunction.bind(cutGrid,cutDof);

        for(int i=0;i<originalGrid->getDimensions().z();i++)
        {
            for(int j=0;j<originalGrid->getDimensions().x();j++)
            {
               typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
               typename CutMeshType::Cell outEntity(*cutGrid);

                fromEntity.getCoordinates().x()=j;
                fromEntity.getCoordinates().y()=4-distributedGrid.getGlobalBegin().y();
                fromEntity.getCoordinates().z()=i;

                outEntity.getCoordinates().x()=j;
                outEntity.getCoordinates().y()=i;

                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<i  <<" Chyba";
            }
        }
    }
}

//================================Synchronization========================================================
TEST(CutDistributedMeshFunction, 2D_Synchronization)
{
   typedef Grid<2, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,2> LinearFunctionType;

   MeshType globalOriginalGrid;
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 3, 4 ) );
   distributedGrid.setGlobalGrid(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   Pointers::SharedPointer<MeshType> originalGrid;
   *originalGrid = globalOriginalGrid->getLocalMesh();

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0);

   Pointers::SharedPointer<MeshFunctionView<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunctionView<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut
   CutDistributedMeshType cutDistributedGrid;
   Pointers::SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<MeshFunctionView<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,typename CutMeshType::IndexType>(5) );

   if(inCut)
   {
       MeshFunctionView<CutMeshType> cutMeshFunction;
       cutMeshFunction.bind(cutGrid,cutDof);

        DistributedMeshSynchronizer< CutDistributedMeshType > synchronizer;
        synchronizer.setDistributedGrid( &cutDistributedGrid );
        synchronizer.synchronize( cutMeshFunction );

        typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
        typename CutMeshType::Cell outEntity(*cutGrid);

        fromEntity.getCoordinates().x()=5-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=0;
        outEntity.getCoordinates().x()=0;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Left overlap";

        fromEntity.getCoordinates().x()=5-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=(cutDof).getSize()-1;
        outEntity.getCoordinates().x()=(cutDof).getSize()-1;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Right overlap";

    }
}

TEST(CutDistributedMeshFunction, 3D_1_Synchronization)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid;
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2,2,3 ) );
   distributedGrid.setGlobalGrid( globalOriginalGrid );
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   Pointers::SharedPointer<MeshType> originalGrid;
   *originalGrid = globalOriginalGrid->getLocalMesh();

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0);

   Pointers::SharedPointer<MeshFunctionView<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunctionView<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut
   CutDistributedMeshType cutDistributedGrid;
   Pointers::SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<MeshFunctionView<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof,
            StaticVector<1,int>(1),
            StaticVector<2,int>(0,2),
            StaticVector<2,typename CutMeshType::IndexType>(4,4) );

   if(inCut)
   {
       MeshFunctionView<CutMeshType> cutMeshFunction;
       cutMeshFunction.bind(cutGrid,cutDof);

        DistributedMeshSynchronizer< CutDistributedMeshType > synchronizer;
        synchronizer.setDistributedGrid( &cutDistributedGrid );
        synchronizer.synchronize( cutMeshFunction );

        typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
        typename CutMeshType::Cell outEntity(*cutGrid);

        fromEntity.getCoordinates().x()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().z()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=0;
        outEntity.getCoordinates().x()=0;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Left overlap";

        fromEntity.getCoordinates().x()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().z()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=(cutDof).getSize()-1;
        outEntity.getCoordinates().x()=(cutDof).getSize()-1;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Right overlap";

    }
}

TEST(CutDistributedMeshFunction, 3D_2_Synchronization)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<2, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid;
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   typename DistributedMeshType::CoordinatesType overlap;
   overlap.setValue(1);
   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2,2,3 ) );
   distributedGrid.setGlobalGrid(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   Pointers::SharedPointer<MeshType> originalGrid;
   *originalGrid = globalOriginalGrid->getLocalMesh();

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0);

   Pointers::SharedPointer<MeshFunctionView<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunctionView<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut
   CutDistributedMeshType cutDistributedGrid;
   Pointers::SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<MeshFunctionView<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof,
            StaticVector<2,int>(0,2),
            StaticVector<1,int>(1),
            StaticVector<1,typename CutMeshType::IndexType>(4) );

   if(inCut)
   {
       MeshFunctionView<CutMeshType> cutMeshFunction;
       cutMeshFunction.bind(cutGrid,cutDof);

        DistributedMeshSynchronizer< CutDistributedMeshType > synchronizer;
        synchronizer.setDistributedGrid( &cutDistributedGrid );
        synchronizer.synchronize( cutMeshFunction );

        typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
        typename CutMeshType::Cell outEntity(*cutGrid);

        for(int i=0;i<distributedGrid.getLocalMeshSize().x();i++)
            for(int j=0;j<distributedGrid.getLocalMeshSize().z();j++)
            {
                fromEntity.getCoordinates().x()=i;
                fromEntity.getCoordinates().z()=j;
                fromEntity.getCoordinates().y()=4-distributedGrid.getGlobalBegin().y();
                outEntity.getCoordinates().x()=i;
                outEntity.getCoordinates().y()=j;
                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Left overlap";

            }
      }
}


//=========================================================================================================
TEST(CutDistributedMeshFunction, 3D_2_Save)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<2, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid;
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   typename DistributedMeshType::CoordinatesType overlap;
   overlap.setValue(1);
   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2,2,3 ) );
   distributedGrid.setGlobalGrid( globalOriginalGrid );
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the MeshFunction interface is fucked up (it should not require us to put SharedPointer everywhere)
   Pointers::SharedPointer<MeshType> originalGrid;
   *originalGrid = globalOriginalGrid->getLocalMesh();

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0);

   Pointers::SharedPointer<MeshFunctionView<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunctionView<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut
   CutDistributedMeshType cutDistributedGrid;
   Pointers::SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<MeshFunctionView<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof,
            StaticVector<2,int>(0,2),
            StaticVector<1,int>(1),
            StaticVector<1,typename CutMeshType::IndexType>(4) );

   if(inCut)
   {
       MeshFunctionView<CutMeshType> cutMeshFunction;
       cutMeshFunction.bind(cutGrid,cutDof);

       // FIXME: DistributedGridIO was removed
//        DistributedGridIO<MeshFunctionView<CutMeshType>,MpiIO> ::save(TEST_FILE_NAME, cutMeshFunction );

        //save globalgrid for debug render
        MPI_Comm communicator=cutDistributedGrid.getCommunicator();
        if(TNL::MPI::GetRank(communicator)==0)
        {
           // FIXME: save was removed from Grid (but this is just for debugging...)
//            File meshFile;
//            meshFile.open( TEST_FILE_NAME+String("-mesh.tnl"),std::ios_base::out);
//            cutDistributedGrid.getGlobalGrid().save( meshFile );
//            meshFile.close();
        }

    }

   if(TNL::MPI::GetRank()==0)
   {
       Pointers::SharedPointer<CutMeshType> globalCutGrid;
       MeshFunctionView<CutMeshType> loadMeshFunctionptr;

       globalCutGrid->setDimensions(typename CutMeshType::CoordinatesType(10));
       globalCutGrid->setDomain(typename CutMeshType::PointType(-0.5),typename CutMeshType::CoordinatesType(10));

       DofType loaddof(globalCutGrid->template getEntitiesCount< typename CutMeshType::Cell >());
       loaddof.setValue(0);
       loadMeshFunctionptr.bind(globalCutGrid,loaddof);

        File file;
        file.open( TEST_FILE_NAME, std::ios_base::in );
        loadMeshFunctionptr.boundLoad(file);
        file.close();

        typename MeshType::Cell fromEntity(globalOriginalGrid);
        typename CutMeshType::Cell outEntity(*globalCutGrid);

        for(int i=0;i<globalOriginalGrid.getDimensions().x();i++)
            for(int j=0;j<globalOriginalGrid.getDimensions().z();j++)
            {
                fromEntity.getCoordinates().x()=i;
                fromEntity.getCoordinates().z()=j;
                fromEntity.getCoordinates().y()=4;
                outEntity.getCoordinates().x()=i;
                outEntity.getCoordinates().y()=j;
                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(loadMeshFunctionptr.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error at "<< i <<" "<< j;

            }

        EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
        EXPECT_EQ( std::remove( (TEST_FILE_NAME+String("-mesh.tnl")).getString()) , 0 );
      }

}
#endif

#endif

#include "../../main_mpi.h"
