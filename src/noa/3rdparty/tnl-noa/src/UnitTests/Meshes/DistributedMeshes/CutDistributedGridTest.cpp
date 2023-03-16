
#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#ifdef HAVE_MPI

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Devices;


template< typename MeshType >
void SetUpDistributedGrid( DistributedMesh<MeshType> &distributedGrid,
                           MeshType &globalGrid,
                           int size,
                           typename MeshType::CoordinatesType distribution )
{
    typename MeshType::PointType globalOrigin;
    typename MeshType::PointType globalProportions;
    using DistributedMeshType = DistributedMesh< MeshType >;

    globalOrigin.setValue( -0.5 );
    globalProportions.setValue( size );

    globalGrid.setDimensions( size );
    globalGrid.setDomain( globalOrigin, globalProportions );

    distributedGrid.setDomainDecomposition( distribution );
    distributedGrid.setGlobalGrid(globalGrid);
    typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
    SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
    distributedGrid.setOverlaps( lowerOverlap, upperOverlap );
}

//===============================================2D================================================================

TEST(CutDistributedGridTest_2D, IsInCut)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 10, CoordinatesType(3,4));

    CutDistributedGridType cutDistributedGrid;
    bool result=cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            );

    if(TNL::MPI::GetRank()%3==1)
    {
        ASSERT_TRUE(result);
    }
    else
    {
        ASSERT_FALSE(result);
    }
}

TEST(CutDistributedGridTest_2D, GloblaGridDimesion)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid, globalGrid, 10, CoordinatesType(3,4));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getMeshDimension(),1) << "Dimenze globálního gridu neodpovídajá řezu";
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().x(),10) << "Rozměry globálního gridu neodpovídají";
    }
}

TEST(CutDistributedGridTest_2D, IsDistributed)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 10, CoordinatesType(3,4));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_TRUE(cutDistributedGrid.isDistributed()) << "Řez by měl být distribuovaný";
    }
}

TEST(CutDistributedGridTest_2D, IsNotDistributed)
{
    typedef Grid<2,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 10, CoordinatesType(12,1));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_FALSE(cutDistributedGrid.isDistributed()) << "Řez by neměl být distribuovaný";
    }
}

//===============================================3D - 1D cut================================================================

TEST(CutDistributedGridTest_3D, IsInCut_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    bool result=cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(2,2)
            );

    if(TNL::MPI::GetRank()%4==0)
    {
        ASSERT_TRUE(result);
    }
    else
    {
        ASSERT_FALSE(result);
    }
}

TEST(CutDistributedGridTest_3D, GloblaGridDimesion_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid, globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(2,2)
            ))
    {
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getMeshDimension(),1) << "Dimenze globálního gridu neodpovídajá řezu";
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().x(),10) << "Rozměry globálního gridu neodpovídají";
    }
}

TEST(CutDistributedGridTest_3D, IsDistributed_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(2,2)
            ))
    {
        EXPECT_TRUE(cutDistributedGrid.isDistributed()) << "Řez by měl být distribuovaný";
    }
}

TEST(CutDistributedGridTest_3D, IsNotDistributed_1D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<1,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 30, CoordinatesType(12,1,1));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<1,int>(2),
            StaticVector<2,int>(0,1),
            StaticVector<2,int>(1,1)
            ))
    {
        EXPECT_FALSE(cutDistributedGrid.isDistributed()) << "Řez by neměl být distribuovaný";
    }
}

//===================================3D-2D cut=========================================================================

TEST(CutDistributedGridTest_3D, IsInCut_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    bool result=cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            );

    int rank=TNL::MPI::GetRank();
    if(rank>3 && rank<8)
    {
        ASSERT_TRUE(result);
    }
    else
    {
        ASSERT_FALSE(result);
    }
}

TEST(CutDistributedGridTest_3D, GloblaGridDimesion_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid, globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getMeshDimension(),2) << "Dimenze globálního gridu neodpovídajá řezu";
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().x(),10) << "Rozměry globálního gridu neodpovídají";
        EXPECT_EQ(cutDistributedGrid.getGlobalGrid().getDimensions().y(),10) << "Rozměry globálního gridu neodpovídají";
    }
}

TEST(CutDistributedGridTest_3D, IsDistributed_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 10, CoordinatesType(2,2,3));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_TRUE(cutDistributedGrid.isDistributed()) << "Řez by měl být distribuovaný";
    }
}

TEST(CutDistributedGridTest_3D, IsNotDistributed_2D)
{
    typedef Grid<3,double,Host,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef DistributedMesh<MeshType> DistributedMeshType;
    typedef Grid<2,double,Host,int> CutGridType;
    typedef DistributedMesh<CutGridType> CutDistributedGridType;

    MeshType globalGrid;
    DistributedMeshType distributedGrid;
    SetUpDistributedGrid<MeshType>(distributedGrid,globalGrid, 30, CoordinatesType(1,1,12));

    CutDistributedGridType cutDistributedGrid;
    if(cutDistributedGrid.SetupByCut(
            distributedGrid,
            StaticVector<2,int>(0,1),
            StaticVector<1,int>(2),
            StaticVector<1,int>(5)
            ))
    {
        EXPECT_FALSE(cutDistributedGrid.isDistributed()) << "Řez by neměl být distribuovaný";
    }
}
#endif

#endif

#include "../../main_mpi.h"
