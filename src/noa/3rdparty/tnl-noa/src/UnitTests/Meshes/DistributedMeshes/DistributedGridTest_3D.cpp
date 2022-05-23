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
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Meshes::DistributedMeshes;

template<typename DofType>
void setDof_3D(DofType &dof, typename DofType::RealType value)
{
    for(int i=0;i<dof.getSize();i++)
        dof[i]=value;
}

template<typename GridType>
int getAdd(GridType &grid,bool bottom, bool north, bool west )
{
    int maxx=grid.getDimensions().x();
    int maxy=grid.getDimensions().y();
    int maxz=grid.getDimensions().z();

    int add=0;
    if(!west)
        add+=maxx-1;
    if(!north)
        add+=(maxy-1)*maxx;
    if(!bottom)
        add+=(maxz-1)*maxx*maxy;

    return add;
}

template<typename DofType,typename GridType>
void checkConner(const GridType &grid, const DofType &dof,bool bottom, bool north, bool west, typename DofType::RealType expectedValue )
{
    int i=getAdd(grid,bottom,north,west);
    EXPECT_EQ( dof[i], expectedValue) << "Conner test failed";

}

template<typename DofType,typename GridType>
void checkXDirectionEdge(const GridType &grid, const DofType &dof, bool bottom, bool north, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,bottom,north,true);
    for(int i=1;i<grid.getDimensions().x()-1;i++)
            EXPECT_EQ( dof[i+add], expectedValue) << "X direction Edge test failed " << i;
}


template<typename DofType,typename GridType>
void checkYDirectionEdge(const GridType &grid, const DofType &dof, bool bottom, bool west, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,bottom,true,west);
    for(int i=1;i<grid.getDimensions().y()-1;i++)
            EXPECT_EQ( dof[grid.getDimensions().x()*i+add], expectedValue) << "Y direction Edge test failed " << i;
}

template<typename DofType,typename GridType>
void checkZDirectionEdge(const GridType &grid, const DofType &dof, bool north, bool west, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,true,north,west);
    for(int i=1;i<grid.getDimensions().z()-1;i++)
            EXPECT_EQ( dof[grid.getDimensions().y()*grid.getDimensions().x()*i+add], expectedValue) << "Z direction Edge test failed " << i;
}

template<typename DofType,typename GridType>
void checkZFace(const GridType &grid, const DofType &dof, bool bottom, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,bottom,true,true);
    for(int i=1;i<grid.getDimensions().y()-1;i++)
        for(int j=1; j<grid.getDimensions().x()-1;j++)
        {
            EXPECT_EQ( dof[grid.getDimensions().x()*i+j+add], expectedValue) << "Z Face test failed "<<i<< " " << j;
        }
}

template<typename DofType,typename GridType>
void checkYFace(const GridType &grid, const DofType &dof, bool north, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,true,north,true);
    for(int i=1;i<grid.getDimensions().z()-1;i++)
        for(int j=1; j<grid.getDimensions().x()-1;j++)
        {
            EXPECT_EQ( dof[grid.getDimensions().y()*grid.getDimensions().x()*i+j+add], expectedValue) << "Y Face test failed "<<i<< " " << j;
        }
}

template<typename DofType,typename GridType>
void checkXFace(const GridType &grid, const DofType &dof, bool west, typename DofType::RealType expectedValue)
{
    int add=getAdd(grid,true,true,west);
    for(int i=1;i<grid.getDimensions().z()-1;i++)
        for(int j=1; j<grid.getDimensions().y()-1;j++)
        {
            EXPECT_EQ( dof[grid.getDimensions().y()*grid.getDimensions().x()*i+grid.getDimensions().x()*j+add], expectedValue) << "X Face test failed "<<i<< " " << j;
        }
}

/*
Expected 27 processes
*/
template<typename DofType,typename GridType>
void check_Boundary_3D(int rank, const GridType &grid, const DofType &dof, typename DofType::RealType expectedValue)
{
    if(rank==0)//Bottom North West
    {
        checkConner(grid,dof,true, true, true, expectedValue );
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkZDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==1)//Bottom North Center
    {
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==2)//Bottom North East
    {
        checkConner(grid,dof,true, true, false, expectedValue );
        checkXDirectionEdge(grid,dof,true,true,expectedValue);
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==3)//Bottom Center West
    {
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==4)//Bottom Center Center
    {
        checkZFace(grid, dof, true, expectedValue);
    }


    if(rank==5)//Bottom Center East
    {
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==6)//Bottom South West
    {
        checkConner(grid,dof,true, false, true, expectedValue );
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkYDirectionEdge(grid,dof,true,true,expectedValue);
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==7)//Bottom South Center
    {
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==8)//Bottom South East
    {
        checkConner(grid,dof,true, false, false, expectedValue );
        checkXDirectionEdge(grid,dof,true,false,expectedValue);
        checkYDirectionEdge(grid,dof,true,false,expectedValue);
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, true, expectedValue);
    }

    if(rank==9)//Center North West
    {
        checkZDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
    }

    if(rank==10)//Center North Center
    {
        checkYFace(grid, dof, true, expectedValue);
    }

    if(rank==11)//Center North East
    {
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
    }

    if(rank==12)//Center Center West
    {
        checkXFace(grid, dof, true, expectedValue);
    }

    if(rank==13)//Center Center Center
    {
        //no Boundary
    }


    if(rank==14)//Center Center East
    {
        checkXFace(grid, dof, false, expectedValue);
    }

    if(rank==15)//Center South West
    {
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
    }

    if(rank==16)//Center South Center
    {
        checkYFace(grid, dof, false, expectedValue);
    }

    if(rank==17)//Center South East
    {
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
    }

    if(rank==18)//Top North West
    {
        checkConner(grid,dof,false, true, true, expectedValue );
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkZDirectionEdge(grid,dof,true,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==19)//Top North Center
    {
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==20)//Top North East
    {
        checkConner(grid,dof,false, true, false, expectedValue );
        checkXDirectionEdge(grid,dof,false,true,expectedValue);
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkZDirectionEdge(grid,dof,true,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==21)//Top Center West
    {
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==22)//Top Center Center
    {
        checkZFace(grid, dof, false, expectedValue);
    }


    if(rank==23)//Top Center East
    {
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==24)//Top South West
    {
        checkConner(grid,dof,false, false, true, expectedValue );
        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkYDirectionEdge(grid,dof,false,true,expectedValue);
        checkZDirectionEdge(grid,dof,false,true,expectedValue);
        checkXFace(grid, dof, true, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==25)//Top South Center
    {
        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

    if(rank==26)//Top South East
    {
        checkConner(grid,dof,false, false, false, expectedValue );
        checkXDirectionEdge(grid,dof,false,false,expectedValue);
        checkYDirectionEdge(grid,dof,false,false,expectedValue);
        checkZDirectionEdge(grid,dof,false,false,expectedValue);
        checkXFace(grid, dof, false, expectedValue);
        checkYFace(grid, dof, false, expectedValue);
        checkZFace(grid, dof, false, expectedValue);
    }

}

template<typename DofType,typename GridType>
void CheckConnerNode_Overlap(const GridType &grid, const DofType &dof,bool bottom, bool north, bool west, typename DofType::RealType expectedValue)
{
   checkConner(grid,dof,!bottom, !north, !west, expectedValue );
   checkXDirectionEdge(grid,dof,!bottom,!north,expectedValue);
   checkYDirectionEdge(grid,dof,!bottom,!west,expectedValue);
   checkZDirectionEdge(grid,dof,!north,!west,expectedValue);
   checkXFace(grid, dof, !west, expectedValue);
   checkYFace(grid, dof, !north, expectedValue);
   checkZFace(grid, dof, !bottom, expectedValue);
}

template<typename DofType,typename GridType>
void CheckXEdgeNode_Overlap(const GridType &grid, const DofType &dof,bool bottom, bool north, typename DofType::RealType expectedValue)
{
   checkConner(grid,dof,!bottom, !north, false, expectedValue );
   checkConner(grid,dof,!bottom, !north, true, expectedValue );
   checkXDirectionEdge(grid,dof,!bottom, !north,expectedValue);
   checkYDirectionEdge(grid,dof,!bottom,false,expectedValue);
   checkYDirectionEdge(grid,dof,!bottom,true,expectedValue);
   checkZDirectionEdge(grid,dof,!north,false,expectedValue);
   checkZDirectionEdge(grid,dof,!north,true,expectedValue);
   checkXFace(grid, dof, false, expectedValue);
   checkXFace(grid, dof, true, expectedValue);
   checkYFace(grid, dof, !north, expectedValue);
   checkZFace(grid, dof, !bottom, expectedValue);
}

template<typename DofType,typename GridType>
void CheckYEdgeNode_Overlap(const GridType &grid, const DofType &dof,bool bottom, bool west, typename DofType::RealType expectedValue)
{
   checkConner(grid,dof,!bottom, false, !west, expectedValue );
   checkConner(grid,dof,!bottom, true, !west, expectedValue );
   checkXDirectionEdge(grid,dof,!bottom,false,expectedValue);
   checkXDirectionEdge(grid,dof,!bottom,true,expectedValue);
   checkYDirectionEdge(grid,dof,!bottom,!west,expectedValue);
   checkZDirectionEdge(grid,dof,false,!west,expectedValue);
   checkZDirectionEdge(grid,dof,true,!west,expectedValue);
   checkXFace(grid, dof, !west, expectedValue);
   checkYFace(grid, dof, false, expectedValue);
   checkYFace(grid, dof, true, expectedValue);
   checkZFace(grid, dof, !bottom, expectedValue);
}

template<typename DofType,typename GridType>
void CheckZEdgeNode_Overlap(const GridType &grid, const DofType &dof,bool north, bool west, typename DofType::RealType expectedValue)
{
   checkConner(grid,dof,false, !north, !west, expectedValue );
   checkConner(grid,dof,true, !north, !west, expectedValue );
   checkXDirectionEdge(grid,dof,false,!north,expectedValue);
   checkXDirectionEdge(grid,dof,true,!north,expectedValue);
   checkYDirectionEdge(grid,dof,false,!west,expectedValue);
   checkYDirectionEdge(grid,dof,true,!west,expectedValue);
   checkZDirectionEdge(grid,dof,!north,!west,expectedValue);
   checkXFace(grid, dof, !west, expectedValue);
   checkYFace(grid, dof, !north, expectedValue);
   checkZFace(grid, dof, false, expectedValue);
   checkZFace(grid, dof, true, expectedValue);
}

template<typename DofType,typename GridType>
void CheckXFaceNode_Overlap(const GridType &grid, const DofType &dof,bool west, typename DofType::RealType expectedValue)
{
   checkConner(grid,dof,false, false, !west, expectedValue );
   checkConner(grid,dof,false, true, !west, expectedValue );
   checkConner(grid,dof,true, false, !west, expectedValue );
   checkConner(grid,dof,true, true, !west, expectedValue );
   checkXDirectionEdge(grid,dof,false,false,expectedValue);
   checkXDirectionEdge(grid,dof,false,true,expectedValue);
   checkXDirectionEdge(grid,dof,true,false,expectedValue);
   checkXDirectionEdge(grid,dof,true,true,expectedValue);
   checkYDirectionEdge(grid,dof,false,!west,expectedValue);
   checkYDirectionEdge(grid,dof,true,!west,expectedValue);
   checkZDirectionEdge(grid,dof,false,!west,expectedValue);
   checkZDirectionEdge(grid,dof,true,!west,expectedValue);
   checkXFace(grid, dof, !west, expectedValue);
   checkYFace(grid, dof, false, expectedValue);
   checkYFace(grid, dof, true, expectedValue);
   checkZFace(grid, dof, false, expectedValue);
   checkZFace(grid, dof, true, expectedValue);
}

template<typename DofType,typename GridType>
void CheckYFaceNode_Overlap(const GridType &grid, const DofType &dof,bool north, typename DofType::RealType expectedValue)
{
   checkConner(grid,dof, false,!north, false, expectedValue );
   checkConner(grid,dof, false,!north, true, expectedValue );
   checkConner(grid,dof, true, !north, false, expectedValue );
   checkConner(grid,dof, true, !north, true, expectedValue );
   checkXDirectionEdge(grid,dof,false,!north,expectedValue);
   checkXDirectionEdge(grid,dof,true,!north,expectedValue);
   checkYDirectionEdge(grid,dof,false,false,expectedValue);
   checkYDirectionEdge(grid,dof,false,true,expectedValue);
   checkYDirectionEdge(grid,dof,true,false,expectedValue);
   checkYDirectionEdge(grid,dof,true,true,expectedValue);
   checkZDirectionEdge(grid,dof,!north,false,expectedValue);
   checkZDirectionEdge(grid,dof,!north,true,expectedValue);
   checkXFace(grid, dof, false, expectedValue);
   checkXFace(grid, dof, true, expectedValue);
   checkYFace(grid, dof, !north, expectedValue);
   checkZFace(grid, dof, false, expectedValue);
   checkZFace(grid, dof, true, expectedValue);
}

template<typename DofType,typename GridType>
void CheckZFaceNode_Overlap(const GridType &grid, const DofType &dof,bool bottom, typename DofType::RealType expectedValue)
{
   checkConner(grid,dof,!bottom, false, false, expectedValue );
   checkConner(grid,dof,!bottom, false, true, expectedValue );
   checkConner(grid,dof,!bottom, true, false, expectedValue );
   checkConner(grid,dof,!bottom, true, true, expectedValue );
   checkXDirectionEdge(grid,dof,!bottom,false,expectedValue);
   checkXDirectionEdge(grid,dof,!bottom,true,expectedValue);
   checkYDirectionEdge(grid,dof,!bottom,false,expectedValue);
   checkYDirectionEdge(grid,dof,!bottom,true,expectedValue);
   checkZDirectionEdge(grid,dof,false,false,expectedValue);
   checkZDirectionEdge(grid,dof,false,true,expectedValue);
   checkZDirectionEdge(grid,dof,true,false,expectedValue);
   checkZDirectionEdge(grid,dof,true,true,expectedValue);
   checkXFace(grid, dof, false, expectedValue);
   checkXFace(grid, dof, true, expectedValue);
   checkYFace(grid, dof, false, expectedValue);
   checkYFace(grid, dof, true, expectedValue);
   checkZFace(grid, dof, !bottom, expectedValue);
}

template<typename DofType,typename GridType>
void CheckCentralNode_Overlap(const GridType &grid, const DofType &dof,typename DofType::RealType expectedValue)
{
   checkConner(grid,dof,false, false, false, expectedValue );
   checkConner(grid,dof,false, false, true, expectedValue );
   checkConner(grid,dof,false, true, false, expectedValue );
   checkConner(grid,dof,false, true, true, expectedValue );
   checkConner(grid,dof,true, false, false, expectedValue );
   checkConner(grid,dof,true, false, true, expectedValue );
   checkConner(grid,dof,true, true, false, expectedValue );
   checkConner(grid,dof,true, true, true, expectedValue );

   checkXDirectionEdge(grid,dof,false,false,expectedValue);
   checkXDirectionEdge(grid,dof,false,true,expectedValue);
   checkXDirectionEdge(grid,dof,true,false,expectedValue);
   checkXDirectionEdge(grid,dof,true,true,expectedValue);
   checkYDirectionEdge(grid,dof,false,false,expectedValue);
   checkYDirectionEdge(grid,dof,false,true,expectedValue);
   checkYDirectionEdge(grid,dof,true,false,expectedValue);
   checkYDirectionEdge(grid,dof,true,true,expectedValue);
   checkZDirectionEdge(grid,dof,false,false,expectedValue);
   checkZDirectionEdge(grid,dof,false,true,expectedValue);
   checkZDirectionEdge(grid,dof,true,false,expectedValue);
   checkZDirectionEdge(grid,dof,true,true,expectedValue);

   checkXFace(grid, dof, false, expectedValue);
   checkXFace(grid, dof, true, expectedValue);
   checkYFace(grid, dof, false, expectedValue);
   checkYFace(grid, dof, true, expectedValue);
   checkZFace(grid, dof, false, expectedValue);
   checkZFace(grid, dof, true, expectedValue);
}

/*
* Expected 27 processes.
*/
template<typename DofType,typename GridType>
void check_Overlap_3D(int rank, const GridType &grid, const DofType &dof, typename DofType::RealType expectedValue)
{
   if(rank==0)
       CheckConnerNode_Overlap(grid,dof,true,true,true,expectedValue);

   if(rank==1)
       CheckXEdgeNode_Overlap(grid,dof,true,true,expectedValue);

   if(rank==2)
       CheckConnerNode_Overlap(grid,dof,true,true,false,expectedValue);

   if(rank==3)
       CheckYEdgeNode_Overlap(grid,dof,true,true,expectedValue);

   if(rank==4)
       CheckZFaceNode_Overlap(grid,dof,true,expectedValue);

   if(rank==5)
       CheckYEdgeNode_Overlap(grid,dof,true,false,expectedValue);

   if(rank==6)
       CheckConnerNode_Overlap(grid,dof,true,false,true,expectedValue);

   if(rank==7)
       CheckXEdgeNode_Overlap(grid,dof,true,false,expectedValue);

   if(rank==8)
       CheckConnerNode_Overlap(grid,dof,true,false,false,expectedValue);

   if(rank==9)
       CheckZEdgeNode_Overlap(grid,dof,true,true,expectedValue);

   if(rank==10)
       CheckYFaceNode_Overlap(grid,dof,true,expectedValue);

   if(rank==11)
       CheckZEdgeNode_Overlap(grid,dof,true,false,expectedValue);

   if(rank==12)
       CheckXFaceNode_Overlap(grid,dof,true,expectedValue);

   if(rank==13)
       CheckCentralNode_Overlap(grid,dof,expectedValue);

   if(rank==14)
       CheckXFaceNode_Overlap(grid,dof,false,expectedValue);

   if(rank==15)
       CheckZEdgeNode_Overlap(grid,dof,false,true,expectedValue);

   if(rank==16)
       CheckYFaceNode_Overlap(grid,dof,false,expectedValue);

   if(rank==17)
       CheckZEdgeNode_Overlap(grid,dof,false,false,expectedValue);

   if(rank==18)
       CheckConnerNode_Overlap(grid,dof,false,true,true,expectedValue);

   if(rank==19)
       CheckXEdgeNode_Overlap(grid,dof,false,true,expectedValue);

   if(rank==20)
       CheckConnerNode_Overlap(grid,dof,false,true,false,expectedValue);

   if(rank==21)
       CheckYEdgeNode_Overlap(grid,dof,false,true,expectedValue);

   if(rank==22)
       CheckZFaceNode_Overlap(grid,dof,false,expectedValue);

   if(rank==23)
       CheckYEdgeNode_Overlap(grid,dof,false,false,expectedValue);

   if(rank==24)
       CheckConnerNode_Overlap(grid,dof,false,false,true,expectedValue);

   if(rank==25)
       CheckXEdgeNode_Overlap(grid,dof,false,false,expectedValue);

   if(rank==26)
       CheckConnerNode_Overlap(grid,dof,false,false,false,expectedValue);

}

template<typename DofType,typename GridType>
void check_Inner_3D(int rank, const GridType& grid, const DofType& dof, typename DofType::RealType expectedValue)
{
   int maxx=grid.getDimensions().x();
   int maxy=grid.getDimensions().y();
   int maxz=grid.getDimensions().z();
   for(int k=1;k<maxz-1;k++)
      for(int j=1;j<maxy-1;j++)//prvni a posledni jsou buď hranice, nebo overlap
         for(int i=1;i<maxx-1;i++) //buď je vlevo hranice, nebo overlap
            EXPECT_EQ( dof[k*maxx*maxy+j*maxx+i], expectedValue) <<" "<<k <<" "<< j<<" "<<i << " " << maxx << " " << maxy<< " " << maxz;
}


/*
 * Light check of 3D distributed grid and its synchronization.
 * expected 27 processes
 */
typedef Grid<3,double,Host,int> GridType;
typedef MeshFunctionView<GridType> MeshFunctionType;
typedef Vector<double,Host,int> DofType;
typedef typename GridType::Cell Cell;
typedef typename GridType::IndexType IndexType;
typedef typename GridType::PointType PointType;
typedef DistributedMesh<GridType> DistributedGridType;
using Synchronizer = DistributedMeshSynchronizer< DistributedGridType >;

class DistributedGridTest_3D : public ::testing::Test
{
   protected:

      DistributedGridType *distributedGrid;
      DofType dof;

      Pointers::SharedPointer<GridType> localGrid;
      Pointers::SharedPointer<MeshFunctionType> meshFunctionPtr;

      MeshFunctionEvaluator< MeshFunctionType, ConstFunction<double,3> > constFunctionEvaluator;
      Pointers::SharedPointer< ConstFunction<double,3>, Host > constFunctionPtr;

      MeshFunctionEvaluator< MeshFunctionType, LinearFunction<double,3> > linearFunctionEvaluator;
      Pointers::SharedPointer< LinearFunction<double,3>, Host > linearFunctionPtr;

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
         globalOrigin.z()=-0.5;
         globalProportions.x()=size;
         globalProportions.y()=size;
         globalProportions.z()=size;

         globalGrid.setDimensions(size,size,size);
         globalGrid.setDomain(globalOrigin,globalProportions);

         distributedGrid=new DistributedGridType();
         distributedGrid->setDomainDecomposition( typename DistributedGridType::CoordinatesType( 3, 3, 3 ) );
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

TEST_F(DistributedGridTest_3D, evaluateAllEntities)
{

    //Check Traversars
    //All entities, witout overlap
    setDof_3D(dof,-1);
    constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
    check_Boundary_3D(rank, *localGrid, dof, rank);
    check_Overlap_3D(rank, *localGrid, dof, -1);
    check_Inner_3D(rank, *localGrid, dof, rank);
}

TEST_F(DistributedGridTest_3D, evaluateBoundaryEntities)
{
    //Boundary entities, witout overlap
    setDof_3D(dof,-1);
    constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionPtr , constFunctionPtr );
    check_Boundary_3D(rank, *localGrid, dof, rank);
    check_Overlap_3D(rank, *localGrid, dof, -1);
    check_Inner_3D(rank, *localGrid, dof, -1);
}

TEST_F(DistributedGridTest_3D, evaluateInteriorEntities)
{
    //Inner entities, witout overlap
    setDof_3D(dof,-1);
    constFunctionEvaluator.evaluateInteriorEntities( meshFunctionPtr , constFunctionPtr );
    check_Boundary_3D(rank, *localGrid, dof, -1);
    check_Overlap_3D(rank, *localGrid, dof, -1);
    check_Inner_3D(rank, *localGrid, dof, rank);
}

TEST_F(DistributedGridTest_3D, LinearFunctionTest)
{
    //fill meshfunction with linear function (physical center of cell corresponds with its coordinates in grid)
    setDof_3D(dof,-1);
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

/* not implemented
TEST_F(DistributedGridTest_3D, SynchronizerNeighborTest)
{

}
*/

TEST_F(DistributedGridTest_3D, PVTIWriterReader)
{
   // create a .pvti file (only rank 0 actually writes to the file)
   const std::string baseName = "DistributedGridTest_3D_" + std::to_string(nproc) + "proc";
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

TEST_F(DistributedGridTest_3D, readDistributedMeshFunction)
{
   const std::string baseName = "DistributedGridTest_MeshFunction_3D_" + std::to_string(nproc) + "proc.pvti";
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
