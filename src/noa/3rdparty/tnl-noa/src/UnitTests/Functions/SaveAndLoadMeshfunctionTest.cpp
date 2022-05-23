#include <TNL/Functions/MeshFunctionView.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include "Functions.h"

#include <iostream>

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;

static const char* TEST_FILE_NAME = "test_SaveAndLoadMeshFunctionTest.vti";

//=====================================TEST CLASS==============================================

template <int dim>
class TestSaveAndLoadMeshfunction
{
    public:
        static void Test()
        {
            typedef Grid<dim,double,Host,int> MeshType;
            typedef MeshFunctionView<MeshType> MeshFunctionType;
            typedef Vector<double,Host,int> DofType;
            typedef typename MeshType::Cell Cell;
            typedef typename MeshType::PointType PointType;

            typedef LinearFunction<double,dim> LinearFunctionType;

            Pointers::SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
            MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;


            PointType localOrigin = -0.5;
            PointType localProportions = 10;

            Pointers::SharedPointer<MeshType>  localGridptr;
            localGridptr->setDimensions(localProportions);
            localGridptr->setDomain(localOrigin,localProportions);

            DofType localDof(localGridptr->template getEntitiesCount< Cell >());

            Pointers::SharedPointer<MeshFunctionType> localMeshFunctionptr;
            localMeshFunctionptr->bind(localGridptr,localDof);
            linearFunctionEvaluator.evaluateAllEntities(localMeshFunctionptr, linearFunctionPtr);

            ASSERT_TRUE( localMeshFunctionptr->write( "funcName", TEST_FILE_NAME ) );

            //load other meshfunction on same localgrid from created file
            Pointers::SharedPointer<MeshType>  loadGridptr;
            loadGridptr->setDimensions(localProportions);
            loadGridptr->setDomain(localOrigin,localProportions);

            DofType loadDof(loadGridptr->template getEntitiesCount< Cell >());
            loadDof.setValue(-1);
            Pointers::SharedPointer<MeshFunctionType> loadMeshFunctionptr;
            loadMeshFunctionptr->bind(loadGridptr,loadDof);

            ASSERT_TRUE( Functions::readMeshFunction( *loadMeshFunctionptr, "funcName", TEST_FILE_NAME ) );

            for(int i=0;i<localDof.getSize();i++)
            {
                EXPECT_EQ( localDof[i], loadDof[i]) << "Compare Loaded and evaluated Dof Failed for: "<< i;
            }

            EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );

        };
};

TEST( CopyEntitiesTest, 1D )
{
    TestSaveAndLoadMeshfunction<1>::Test();
}

TEST( CopyEntitiesTest, 2D )
{
    TestSaveAndLoadMeshfunction<2>::Test();
}

TEST( CopyEntitiesTest, 3D )
{
    TestSaveAndLoadMeshfunction<3>::Test();
}
#endif

#include "../main.h"
