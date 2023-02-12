#include <TNL/Meshes/DistributedMeshes/CopyEntitiesHelper.h>
#include <TNL/Functions/MeshFunctionView.h>


#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include "../../Functions/Functions.h"

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Meshes::DistributedMeshes;

//================================TEST RESULT============================================
template <typename MeshFunctionType,
		  int dim=MeshFunctionType::getMeshDimension()>
class TestMovedMeshfunction
{
	public:
		static void Test(MeshFunctionType &meshFunction)
		{}
};

template <typename MeshFunctionType>
class TestMovedMeshfunction<MeshFunctionType,1>
{
	public:
		static void Test(MeshFunctionType &meshFunction)
		{

			for(int i=0;i<meshFunction.getData().getSize();i++)
				EXPECT_EQ( meshFunction.getData()[i], i+1) << "Copy entities Failed for: "<< i;
		}
};

template <typename MeshFunctionType>
class TestMovedMeshfunction<MeshFunctionType,2>
{
	public:
		static void Test(MeshFunctionType &meshFunction)
		{
			typename MeshFunctionType::MeshType::Cell entity(meshFunction.getMesh());
			auto size=meshFunction.getMesh().getDimensions();
			for(int j=0;j<size.y();j++)
				for(int i=0;i<size.x();i++)
				{
				    entity.getCoordinates().x()=i;
					entity.getCoordinates().y()=j;
				    entity.refresh();
					EXPECT_EQ( meshFunction.getData()[entity.getIndex()], 10*(j+1)+(i+1)) << "Copy entities Failed for: "<< i;
				}
		}
};

template <typename MeshFunctionType>
class TestMovedMeshfunction<MeshFunctionType,3>
{
	public:
		static void Test(MeshFunctionType &meshFunction)
		{
			typename MeshFunctionType::MeshType::Cell entity(meshFunction.getMesh());
			auto size=meshFunction.getMesh().getDimensions();
			for(int k=0;k<size.z();k++)
				for(int j=0;j<size.y();j++)
					for(int i=0;i<size.x();i++)
					{
				    	entity.getCoordinates().x()=i;
						entity.getCoordinates().y()=j;
						entity.getCoordinates().z()=k;
				    	entity.refresh();
						EXPECT_EQ( meshFunction.getData()[entity.getIndex()], 100*(k+1)+10*(j+1)+(i+1)) << "Copy entities Failed for: "<< i;
					}
		}
};

//================================SET INPUT============================================

template <typename MeshFunctionType,
		  int dim=MeshFunctionType::getMeshDimension()>
class EvalMeshFunction
{
	public:
		static void Eval(MeshFunctionType &meshFunction)
		{}
};

template <typename MeshFunctionType>
class EvalMeshFunction<MeshFunctionType,1>
{
	public:
		static void Eval(MeshFunctionType &meshFunction)
		{
			for(int i=0;i<meshFunction.getData().getSize();i++)
				meshFunction.getData()[i]=i;
		}
};

template <typename MeshFunctionType>
class EvalMeshFunction<MeshFunctionType,2>
{
	public:
		static void Eval(MeshFunctionType &meshFunction)
		{
			typename MeshFunctionType::MeshType::Cell entity(meshFunction.getMesh());
			auto size=meshFunction.getMesh().getDimensions();
			for(int j=0;j<size.y();j++)
				for(int i=0;i<size.x();i++)
				{
				    entity.getCoordinates().x()=i;
					entity.getCoordinates().y()=j;
				    entity.refresh();
					meshFunction.getData()[entity.getIndex()]= 10*j+i;
				}
		}
};

template <typename MeshFunctionType>
class EvalMeshFunction<MeshFunctionType,3>
{
	public:
		static void Eval(MeshFunctionType &meshFunction)
		{
			typename MeshFunctionType::MeshType::Cell entity(meshFunction.getMesh());
			auto size=meshFunction.getMesh().getDimensions();
			for(int k=0;k<size.z();k++)
				for(int j=0;j<size.y();j++)
					for(int i=0;i<size.x();i++)
					{
				    	entity.getCoordinates().x()=i;
						entity.getCoordinates().y()=j;
						entity.getCoordinates().z()=k;
				    	entity.refresh();
						meshFunction.getData()[entity.getIndex()]=100*k+10*j+i;
					}
		}
};


//=====================================TEST CLASS==============================================

template <int dim>
class TestCopyEntities
{
	public:
		static void Test()
		{
			typedef Grid<dim,double,Host,int> MeshType;
			typedef MeshFunctionView<MeshType> MeshFunctionType;
			typedef Vector<double,Host,int> DofType;

			typedef typename MeshType::PointType PointType;
			typedef typename MeshType::CoordinatesType CoordinatesType;
			typedef typename MeshType::Cell Cell;

			PointType origin;
			PointType proportions;
			Pointers::SharedPointer<MeshType> gridptr;

			origin.setValue(-0.5);
			proportions.setValue(10);

			gridptr->setDimensions(proportions);
			gridptr->setDomain(origin,proportions);

			DofType inputDof(gridptr-> template getEntitiesCount< Cell >());

			MeshFunctionType inputMeshFunction;
			inputMeshFunction.bind(gridptr,inputDof);

			EvalMeshFunction<MeshFunctionType> :: Eval(inputMeshFunction);


			PointType originOut;
			PointType proportionsOut;
			Pointers::SharedPointer<MeshType> gridOutPtr;

			originOut.setValue(0.5);
			proportionsOut.setValue(8);
			gridOutPtr->setDimensions(proportionsOut);
			gridOutPtr->setDomain(originOut,proportionsOut);
			DofType outDof(gridOutPtr-> template getEntitiesCount< Cell >());

			MeshFunctionType outputMeshFunction;
			outputMeshFunction.bind(gridOutPtr,outDof);

			CoordinatesType zero;
			zero.setValue(0);
			CoordinatesType begin;
			begin.setValue(1);
			CoordinatesType size;
			size.setValue(8);

			CopyEntitiesHelper< MeshFunctionType >::Copy(inputMeshFunction,outputMeshFunction, begin,zero, size);

			TestMovedMeshfunction<MeshFunctionType>::Test(outputMeshFunction);
		}
};

TEST( CopyEntitiesTest, 1D )
{
	TestCopyEntities<1>::Test();
}

/*TEST( CopyEntitiesTest, 2D )
{
	TestCopyEntities<2>::Test();
}

TEST( CopyEntitiesTest, 3D )
{
	TestCopyEntities<3>::Test();
}*/


#endif

#include "../../main.h"
