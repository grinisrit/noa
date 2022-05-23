#include <iostream>

#include <TNL/Functions/MeshFunctionView.h>

#include "../../UnitTests/Functions/Functions.h"

using namespace std;

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;

int main(int argc, char ** argv)
{

    int size=1024;
    int cycles=1;
    if(argc==3)
    {
      size=strtol(argv[1],NULL,10);
      cycles=strtol(argv[2],NULL,10);
      cout << "size: "<< size <<" cycles: "<< cycles <<endl;
     }

    Timer time;
    time.start();

#ifdef HAVE_CUDA
    using Device=Devices::Cuda;
#else
    using Device=Devices::Host;
#endif

  using MeshType= Grid<2, double,Device,int>;
  using MeshFunctionType = MeshFunctionView<MeshType>;
  using DofType = Vector<double,Device,int>;
  using Cell = typename MeshType::Cell ;
  using IndexType = typename MeshType::IndexType ; 
  using PointType = typename MeshType::PointType ;
  using CoordinatesType = typename MeshType::CoordinatesType;

  using LinearFunctionType=LinearFunction<double,2> ;

  PointType origin(-0.5);
  PointType proportions(size);
 
  Pointers::SharedPointer<MeshType> gridPtr;
  gridPtr->setDimensions(proportions);
  gridPtr->setDomain(origin,proportions);

  Pointers::SharedPointer<MeshFunctionType> meshFunctionptr;
  MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;

  DofType dof(gridPtr->template getEntitiesCount< Cell >());
  dof.setValue(0);  
  meshFunctionptr->bind(gridPtr,dof);  
  
  Pointers::SharedPointer< LinearFunctionType, Device > linearFunctionPtr;

  for(int i=0;i<cycles;i++)
  {
    linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
  }


  time.stop();

  cout << time.getRealTime()<<"s"<<endl;

}
