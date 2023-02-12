#ifndef PARALLELGODUNOVEIKONAL2D_IMPL_H_
#define PARALLELGODUNOVEIKONAL2D_IMPL_H_


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef __CUDACC__
   __device__ __host__
#endif
Real parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >:: positivePart(const Real arg) const
{
	if(arg > 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef __CUDACC__
   __device__ __host__
#endif
Real  parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: negativePart(const Real arg) const
{
	if(arg < 0.0)
		return -arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef __CUDACC__
   __device__ __host__
#endif
Real parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: sign(const Real x, const Real eps) const
{
	if(x > eps)
		return 1.0;
	if (x < -eps)
		return (-1.0);

	if ( x == 0.0)
		return 0.0;

	return sin(/*(M_PI*x)/(2.0*eps)	*/(M_PI/2.0)*(x/eps));
}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef __CUDACC__
   __device__ __host__
#endif
bool parallelGodunovEikonalScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
{

	   const String& meshFile = parameters.getParameter< String >( "mesh" );
	   //MeshType originalMesh;
	   if( ! originalMesh.load( meshFile ) )
	   {
		   //cerr << "I am not able to load the mesh from the file " << meshFile << "." <<std::endl;
		   return false;
	   }




	   hx = originalMesh.template getSpaceStepsProducts< 1, 0 >();
	   ihx = 1.0/hx;
	   hy = originalMesh.template getSpaceStepsProducts< 0, 1 >();
	   ihy = 1.0/hy;

	   this->epsilon = parameters. getParameter< double >( "epsilon" );

	   this->epsilon *=sqrt( hx*hx + hy*hy );

//	   dofVector. setSize( this->mesh.getDofs() );

	   return true;

}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef __CUDACC__
   __device__ __host__
#endif
String parallelGodunovEikonalScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
   return String( "parallelGodunovEikonalScheme< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
#ifdef __CUDACC__
__device__ __host__
#endif
Real parallelGodunovEikonalScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType boundaryCondition,
          	  	  	  	  	  	  	  	  	  	                     const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities ) const
{


	if ( ((coordinates.x() == 0 && (boundaryCondition & 4)) or
		 (coordinates.x() == mesh.getDimensions().x() - 1 && (boundaryCondition & 2)) or
		 (coordinates.y() == 0 && (boundaryCondition & 8)) or
		 (coordinates.y() == mesh.getDimensions().y() - 1  && (boundaryCondition & 1)))
		 /*and
		 !(		 (coordinates.y() == 0 or coordinates.y() == mesh.getDimensions().y() - 1)
				 and
				 ( coordinates.x() == 0 or coordinates.x() == mesh.getDimensions().x() - 1)
		  )*/
		)
	{
		return 0.0;
	}


	//-----------------------------------

	RealType signui;
	signui = sign(u[cellIndex],this->epsilon);


#ifdef __CUDACC__
	//printf("%d   :    %d ;;;; %d   :   %d  , %f \n",threadIdx.x, mesh.getDimensions().x() , threadIdx.y,mesh.getDimensions().y(), epsilon );
#endif

	RealType xb = u[cellIndex];
	RealType xf = -u[cellIndex];
	RealType yb = u[cellIndex];
	RealType yf = -u[cellIndex];
	RealType a,b,c;


	   if(coordinates.x() == mesh.getDimensions().x() - 1)
		   xf += u[neighborEntities.template getEntityIndex< -1,  0 >()];
	   else
		   xf += u[neighborEntities.template getEntityIndex< 1,  0 >()];

	   if(coordinates.x() == 0)
		   xb -= u[neighborEntities.template getEntityIndex< 1,  0 >()];
	   else
		   xb -= u[neighborEntities.template getEntityIndex< -1,  0 >()];

	   if(coordinates.y() == mesh.getDimensions().y() - 1)
		   yf += u[neighborEntities.template getEntityIndex< 0,  -1 >()];
	   else
		   yf += u[neighborEntities.template getEntityIndex< 0,  1 >()];

	   if(coordinates.y() == 0)
		   yb -= u[neighborEntities.template getEntityIndex< 0,  1 >()];
	   else
		   yb -= u[neighborEntities.template getEntityIndex< 0,  -1 >()];


	   //xb *= ihx;
	   //xf *= ihx;
	  // yb *= ihy;
	   //yf *= ihy;

	   if(signui > 0.0)
	   {
		   xf = negativePart(xf);

		   xb = positivePart(xb);

		   yf = negativePart(yf);

		   yb = positivePart(yb);

	   }
	   else if(signui < 0.0)
	   {

		   xb = negativePart(xb);

		   xf = positivePart(xf);

		   yb = negativePart(yb);

		   yf = positivePart(yf);
	   }


	   if(xb - xf > 0.0)
		   a = xb;
	   else
		   a = xf;

	   if(yb - yf > 0.0)
		   b = yb;
	   else
		   b = yf;

	   c =(1.0 - sqrt(a*a+b*b)*ihx );

	   if(sign(c) > 0.0 )
		   return sign(u[cellIndex])*c;
	   else
		   return signui*c;

	   //--------------------------------------------------

//
//
//
//	RealType acc = hx*hy*hx*hy;
//
//	RealType nabla, xb, xf, yb, yf, signui;
//
//	signui = sign(u[cellIndex],this->epsilon);
//
//
//	//if(fabs(u[cellIndex]) < acc) return 0.0;
//
//	   if(signui > 0.0)
//	   {
//	/**/ /*  if(boundaryCondition & 2)
//			   xf = (u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/hx;
//		   else *//*if(boundaryCondition & 4)
//			   xf = 0.0;
//		   else /**/if(coordinates.x() == mesh.getDimensions().x() - 1)
//			   xf = negativePart((u[neighborEntities.template getEntityIndex< -1,  0 >()] - u[cellIndex])*ihx);
//		   else
//			   xf = negativePart((u[neighborEntities.template getEntityIndex< 1,  0 >()] - u[cellIndex])*ihx);
//
//	/**/ /*  if(boundaryCondition & 4)
//			   xb = (u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])*ihx;
//		   else *//*if(boundaryCondition & 2)
//			   xb = 0.0;
//		   else /**/if(coordinates.x() == 0)
//			   xb = positivePart((u[cellIndex] - u[mesh.template getCellNextToCell<+1,0>( cellIndex )])*ihx);
//		   else
//			   xb = positivePart((u[cellIndex] - u[neighborEntities.template getEntityIndex< -1,  0 >()])*ihx);
//
//	/**/  /* if(boundaryCondition & 1)
//			   yf = (u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])*ihy;
//		   else *//*if(boundaryCondition & 8)
//			   yf = 0.0;
//		   else /**/if(coordinates.y() == mesh.getDimensions().y() - 1)
//			   yf = negativePart((u[neighborEntities.template getEntityIndex< 0,  -1 >()] - u[cellIndex])*ihy);
//		   else
//			   yf = negativePart((u[neighborEntities.template getEntityIndex< 0,  1 >()] - u[cellIndex])*ihy);
//
//	/**/  /* if(boundaryCondition & 8)
//			   yb = (u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])*ihy;
//		   else *//*if(boundaryCondition & 1)
//			   yb = 0.0;
//		   else /**/if(coordinates.y() == 0)
//			   yb = positivePart((u[cellIndex] - u[neighborEntities.template getEntityIndex< 0,  1 >()])*ihy);
//		   else
//			   yb = positivePart((u[cellIndex] - u[neighborEntities.template getEntityIndex< 0,  -1 >()])*ihy);
//
//		   if(xb - xf > 0.0)
//			   xf = 0.0;
//		   else
//			   xb = 0.0;
//
//		   if(yb - yf > 0.0)
//			   yf = 0.0;
//		   else
//			   yb = 0.0;
//
//		   nabla = sqrt (xf*xf + xb*xb + yf*yf + yb*yb );
//		   if(fabs(1.0-nabla) < acc)
//			   return 0.0;
//		   return signui*(1.0 - nabla);
//	   }
//	   else if (signui < 0.0)
//	   {
//
//	/**/  /* if(boundaryCondition & 2)
//			   xf = (u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])*ihx;
//		   else*//* if(boundaryCondition & 4)
//			   xf = 0.0;
//		   else /**/if(coordinates.x() == mesh.getDimensions().x() - 1)
//			   xf = positivePart((u[neighborEntities.template getEntityIndex< -1,  0 >()] - u[cellIndex])*ihx);
//		   else
//			   xf = positivePart((u[neighborEntities.template getEntityIndex< 1,  0 >()] - u[cellIndex])*ihx);
//
//	/**/  /* if(boundaryCondition & 4)
//			   xb = (u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])*ihx;
//		   else*//* if(boundaryCondition & 2)
//			   xb = 0.0;
//		   else /**/if(coordinates.x() == 0)
//			   xb = negativePart((u[cellIndex] - u[neighborEntities.template getEntityIndex< 1,  0 >()])*ihx);
//		   else
//			   xb = negativePart((u[cellIndex] - u[neighborEntities.template getEntityIndex< -1,  0 >()])*ihx);
//
//	/**/ /*  if(boundaryCondition & 1)
//			   yf = (u[mesh.getCellYSuccessor( cellIndex )] - u[cellIndex])*ihy;
//		   else *//*if(boundaryCondition & 8)
//			   yf = 0.0;
//		   else /**/if(coordinates.y() == mesh.getDimensions().y() - 1)
//			   yf = positivePart((u[neighborEntities.template getEntityIndex< 0,  -1 >()] - u[cellIndex])*ihy);
//		   else
//			   yf = positivePart((u[neighborEntities.template getEntityIndex< 0,  1 >()] - u[cellIndex])*ihy);
//
//	/**/  /* if(boundaryCondition & 8)
//			   yb = (u[cellIndex] - u[mesh.getCellYPredecessor( cellIndex )])*ihy;
//		   else*//* if(boundaryCondition & 1)
//			   yb = 0.0;
//		   else /**/if(coordinates.y() == 0)
//			   yb = negativePart((u[cellIndex] - u[neighborEntities.template getEntityIndex< 0,  1 >()])*ihy);
//		   else
//			   yb = negativePart((u[cellIndex] - u[neighborEntities.template getEntityIndex< 0,  -1 >()])*ihy);
//
//
//		   if(xb - xf > 0.0)
//			   xf = 0.0;
//		   else
//			   xb = 0.0;
//
//		   if(yb - yf > 0.0)
//			   yf = 0.0;
//		   else
//			   yb = 0.0;
//
//		   nabla = sqrt (xf*xf + xb*xb + yf*yf + yb*yb );
//
//		   if(fabs(1.0-nabla) < acc)
//			   return 0.0;
//		   return signui*(1.0 - nabla);
//	   }
//	   else
//	   {
//		   return 0.0;
//	   }

}









template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >

#ifdef __CUDACC__
__device__
#endif
Real parallelGodunovEikonalScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >:: getValueDev( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real* u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType boundaryCondition,
          	  	  	  	  	  	  	  	  	  	                     const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities) const
{

	RealType signui;
	if(boundaryCondition == 0)
		signui = sign(u[cellIndex],/*(boundaryCondition != 0) * */this->epsilon);
	else
		signui = sign(u[cellIndex]);

	RealType xb = u[cellIndex];
	RealType xf = -u[cellIndex];
	RealType yb = u[cellIndex];
	RealType yf = -u[cellIndex];
	RealType a,b/*,c*/;


	   if(coordinates.x() == mesh.getDimensions().x() - 1)
		   xf += u[neighborEntities.template getEntityIndex< -1,  0 >()];
	   else
		   xf += u[neighborEntities.template getEntityIndex< 1,  0 >()];

	   if(coordinates.x() == 0)
		   xb -= u[neighborEntities.template getEntityIndex< 1,  0 >()];
	   else
		   xb -= u[neighborEntities.template getEntityIndex< -1,  0 >()];

	   if(coordinates.y() == mesh.getDimensions().y() - 1)
		   yf += u[neighborEntities.template getEntityIndex< 0,  -1 >()];
	   else
		   yf += u[neighborEntities.template getEntityIndex< 0,  1 >()];

	   if(coordinates.y() == 0)
		   yb -= u[neighborEntities.template getEntityIndex< 0,  1 >()];
	   else
		   yb -= u[neighborEntities.template getEntityIndex< 0,  -1 >()];


	   if(signui > 0.0)
	   {
		   xf = negativePart(xf);

		   xb = positivePart(xb);

		   yf = negativePart(yf);

		   yb = positivePart(yb);

	   }
	   else if(signui < 0.0)
	   {

		   xb = negativePart(xb);

		   xf = positivePart(xf);

		   yb = negativePart(yb);

		   yf = positivePart(yf);
	   }


	   if(xb > xf)
		   a = xb;
	   else
		   a = xf;

	   if(yb > yf)
		   b = yb;
	   else
		   b = yf;

//	   c =(1.0 - sqrt(a*a+b*b)*ihx );

//	   if(c > 0.0 )
//		   return sign(u[cellIndex])*c;
//	   else
		   return signui*(1.0 - sqrt(a*a+b*b)*ihx );
}


#endif /* PARALLELGODUNOVEIKONAL2D_IMPL_H_ */
