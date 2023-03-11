#ifndef PARALLELGODUNOVMAP2D_IMPL_H_
#define PARALLELGODUNOVMAP2D_IMPL_H_


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
#ifdef __CUDACC__
   __device__ __host__
#endif
Real parallelGodunovMapScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index >:: positivePart(const Real arg) const
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
Real  parallelGodunovMapScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: negativePart(const Real arg) const
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
Real parallelGodunovMapScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: sign(const Real x, const Real eps) const
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
bool parallelGodunovMapScheme< tnlGrid< 2,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
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
String parallelGodunovMapScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
   return String( "parallelGodunovMapScheme< " ) +
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
Real parallelGodunovMapScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType boundaryCondition,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& map) const
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


	RealType signui;
//	if(boundaryCondition == 0)
		signui = sign(u[cellIndex],/*(boundaryCondition != 0) * */this->epsilon);
//	else
//		signui = sign(u[cellIndex]);

	RealType value;
//	if(map[cellIndex] == 0.0)
//	{
////		value = INT_MAX;
//		u[cellIndex] = sign(u[cellIndex])*INT_MAX;
//		return 0.0;
//	}
//	else
		value = 1.0/ map[cellIndex];


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

	   /*if(boundaryCondition &1)
		   yb = 0.0;
	   else
		   yf = 0.0;

	   if(boundaryCondition & 2)
		   xb = 0.0;
	   else
		   xf = 0.0;*/

	   if(xb - xf > 0.0)
		   a = xb;
	   else
		   a = xf;

	   if(yb - yf > 0.0)
		   b = yb;
	   else
		   b = yf;

	   c =(value - sqrt(a*a+b*b)*ihx );

	   if(c > 0.0 )
		   return sign(u[cellIndex])*c;
	   else

	   return signui*c;

}









template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >

#ifdef __CUDACC__
__device__
#endif
Real parallelGodunovMapScheme< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >:: getValueDev( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real* u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType boundaryCondition,
          	  	  	  	  	  	  	  	  	  	                     const tnlNeighborGridEntityGetter<tnlGridEntity< MeshType, 2, tnlGridEntityNoStencilStorage >,2> neighborEntities,
          	  	  	  	  	  	  	  	  	  	                     const Real* map) const
{
//	int gid = (blockDim.y*blockIdx.y + threadIdx.y)*blockDim.x*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;

	RealType signui;
	if(boundaryCondition == 0)
		signui = sign(u[cellIndex],/*(boundaryCondition != 0) * */this->epsilon);
	else
		signui = sign(u[cellIndex]);

//	RealType value;
//	if(map[cellIndex] == 0.0)
//	{
////		value = INT_MAX;
//		u[cellIndex] = sign(u[cellIndex])*INT_MAX;
//		return 0.0;
//	}
//	else
	RealType value = /*1.0/*/ map[cellIndex];


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

//	   if(boundaryCondition &1)
//		   yb = 0.0;
//	   else
//		   yf = 0.0;
//
//	   if(boundaryCondition & 2)
//		   xb = 0.0;
//	   else
//		   xf = 0.0;

	   if(xb > xf)
		   a = xb;
	   else
		   a = xf;

	   if(yb > yf)
		   b = yb;
	   else
		   b = yf;

	   c = (value - sqrt(a*a+b*b)*ihx );

//	   if(c > 0.0 )
//		   return sign(u[cellIndex])*c;
//	   else
		   return signui*c;
}


#endif /* PARALLELGODUNOVMAP2D_IMPL_H_ */
