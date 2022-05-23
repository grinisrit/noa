#pragma once


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real godunovEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: positivePart(const Real arg) const
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
Real  godunovEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: negativePart(const Real arg) const
{
	if(arg < 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
Real godunovEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: sign(const Real x, const Real eps) const
{
	if(x > eps)
		return 1.0;
	else if (x < -eps)
		return (-1.0);

	if(eps == 0.0)
		return 0.0;

	return sin( ( M_PI * x ) / (2 * eps) );
}





template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
bool godunovEikonalScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index > :: init( const Config::ParameterContainer& parameters )
{


	   const String& meshFile = parameters.getParameter< String >( "mesh" );
	   if( ! this->originalMesh.load( meshFile ) )
	   {
		  std::cerr << "I am not able to load the mesh from the file " << meshFile << "." <<std::endl;
		   return false;
	   }


	   h = originalMesh.template getSpaceSteps().x();
	  std::cout << "h = " << h <<std::endl;

	   epsilon = parameters. getParameter< double >( "epsilon" );

	   if(epsilon != 0.0)
		   epsilon *=h;

	   /*dofVector. setSize( this->mesh.getDofs() );*/

	   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String godunovEikonalScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index > :: getType()
{
   return String( "godunovEikonalScheme< " ) +
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
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real godunovEikonalScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time ) const
{
/*
	RealType nabla, xb, xf, signui;

	signui = sign(u[cellIndex],epsilon);

	   if(signui > 0.0)
	   {
		   xf = negativePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/h);
		   xb = positivePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/h);

		   if(xb + xf > 0.0)
			   xf = 0.0;
		   else
			   xb = 0.0;

		   nabla = sqrt (xf*xf + xb*xb );

		   return signui*(1.0 - nabla);
	   }
	   else if (signui < 0.0)
	   {
		   xf = positivePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/h);
		   xb = negativePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/h);

		   if(xb + xf > 0.0)
			   xb = 0.0;
		   else
			   xf = 0.0;

		   nabla = sqrt (xf*xf + xb*xb );

		   return signui*(1.0 - nabla);
	   }
	   else
*/	   {
		   return 0.0;
	   }

}

