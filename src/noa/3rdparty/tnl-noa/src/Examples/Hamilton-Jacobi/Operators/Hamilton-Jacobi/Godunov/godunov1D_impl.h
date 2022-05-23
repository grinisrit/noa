#ifndef GODUNOV1D_IMPL_H_
#define GODUNOV1D_IMPL_H_



template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
Real godunovScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: positivePart(const Real arg) const
{
	if(arg > 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
Real  godunovScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: negativePart(const Real arg) const
{
	if(arg < 0.0)
		return arg;
	return 0.0;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
Real godunovScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: sign(const Real x, const Real eps) const
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
          typename Index,
		  typename Function >
bool godunovScheme< tnlGrid< 1,MeshReal, Device, MeshIndex >, Real, Index, Function > :: init( const Config::ParameterContainer& parameters )
{
	   const String& meshFile = parameters.getParameter< String >( "mesh" );
	   if( ! this->originalMesh.load( meshFile ) )
	   {
		  std::cerr << "I am not able to load the mesh from the file " << meshFile << "." <<std::endl;
		   return false;
	   }


	   h = originalMesh.getSpaceSteps().x();

	   epsilon = parameters. getParameter< double >( "epsilon" );

	   if(epsilon != 0.0)
		   epsilon *=h;

	   f.setup( parameters );

	   /*dofVector. setSize( this->mesh.getDofs() );*/

	   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
String godunovScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, Function > :: getType()
{
   return String( "godunovScheme< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
		  typename Function >
template< typename Vector >
#ifdef HAVE_CUDA
__device__ __host__
#endif
Real godunovScheme< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index, Function >:: getValue( const MeshType& mesh,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const IndexType cellIndex,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const CoordinatesType& coordinates,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Vector& u,
          	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	  	 const Real& time ) const
{

	RealType nabla, xb, xf, fi;

	fi = f.getValue(mesh.getCellCenter( coordinates ),time);

	   if(fi > 0.0)
	   {
		   xf = negativePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/h);
		   xb = positivePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/h);

		   if(xb + xf > 0.0)
			   xf = 0.0;
		   else
			   xb = 0.0;

		   nabla = sqrt (xf*xf + xb*xb );

		   return -fi*( nabla);
	   }
	   else if (fi < 0.0)
	   {
		   xf = positivePart((u[mesh.getCellXSuccessor( cellIndex )] - u[cellIndex])/h);
		   xb = negativePart((u[cellIndex] - u[mesh.getCellXPredecessor( cellIndex )])/h);

		   if(xb + xf > 0.0)
			   xb = 0.0;
		   else
			   xf = 0.0;

		   nabla = sqrt (xf*xf + xb*xb );

		   return -fi*( nabla);
	   }
	   else
	   {
		   return 0.0;
	   }

}


#endif /* GODUNOV1D_IMPL_H_ */
