#pragma once

template< typename Real,
        typename Device,
        typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
        MeshFunctionPointer& _output,
        InterfaceMapPointer& _interfaceMap, 
        StaticVector vLower, StaticVector vUpper )
{
  if( std::is_same< Device, Devices::Cuda >::value )
  {
#ifdef __CUDACC__
    const MeshType& mesh = _input->getMesh();
    
    const int cudaBlockSize( 8 );
    int numBlocksX = Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
    int numBlocksY = Cuda::getNumberOfBlocks( mesh.getDimensions().y(), cudaBlockSize );
    int numBlocksZ = Cuda::getNumberOfBlocks( mesh.getDimensions().z(), cudaBlockSize );
    if( cudaBlockSize * cudaBlockSize * cudaBlockSize > 1024 || numBlocksX > 1024 || numBlocksY > 1024 || numBlocksZ > 64 )
      std::cout << "Invalid kernel call. Dimensions of grid are max: [1024,1024,64], and maximum threads per block are 1024!" << std::endl;
    dim3 blockSize( cudaBlockSize, cudaBlockSize, cudaBlockSize );
    dim3 gridSize( numBlocksX, numBlocksY, numBlocksZ );
    Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
    CudaInitCaller3d<<< gridSize, blockSize >>>( _input.template getData< Device >(),
            _output.template modifyData< Device >(),
            _interfaceMap.template modifyData< Device >(), vLower, vUpper );
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
#endif
  }
  if( std::is_same< Device, Devices::Host >::value )
  {
    const MeshFunctionType& input =  _input.getData();
    MeshFunctionType& output =  _output.modifyData();
    InterfaceMapType& interfaceMap =  _interfaceMap.modifyData();
    
    const MeshType& mesh = input.getMesh();
    typedef typename MeshType::Cell Cell;
    
    Cell cell( mesh );
    for( cell.getCoordinates().z() = 0;
            cell.getCoordinates().z() < mesh.getDimensions().z();
            cell.getCoordinates().z() ++ )
      for( cell.getCoordinates().y() = 0;
              cell.getCoordinates().y() < mesh.getDimensions().y();
              cell.getCoordinates().y() ++ )
        for( cell.getCoordinates().x() = 0;
                cell.getCoordinates().x() < mesh.getDimensions().x();
                cell.getCoordinates().x() ++ )
        {
          cell.refresh();
          output[ cell.getIndex() ] =
                  input( cell ) > 0 ? std::numeric_limits< RealType >::max() :
                    - std::numeric_limits< RealType >::max();
          interfaceMap[ cell.getIndex() ] = false;
        }
    
    const RealType& hx = mesh.getSpaceSteps().x();
    const RealType& hy = mesh.getSpaceSteps().y();
    const RealType& hz = mesh.getSpaceSteps().z();
    for( cell.getCoordinates().z() = 0 + vLower[2];
            cell.getCoordinates().z() < mesh.getDimensions().z() - vUpper[2];
            cell.getCoordinates().z() ++ )   
      for( cell.getCoordinates().y() = 0 + vLower[1];
              cell.getCoordinates().y() < mesh.getDimensions().y() - vUpper[1];
              cell.getCoordinates().y() ++ )
        for( cell.getCoordinates().x() = 0 + vLower[0];
                cell.getCoordinates().x() < mesh.getDimensions().x() - vUpper[0];
                cell.getCoordinates().x() ++ )
        {
          cell.refresh();
          const RealType& c = input( cell );
          if( ! cell.isBoundaryEntity() )
          {
            auto neighbors = cell.getNeighborEntities();
            Real pom = 0;
            const IndexType e = neighbors.template getEntityIndex<  1,  0,  0 >();
            const IndexType n = neighbors.template getEntityIndex<  0,  1,  0 >();
            const IndexType t = neighbors.template getEntityIndex<  0,  0,  1 >();
            
            
            if( c * input[ n ] <= 0 )
            {
              pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
              if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) ) 
                output[ cell.getIndex() ] = pom;
              pom = pom - TNL::sign( c )*hy;
              if( TNL::abs( output[ n ] ) > TNL::abs( pom ) )
                output[ n ] = pom; //( hy * c )/( c - input[ n ]) - hy;
              
              interfaceMap[ cell.getIndex() ] = true;
              interfaceMap[ n ] = true;
            }
            
            if( c * input[ e ] <= 0 )
            {
              pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
              if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) ) 
                output[ cell.getIndex() ] = pom;
              pom = pom - TNL::sign( c )*hx;
              if( TNL::abs( output[ e ] ) > TNL::abs( pom ) )
                output[ e ] = pom; 
              
              interfaceMap[ cell.getIndex() ] = true;
              interfaceMap[ e ] = true;
            }
            
            if( c * input[ t ] <= 0 )
            {
              pom = TNL::sign( c )*( hz * c )/( c - input[ t ]);
              if( TNL::abs( output[ cell.getIndex() ] ) > TNL::abs( pom ) ) 
                output[ cell.getIndex() ] = pom;
              pom = pom - TNL::sign( c )*hz;
              if( TNL::abs( output[ t ] ) > TNL::abs( pom ) )
                output[ t ] = pom; 
              
              interfaceMap[ cell.getIndex() ] = true;
              interfaceMap[ t ] = true;
            }  
          }
        }
  }
}

template< typename Real,
        typename Device,
        typename Index >
template< typename MeshEntity >
__cuda_callable__
bool
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
        const MeshEntity& cell, 
        const RealType v )
{
  const auto& neighborEntities = cell.template getNeighborEntities< 3 >();
  const MeshType& mesh = cell.getMesh();
  
  const RealType& hx = mesh.getSpaceSteps().x();
  const RealType& hy = mesh.getSpaceSteps().y();
  const RealType& hz = mesh.getSpaceSteps().z();
  const RealType value = u( cell );
  //std::cout << value << std::endl;
  RealType a, b, c, tmp = std::numeric_limits< RealType >::max();
  
  
  if( cell.getCoordinates().x() == 0 )
    a = u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ];
  else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
    a = u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ];
  else
  {
    a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1, 0, 0 >() ],
            u[ neighborEntities.template getEntityIndex< 1, 0, 0 >() ] );
  }
  
  if( cell.getCoordinates().y() == 0 )
    b = u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ];
  else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
    b = u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ];
  else
  {
    b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, -1, 0 >() ],
            u[ neighborEntities.template getEntityIndex< 0, 1, 0 >() ] );
  }
  
  if( cell.getCoordinates().z() == 0 )
    c = u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ];
  else if( cell.getCoordinates().z() == mesh.getDimensions().z() - 1 )
    c = u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ];
  else
  {
    c = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0, 0, -1 >() ],
            u[ neighborEntities.template getEntityIndex< 0, 0, 1 >() ] );
  }
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() && 
          fabs( b ) == std::numeric_limits< RealType >::max() &&
          fabs( c ) == std::numeric_limits< RealType >::max() )
    return false;
  
  RealType pom[6] = { a, b, c, hx, hy, hz};
  tmp = getNewValue( pom , value, v );
  
  u[ cell.getIndex() ] = tmp;
  tmp = value - u[ cell.getIndex() ];
  if ( fabs( tmp ) >  0.001*hx )
    return true;
  else
    return false;
  /*sortMinims( pom );   
  tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ];
  
  if( fabs( tmp ) < fabs( pom[ 1 ] ) )
  {
    u[ cell.getIndex() ] = argAbsMin( value, tmp );
    tmp = value - u[ cell.getIndex() ];
    if ( fabs( tmp ) > 0.001*hx ){
      return true;
    }else{
      return false;
    }
  }
  else
  {
    tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
    if( fabs( tmp ) < fabs( pom[ 2 ]) ) 
    {
      u[ cell.getIndex() ] = argAbsMin( value, tmp );
      tmp = value - u[ cell.getIndex() ];
      if ( fabs( tmp ) > 0.001*hx ){
        return true;
      }else{
        return false;
      }
    }
    else
    {
      tmp = ( hy * hy * hz * hz * a + hx * hx * hz * hz * b + hx * hx * hy * hy * c +
              TNL::sign( value ) * hx * hy * hz * TNL::sqrt( ( hx * hx * hz * hz + hy * hy * hz * hz + hx * hx * hy * hy)/( v * v ) - 
              hz * hz * ( a - b ) * ( a - b ) - hy * hy * ( a - c ) * ( a - c ) -
              hx * hx * ( b - c ) * ( b - c ) ) )/( hx * hx * hy * hy + hy * hy * hz * hz + hz * hz * hx *hx );
      u[ cell.getIndex() ] = argAbsMin( value, tmp );
      tmp = value - u[ cell.getIndex() ];
      if ( fabs( tmp ) > 0.001*hx ){
        return true;
      }else{
        return false;
      }
    }
  }*/
}

template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
__cuda_callable__ 
bool 
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateCell( volatile Real *sArray, int thri, int thrj, int thrk,
        const Real hx, const Real hy, const Real hz, const Real v )
{
  const RealType value = sArray[thrk *sizeSArray * sizeSArray + thrj * sizeSArray + thri];
  
  RealType a, b, c, tmp = std::numeric_limits< RealType >::max();
  
  c = TNL::argAbsMin( sArray[ (thrk+1)* sizeSArray*sizeSArray + thrj * sizeSArray + thri ],
          sArray[ (thrk-1) * sizeSArray *sizeSArray + thrj* sizeSArray + thri ] );
  
  b = TNL::argAbsMin( sArray[ thrk* sizeSArray*sizeSArray + (thrj+1) * sizeSArray + thri ],
          sArray[ thrk* sizeSArray * sizeSArray + (thrj-1)* sizeSArray +thri ] );
  
  a = TNL::argAbsMin( sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri+1 ],
          sArray[ thrk* sizeSArray * sizeSArray + thrj* sizeSArray +thri-1 ] );
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() && 
          fabs( b ) == std::numeric_limits< RealType >::max() &&
          fabs( c ) == std::numeric_limits< RealType >::max() )
    return false;
  
  RealType pom[6] = { a, b, c, hx, hy, hz};
  
  tmp = getNewValue( pom , value, v );
  
  sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ] = tmp;
  tmp = value - sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ];
  if ( fabs( tmp ) >  0.001*hx )
    return true;
  else
    return false;
  /*sortMinims( pom );
  
  // calculation of real value taken from ZHAO
  tmp = pom[ 0 ] + TNL::sign( value ) * pom[ 3 ]/v;
  if( fabs( tmp ) < fabs( pom[ 1 ] ) ) 
  {
    sArray[ thrk* sizeSArray* sizeSArray + thrj* sizeSArray + thri ] = argAbsMin( value, tmp );
    tmp = value - sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ];
    if ( fabs( tmp ) >  0.001*hx )
      return true;
    else
      return false;
  }
  else
  {
    tmp = ( pom[ 3 ] * pom[ 3 ] * pom[ 1 ] + pom[ 4 ] * pom[ 4 ] * pom[ 0 ] + 
            TNL::sign( value ) * pom[ 3 ] * pom[ 4 ] * TNL::sqrt( ( pom[ 3 ] * pom[ 3 ] +  pom[ 4 ] *  pom[ 4 ] )/( v * v ) - 
            ( pom[ 1 ] - pom[ 0 ] ) * ( pom[ 1 ] - pom[ 0 ] ) ) )/( pom[ 3 ] * pom[ 3 ] + pom[ 4 ] * pom[ 4 ] );
    if( fabs( tmp ) < fabs( pom[ 2 ]) ) 
    {
      sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ] = argAbsMin( value, tmp );
      tmp = value - sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ];
      if ( fabs( tmp ) > 0.001*hx )
        return true;
      else
        return false;
    }
    else
    {
      tmp = ( hy * hy * hz * hz * a + hx * hx * hz * hz * b + hx * hx * hy * hy * c +
              TNL::sign( value ) * hx * hy * hz * TNL::sqrt( ( hx * hx * hz * hz + hy * hy * hz * hz + hx * hx * hy * hy)/( v * v ) - 
              hz * hz * ( a - b ) * ( a - b ) - hy * hy * ( a - c ) * ( a - c ) -
              hx * hx * ( b - c ) * ( b - c ) ) )/( hx * hx * hy * hy + hy * hy * hz * hz + hz * hz * hx *hx );
      sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ] = argAbsMin( value, tmp );
      tmp = value - sArray[ thrk* sizeSArray* sizeSArray  + thrj* sizeSArray + thri ];
      if ( fabs( tmp ) > 0.001*hx )
        return true;
      else
        return false;
    }
  }*/
  
}


template< typename Real,
        typename Device,
        typename Index >
__cuda_callable__ 
Real
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
getNewValue( RealType valuesAndSteps[], const RealType originalValue, const RealType v )
{
  RealType newValue = std::numeric_limits< RealType >::max();
  sortMinims( valuesAndSteps );
  
  // calculation of real value taken from ZHAO
  newValue = valuesAndSteps[ 0 ] + TNL::sign( originalValue ) * valuesAndSteps[ 3 ]/v;
  if( fabs( newValue ) < fabs( valuesAndSteps[ 1 ] ) ) 
  {
    newValue = argAbsMin( originalValue, newValue );
  }
  else
  {
    newValue = ( valuesAndSteps[ 3 ] * valuesAndSteps[ 3 ] * valuesAndSteps[ 1 ] + 
            valuesAndSteps[ 4 ] * valuesAndSteps[ 4 ] * valuesAndSteps[ 0 ] + 
            TNL::sign( originalValue ) * valuesAndSteps[ 3 ] * valuesAndSteps[ 4 ] * 
            TNL::sqrt( ( valuesAndSteps[ 3 ] * valuesAndSteps[ 3 ] +  valuesAndSteps[ 4 ] *  valuesAndSteps[ 4 ] )/( v * v ) - 
            ( valuesAndSteps[ 1 ] - valuesAndSteps[ 0 ] ) * 
            ( valuesAndSteps[ 1 ] - valuesAndSteps[ 0 ] ) ) )/
            ( valuesAndSteps[ 3 ] * valuesAndSteps[ 3 ] + valuesAndSteps[ 4 ] * valuesAndSteps[ 4 ] );
    if( fabs( newValue ) < fabs( valuesAndSteps[ 2 ]) ) 
    {
      newValue = argAbsMin( originalValue, newValue );
    }
    else
    {
      // Value from algorithm by Zhao
      newValue = ( valuesAndSteps[4] * valuesAndSteps[4] * valuesAndSteps[5] * valuesAndSteps[5] * valuesAndSteps[0] +
              valuesAndSteps[3] * valuesAndSteps[3] * valuesAndSteps[5] * valuesAndSteps[5] * valuesAndSteps[1] +
              valuesAndSteps[3] * valuesAndSteps[3] * valuesAndSteps[4] * valuesAndSteps[4] * valuesAndSteps[2] +
              TNL::sign(originalValue) * valuesAndSteps[3] * valuesAndSteps[4] * valuesAndSteps[5] * TNL::sqrt(
              (valuesAndSteps[3] * valuesAndSteps[3] * valuesAndSteps[5] * valuesAndSteps[5] +
              valuesAndSteps[4] * valuesAndSteps[4] * valuesAndSteps[5] * valuesAndSteps[5] +
              valuesAndSteps[3] * valuesAndSteps[3] * valuesAndSteps[4] * valuesAndSteps[4]) / (v * v) -
              valuesAndSteps[5] * valuesAndSteps[5] * (valuesAndSteps[0] - valuesAndSteps[1]) * (valuesAndSteps[0] - valuesAndSteps[1]) -
              valuesAndSteps[4] * valuesAndSteps[4] * (valuesAndSteps[0] - valuesAndSteps[2]) * (valuesAndSteps[0] - valuesAndSteps[2]) -
              valuesAndSteps[3] * valuesAndSteps[3] * (valuesAndSteps[1] - valuesAndSteps[2]) * (valuesAndSteps[1] - valuesAndSteps[2]))) / (
              valuesAndSteps[3] * valuesAndSteps[3] * valuesAndSteps[4] * valuesAndSteps[4] +
              valuesAndSteps[4] * valuesAndSteps[4] * valuesAndSteps[5] * valuesAndSteps[5] +
              valuesAndSteps[5] * valuesAndSteps[5] * valuesAndSteps[3] * valuesAndSteps[3]);
      newValue = argAbsMin( originalValue, newValue );
    }
  }
  
  return newValue;
}


#ifdef __CUDACC__
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller3d( const Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& input, 
        Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& output,
        Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
        Containers::StaticVector< 3, Index > vLower, Containers::StaticVector< 3, Index > vUpper )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  int k = blockDim.z*blockIdx.z + threadIdx.z;
  const Meshes::Grid< 3, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
  
  if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() && k < mesh.getDimensions().z() )
  {
    typedef typename Meshes::Grid< 3, Real, Device, Index >::Cell Cell;
    Cell cell( mesh );
    cell.getCoordinates().x() = i; cell.getCoordinates().y() = j; cell.getCoordinates().z() = k;
    cell.refresh();
    const Index cind = cell.getIndex();
    
    
    output[ cind ] =
            input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
              - std::numeric_limits< Real >::max();
    interfaceMap[ cind ] = false; 
    cell.refresh();
    
    if( i < mesh.getDimensions().x() - vUpper[0] && j < mesh.getDimensions().y() - vUpper[1] &&
            k < mesh.getDimensions().y() - vUpper[2] && i>vLower[0]-1 && j> vLower[1]-1 && k>vLower[2]-1 )
    {
      const Real& hx = mesh.getSpaceSteps().x();
      const Real& hy = mesh.getSpaceSteps().y();
      const Real& hz = mesh.getSpaceSteps().z();
      const Real& c = input( cell );
      if( ! cell.isBoundaryEntity()  )
      {
        auto neighbors = cell.getNeighborEntities();
        Real pom = 0;
        const Index e = neighbors.template getEntityIndex<  1, 0, 0 >();
        const Index w = neighbors.template getEntityIndex<  -1, 0, 0 >();
        const Index n = neighbors.template getEntityIndex<  0, 1, 0 >();
        const Index s = neighbors.template getEntityIndex<  0, -1, 0 >();
        const Index t = neighbors.template getEntityIndex<  0, 0, 1 >();
        const Index b = neighbors.template getEntityIndex<  0, 0, -1 >();
        
        if( c * input[ n ] <= 0 )
        {
          pom = TNL::sign( c )*( hy * c )/( c - input[ n ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
            output[ cind ] = pom;
          
          interfaceMap[ cind ] = true;
        }
        if( c * input[ e ] <= 0 )
        {
          pom = TNL::sign( c )*( hx * c )/( c - input[ e ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) )
            output[ cind ] = pom;                       
          
          interfaceMap[ cind ] = true;
        }
        if( c * input[ w ] <= 0 )
        {
          pom = TNL::sign( c )*( hx * c )/( c - input[ w ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
            output[ cind ] = pom;
          
          interfaceMap[ cind ] = true;
        }
        if( c * input[ s ] <= 0 )
        {
          pom = TNL::sign( c )*( hy * c )/( c - input[ s ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
            output[ cind ] = pom;
          
          interfaceMap[ cind ] = true;
        }
        if( c * input[ b ] <= 0 )
        {
          pom = TNL::sign( c )*( hz * c )/( c - input[ b ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
            output[ cind ] = pom;
          
          interfaceMap[ cind ] = true;
        }
        if( c * input[ t ] <= 0 )
        {
          pom = TNL::sign( c )*( hz * c )/( c - input[ t ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( pom ) ) 
            output[ cind ] = pom;
          
          interfaceMap[ cind ] = true;
        }
      }
    }
  }
}


template < typename Index >
__global__ void GetNeighbours( TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterPom,
        int numBlockX, int numBlockY, int numBlockZ )
{
  int i = blockIdx.x * 1024 + threadIdx.x;
  
  if( i < numBlockX * numBlockY * numBlockZ )
  {
    int pom = 0;//BlockIterPom[ i ] = 0;
    int m=0, l=0, k=0;
    l = i/( numBlockX * numBlockY );
    k = (i-l*numBlockX * numBlockY )/(numBlockX );
    m = (i-l*numBlockX * numBlockY )%( numBlockX );
    if( m > 0 && BlockIterDevice[ i - 1 ] ){ // left neighbour
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterDevice[ i + 1 ] ){ // right neighbour
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterDevice[ i - numBlockX ] ){ // bottom neighbour
      pom = 1;// BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterDevice[ i + numBlockX ] ){ // top neighbour
      pom = 1;//BlockIterPom[ i ] = 1;
    }else if( l > 0 && BlockIterDevice[ i - numBlockX*numBlockY ] ){ // neighbour behind 
      pom = 1;
    }else if( l < numBlockZ-1 && BlockIterDevice[ i + numBlockX*numBlockY ] ){ // neighbour in front
      pom = 1;
    }
    
    if( !BlockIterDevice[ i ] ) // only in CudaUpdateCellCaller can BlockIterDevice gain 0
      BlockIterPom[ i ] = pom;
    else
      BlockIterPom[ i ] = 1;
  }
}


template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > > ptr,
        const Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index >, 3, bool >& interfaceMap,
        const Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& aux,
        Functions::MeshFunctionView< Meshes::Grid< 3, Real, Device, Index > >& helpFunc,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > BlockIterDevice,
        Containers::StaticVector< 3, Index > vecLowerOverlaps, Containers::StaticVector< 3, Index > vecUpperOverlaps )
{
  int thri = threadIdx.x; int thrj = threadIdx.y; int thrk = threadIdx.z;
  int i = threadIdx.x + blockDim.x*blockIdx.x + vecLowerOverlaps[0]; // WITH OVERLAPS!!! i,j,k aren't coordinates of all values
  int j = blockDim.y*blockIdx.y + threadIdx.y + vecLowerOverlaps[1];
  int k = blockDim.z*blockIdx.z + threadIdx.z + vecLowerOverlaps[2];
  int currentIndex = thrk * blockDim.x * blockDim.y + thrj * blockDim.x + thri;
  const Meshes::Grid< 3, Real, Device, Index >& mesh = interfaceMap.template getMesh< Devices::Cuda >();
  
  // should this block calculate?
  if( BlockIterDevice[ blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x ] ) 
  {
    __syncthreads();
    
    // Array indicates weather some threads calculated (for parallel reduction)
    __shared__ volatile bool changed[ (sizeSArray - 2)*(sizeSArray - 2)*(sizeSArray - 2) ];
    changed[ currentIndex ] = false;
    
    if( thrj == 0 && thri == 0 && thrk == 0 )
      changed[ 0 ] = true; // first indicates weather we should calculate again (princip of parallel reduction)
    
    //getting stepps and size of mesh
    const Real hx = mesh.getSpaceSteps().x(); const int dimX = mesh.getDimensions().x(); 
    const Real hy = mesh.getSpaceSteps().y(); const int dimY = mesh.getDimensions().y();
    const Real hz = mesh.getSpaceSteps().z(); const int dimZ  = mesh.getDimensions().z();
    
    if( thrj == 1 && thri == 1 && thrk == 1 )
    {
      // we dont know if we will calculate in here, more info down in code
      BlockIterDevice[ blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x ] = 0;
    }
    
    // sArray contains values of one block (coppied from aux) and edges (not MPI) of those blocks
    __shared__ volatile Real sArray[ sizeSArray * sizeSArray * sizeSArray ];
    sArray[(thrk+1)* sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thri+1] = std::numeric_limits< Real >::max();
    
    
    int xkolik = blockDim.x + 1;// maximum of threads in x direction (for all blocks different)
    int ykolik = blockDim.y + 1;
    int zkolik = blockDim.z + 1;
    __syncthreads();
    
    if( gridDim.x - 1 == blockIdx.x )
      xkolik = (dimX-vecUpperOverlaps[0]-vecLowerOverlaps[0]) -  blockIdx.x*blockDim.x+1;
    if( gridDim.y -1 == blockIdx.y )
      ykolik = (dimY-vecUpperOverlaps[1]-vecLowerOverlaps[1]) - (blockIdx.y)*blockDim.y+1;
    if( gridDim.z-1 == blockIdx.z )
      zkolik = (dimZ-vecUpperOverlaps[2]-vecLowerOverlaps[2]) - (blockIdx.z)*blockDim.z+1;
    __syncthreads();
    
     //filling sArray edges
    if( thri == 0 ) //x bottom
    {        
      if(  (blockIdx.x != 0 || vecLowerOverlaps[0] !=0) && thrj+1 < ykolik && thrk+1 < zkolik )
        sArray[ (thrk+1 )* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0 ] = 
                aux[ (blockIdx.z*blockDim.z + vecLowerOverlaps[2]) * dimX * dimY + (blockIdx.y * blockDim.y+vecLowerOverlaps[1])*dimX 
                + blockIdx.x*blockDim.x + thrj * dimX -1 + thrk*dimX*dimY + vecLowerOverlaps[0] ];
    else
        sArray[(thrk+1)* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 1 ) //xtop
    {
      if( dimX - vecLowerOverlaps[ 0 ] >  (blockIdx.x+1) * blockDim.x && thrj+1 < ykolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + xkolik ] =
                aux[ (blockIdx.z*blockDim.z + vecLowerOverlaps[2]) * dimX * dimY + (blockIdx.y * blockDim.y+vecLowerOverlaps[1])*dimX
                + blockIdx.x*blockDim.x + blockDim.x + thrj * dimX + thrk*dimX*dimY + vecLowerOverlaps[0] ];
     else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1)*sizeSArray + xkolik] = std::numeric_limits< Real >::max();
    }
    if( thri == 2 ) //y bottom
    {        
      if( (blockIdx.y != 0 || vecLowerOverlaps[1] !=0) && thrj+1 < xkolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray +0*sizeSArray + thrj+1] =
                aux[ (blockIdx.z*blockDim.z + vecLowerOverlaps[2]) * dimX * dimY + (blockIdx.y * blockDim.y+vecLowerOverlaps[1])*dimX
                + blockIdx.x*blockDim.x - dimX + thrj + thrk*dimX*dimY + vecLowerOverlaps[0] ];
      else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + 0*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 3 ) //y top
    {
      if( dimY - vecLowerOverlaps[ 1 ] > (blockIdx.y+1) * blockDim.y && thrj+1 < xkolik && thrk+1 < zkolik )
        sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] =
                aux[ (blockIdx.z*blockDim.z + vecLowerOverlaps[2]) * dimX * dimY + ((blockIdx.y+1) * blockDim.y+vecLowerOverlaps[1])*dimX
                + blockIdx.x*blockDim.x + thrj + thrk*dimX*dimY + vecLowerOverlaps[0] ];
     else
        sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    if( thri == 4 ) //z bottom
    {        
      if( (blockIdx.z != 0 || vecLowerOverlaps[2] !=0) && thrj+1 < ykolik && thrk+1 < xkolik )
        sArray[ 0 * sizeSArray * sizeSArray +(thrj+1 )* sizeSArray + thrk+1] =
                aux[ (blockIdx.z*blockDim.z + vecLowerOverlaps[2]) * dimX * dimY + (blockIdx.y * blockDim.y+vecLowerOverlaps[1])*dimX
                + blockIdx.x*blockDim.x - dimX * dimY + thrj * dimX + thrk + vecLowerOverlaps[0] ];
     else
        sArray[0 * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thrk+1] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 5 ) //z top
    {
      if( dimZ - vecLowerOverlaps[ 2 ] > (blockIdx.z+1) * blockDim.z && thrj+1 < ykolik && thrk+1 < xkolik )
        sArray[ zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] =
                aux[ ((blockIdx.z+1)*blockDim.z + vecLowerOverlaps[2]) * dimX * dimY + (blockIdx.y * blockDim.y+vecLowerOverlaps[1])*dimX
                + blockIdx.x*blockDim.x + thrj * dimX + thrk + vecLowerOverlaps[0] ];
     else
        sArray[zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] = std::numeric_limits< Real >::max();
    }
    
    // Copy all other values that aren't edges
    if( i - vecLowerOverlaps[0] < dimX && j - vecLowerOverlaps[1] < dimY && k - vecLowerOverlaps[2] < dimZ &&
        thri+1 < xkolik + vecUpperOverlaps[0] && thrj+1 < ykolik + vecUpperOverlaps[1] && thrk+1 < zkolik + vecUpperOverlaps[2] )
    {
      sArray[(thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + thri+1] = aux[ k*dimX*dimY + j*dimX + i ];
    }
    __syncthreads(); 
    
    //main while cycle. each value can get information only from neighbour but that information has to spread there
    while( changed[ 0 ] )
    {
      __syncthreads();
      
      changed[ currentIndex ] = false;
      
      //calculation of update cell
      if( i < dimX - vecUpperOverlaps[0] && j < dimY - vecUpperOverlaps[1] && k < dimZ - vecUpperOverlaps[2] )
      {
        if( ! interfaceMap[ k*dimX*dimY + j * dimX + i ] )
        {
          // calculate new value depending on neighbours in sArray on (thri+1, thrj+1) coordinates
          changed[ currentIndex ] = ptr.updateCell< sizeSArray >( sArray, thri+1, thrj+1, thrk+1, hx,hy,hz); 
        }
      }
      __syncthreads();
      
      //pyramid reduction (parallel reduction)
      if( blockDim.x*blockDim.y*blockDim.z == 1024 )
      {
        if( currentIndex < 512 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 512 ];
        }
      }
      __syncthreads();
      if( blockDim.x*blockDim.y*blockDim.z >= 512 )
      {
        if( currentIndex < 256 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 256 ];
        }
      }
      __syncthreads();
      if( blockDim.x*blockDim.y*blockDim.z >= 256 )
      {
        if( currentIndex < 128 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 128 ];
        }
      }
      __syncthreads();
      if( blockDim.x*blockDim.y*blockDim.z >= 128 )
      {
        if( currentIndex < 64 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 64 ];
        }
      }
      __syncthreads();
      if( currentIndex < 32 )
      {
        if( true ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 32 ];
        if( currentIndex < 16 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 16 ];
        if( currentIndex < 8 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 8 ];
        if( currentIndex < 4 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 4 ];
        if( currentIndex < 2 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 2 ];
        if( currentIndex < 1 ) changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 1 ];
      }
      __syncthreads();
      
      // if we calculated, then the BlockIterDevice should contain the info about this whole block! (only one number for one block)
      if( changed[ 0 ] && thri == 0 && thrj == 0 && thrk == 0 )
      {
        BlockIterDevice[ blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x ] = 1;
      }
      __syncthreads();
    }
    
    // copy results into helpFunc (not into aux bcs of conflicts)
    if( i < dimX && j < dimY && k < dimZ && thri+1 < xkolik && thrj+1 < ykolik && thrk+1 < zkolik )
      helpFunc[ k*dimX*dimY + j * dimX + i ] = sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thri+1 ];
    
  }
  else // if not, then it should at least copy the values from aux to helpFunc.
  {
    if( i < mesh.getDimensions().x() - vecUpperOverlaps[0] && j < mesh.getDimensions().y() - vecUpperOverlaps[1]
            && k < mesh.getDimensions().z() - vecUpperOverlaps[2])
      helpFunc[ k * mesh.getDimensions().x() * mesh.getDimensions().y() + j * mesh.getDimensions().x() + i ] =
              aux[ k * mesh.getDimensions().x() * mesh.getDimensions().y() + j * mesh.getDimensions().x() + i ];
  }
}  
#endif


/// ==========================OPEN=MP=================================
template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
updateBlocks( const InterfaceMapType interfaceMap,
        const MeshFunctionType aux,
        MeshFunctionType& helpFunc,
        ArrayContainer BlockIterHost, int numThreadsPerBlock/*, Real **sArray*/ )
{  
  #pragma omp parallel for schedule( dynamic )
  for( IndexType i = 0; i < BlockIterHost.getSize(); i++ )
  {
    if( BlockIterHost[ i ] )
    {
      MeshType mesh = interfaceMap.template getMesh< Devices::Host >();
      
      int dimX = mesh.getDimensions().x(); int dimY = mesh.getDimensions().y();
      int dimZ = mesh.getDimensions().z();
      //std::cout << "dimX = " << dimX << " ,dimY = " << dimY << std::endl;
      int numOfBlocky = dimY/numThreadsPerBlock + ((dimY%numThreadsPerBlock != 0) ? 1:0);
      int numOfBlockx = dimX/numThreadsPerBlock + ((dimX%numThreadsPerBlock != 0) ? 1:0);
      int numOfBlockz = dimZ/numThreadsPerBlock + ((dimZ%numThreadsPerBlock != 0) ? 1:0);
      //std::cout << "numOfBlockx = " << numOfBlockx << " ,numOfBlocky = " << numOfBlocky << std::endl;
      int xkolik = numThreadsPerBlock + 1;
      int ykolik = numThreadsPerBlock + 1;
      int zkolik = numThreadsPerBlock + 1;
      
      
      int blIdz = i/( numOfBlockx * numOfBlocky );
      int blIdy = (i-blIdz*numOfBlockx * numOfBlocky )/(numOfBlockx );
      int blIdx = (i-blIdz*numOfBlockx * numOfBlocky )%( numOfBlockx );
      //std::cout << "blIdx = " << blIdx << " ,blIdy = " << blIdy << std::endl;
      
      if( numOfBlockx - 1 == blIdx )
        xkolik = dimX - (blIdx)*numThreadsPerBlock+1;
      if( numOfBlocky -1 == blIdy )
        ykolik = dimY - (blIdy)*numThreadsPerBlock+1;
      if( numOfBlockz-1 == blIdz )
        zkolik = dimZ - (blIdz)*numThreadsPerBlock+1;
      //std::cout << "xkolik = " << xkolik << " ,ykolik = " << ykolik << std::endl;
      
      
      /*bool changed[numThreadsPerBlock*numThreadsPerBlock];
       changed[ 0 ] = 1;*/
      Real hx = mesh.getSpaceSteps().x();
      Real hy = mesh.getSpaceSteps().y();
      Real hz = mesh.getSpaceSteps().z();
      
      bool changed = false;
      BlockIterHost[ i ] = 0;
      
      
      Real *sArray;
      sArray = new Real[ sizeSArray * sizeSArray * sizeSArray ];
      if( sArray == nullptr )
        std::cout << "Error while allocating memory for sArray." << std::endl;
      
      for( IndexType k = 0; k < sizeSArray; k++ )
        for( IndexType l = 0; l < sizeSArray; l++ )
          for( IndexType m = 0; m < sizeSArray; m++ ){
            sArray[ m * sizeSArray * sizeSArray + k * sizeSArray + l ] = std::numeric_limits< Real >::max();
          }
      
      
      for( IndexType thrk = 0; thrk < numThreadsPerBlock; thrk++ )
        for( IndexType thrj = 0; thrj < numThreadsPerBlock; thrj++ )
        {
          if( blIdx != 0 && thrj+1 < ykolik && thrk+1 < zkolik )
            sArray[(thrk+1 )* sizeSArray * sizeSArray + (thrj+1)*sizeSArray + 0] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock + thrj * dimX -1 + thrk*dimX*dimY ];
          
          if( dimX > (blIdx+1) * numThreadsPerBlock && thrj+1 < ykolik && thrk+1 < zkolik )
            sArray[ (thrk+1) * sizeSArray * sizeSArray + (thrj+1) *sizeSArray + xkolik ] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy *numThreadsPerBlock*dimX+ blIdx*numThreadsPerBlock + numThreadsPerBlock + thrj * dimX + thrk*dimX*dimY ];
          
          if( blIdy != 0 && thrj+1 < xkolik && thrk+1 < zkolik )
            sArray[ (thrk+1) * sizeSArray * sizeSArray +0*sizeSArray + thrj+1] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock - dimX + thrj + thrk*dimX*dimY ];
          
          if( dimY > (blIdy+1) * numThreadsPerBlock && thrj+1 < xkolik && thrk+1 < zkolik )
            sArray[ (thrk+1) * sizeSArray * sizeSArray + ykolik*sizeSArray + thrj+1] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + (blIdy+1) * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock + thrj + thrk*dimX*dimY ];
          
          if( blIdz != 0 && thrj+1 < ykolik && thrk+1 < xkolik )
            sArray[ 0 * sizeSArray * sizeSArray +(thrj+1 )* sizeSArray + thrk+1] = 
                    aux[ blIdz*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock - dimX * dimY + thrj * dimX + thrk ];
          
          if( dimZ > (blIdz+1) * numThreadsPerBlock && thrj+1 < ykolik && thrk+1 < xkolik )
            sArray[zkolik * sizeSArray * sizeSArray + (thrj+1) * sizeSArray + thrk+1] = 
                    aux[ (blIdz+1)*numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock*dimX + blIdx*numThreadsPerBlock + thrj * dimX + thrk ];
        }
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = 0; l < numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
              sArray[(m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1] = 
                      aux[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l ];
          }
        }
      }
      /*string s;
       int numWhile = 0;
       for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
          for( IndexType l = 0; l < numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ ){
              //std::cout << "proslo i = " << k * numThreadsPerBlock + l << std::endl;
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                //printf("In with point m  = %d, k = %d, l = %d\n", m, k, l);
                changed = this->template updateCell< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz) || changed;
                
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = 0; l <numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );
       */
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell< sizeSArray >(  sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = 0; l <numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = 0; l <numThreadsPerBlock; l++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      
      for( IndexType m = numThreadsPerBlock-1; m >-1; m-- ){
        for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
          for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )
            {
              if( ! interfaceMap[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l  ] )
              {
                this->template updateCell< sizeSArray >( sArray, l+1, k+1, m+1, hx,hy,hz);
              }
            }
          }
        }
      }
      /*for( int k = 0; k < numThreadsPerBlock; k++ ){
       for( int l = 0; l < numThreadsPerBlock; l++ ) 
       for( int m = 0; m < numThreadsPerBlock; m++ )
       if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ )     
       helpFunc[ m*dimX*dimY + k*dimX + l ] = sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
       } 
       numWhile++;
       s = "helpFunc-"+ std::to_string(numWhile) + ".tnl";
       helpFunc.save( s );*/
      
      if( changed ){
        BlockIterHost[ i ] = 1;
      }
      
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ) {
          for( IndexType m = 0; m < numThreadsPerBlock; m++ ){
            if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX && blIdz * numThreadsPerBlock + m < dimZ ){      
              helpFunc[ blIdz * numThreadsPerBlock * dimX * dimY + blIdy * numThreadsPerBlock * dimX + blIdx*numThreadsPerBlock + m*dimX*dimY + k*dimX + l ] = 
                      sArray[ (m+1) * sizeSArray * sizeSArray + (k+1) *sizeSArray + l+1 ];
              //std::cout << helpFunc[ m*dimX*dimY + k*dimX + l ] << " ";
            }
          }
          //std::cout << std::endl;
        }
        //std::cout << std::endl;
      }
      //helpFunc.save( "helpF.tnl");
      delete []sArray;
    }
  }
}

template< typename Real,
        typename Device,
        typename Index >
void 
tnlDirectEikonalMethodsBase< Meshes::Grid< 3, Real, Device, Index > >::
getNeighbours( ArrayContainerView BlockIterHost, int numBlockX, int numBlockY, int numBlockZ )
{
  int* BlockIterPom; 
  BlockIterPom = new int [ numBlockX * numBlockY * numBlockZ ];
  
  for( int i = 0; i< BlockIterHost.getSize(); i++)
  {
    BlockIterPom[ i ] = 0;
    
    int m=0, l=0, k=0;
    l = i/( numBlockX * numBlockY );
    k = (i-l*numBlockX * numBlockY )/(numBlockX );
    m = (i-l*numBlockX * numBlockY )%( numBlockX );
    
    if( m > 0 && BlockIterHost[ i - 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterHost[ i + 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterHost[ i - numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterHost[ i + numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }else if( l > 0 && BlockIterHost[ i - numBlockX*numBlockY ] ){
      BlockIterPom[ i ] = 1;
    }else if( l < numBlockZ-1 && BlockIterHost[ i + numBlockX*numBlockY ] ){
      BlockIterPom[ i ] = 1;
    }
  }
  for( int i = 0; i< BlockIterHost.getSize(); i++)
  { 
    BlockIterHost[ i ] = BlockIterPom[ i ];
  }
}
