#pragma once

template< typename Real,
        typename Device,
        typename Index >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
initInterface( const MeshFunctionPointer& _input,
        MeshFunctionPointer& _output,
        InterfaceMapPointer& _interfaceMap, 
        const StaticVector vecLowerOverlaps, 
        const StaticVector vecUpperOverlaps )
{
  
  if( std::is_same< Device, Devices::Cuda >::value )
  {
#ifdef HAVE_CUDA
    const MeshType& mesh = _input->getMesh();
    
    const int cudaBlockSize( 16 );
    int numBlocksX = Cuda::getNumberOfBlocks( mesh.getDimensions().x(), cudaBlockSize );
    int numBlocksY = Cuda::getNumberOfBlocks( mesh.getDimensions().y(), cudaBlockSize );
    dim3 blockSize( cudaBlockSize, cudaBlockSize );
    dim3 gridSize( numBlocksX, numBlocksY );
    Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
    CudaInitCaller<<< gridSize, blockSize >>>( _input.template getData< Device >(),
            _output.template modifyData< Device >(),
            _interfaceMap.template modifyData< Device >(),
            vecLowerOverlaps, vecUpperOverlaps);
    cudaDeviceSynchronize();
    TNL_CHECK_CUDA_DEVICE;
#endif
  }
  if( std::is_same< Device, Devices::Host >::value )
  {
    MeshFunctionType input = _input.getData();    
    MeshFunctionType& output = _output.modifyData();
    InterfaceMapType& interfaceMap = _interfaceMap.modifyData();
    const MeshType& mesh = input.getMesh();
    typedef typename MeshType::Cell Cell;
    Cell cell( mesh );
    for( cell.getCoordinates().y() = 0;
            cell.getCoordinates().y() < mesh.getDimensions().y();
            cell.getCoordinates().y() ++ )
      for( cell.getCoordinates().x() = 0;
              cell.getCoordinates().x() < mesh.getDimensions().x();
              cell.getCoordinates().x() ++ )
      {
        cell.refresh();
        output[ cell.getIndex() ] =
                input( cell ) >= 0 ? std::numeric_limits< RealType >::max() :
                  - std::numeric_limits< RealType >::max();
        interfaceMap[ cell.getIndex() ] = false;
      }
    
    const RealType& hx = mesh.getSpaceSteps().x();
    const RealType& hy = mesh.getSpaceSteps().y();     
    for( cell.getCoordinates().y() = 0 + vecLowerOverlaps[1];
            cell.getCoordinates().y() < mesh.getDimensions().y() - vecUpperOverlaps[1];
            cell.getCoordinates().y() ++ )
      for( cell.getCoordinates().x() = 0 + vecLowerOverlaps[0];
              cell.getCoordinates().x() < mesh.getDimensions().x() - vecUpperOverlaps[0];
              cell.getCoordinates().x() ++ )
      {
        cell.refresh();
        const RealType& c = input( cell );
        if( ! cell.isBoundaryEntity()  )
        {
          auto neighbors = cell.getNeighborEntities();
          Real pom = 0;
          const IndexType e = neighbors.template getEntityIndex<  1,  0 >();
          const IndexType n = neighbors.template getEntityIndex<  0,  1 >();
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
            
            pom = pom - TNL::sign( c )*hx; //output[ e ] = (hx * c)/( c - input[ e ]) - hx;
            if( TNL::abs( output[ e ] ) > TNL::abs( pom ) )
              output[ e ] = pom; 
            
            interfaceMap[ cell.getIndex() ] = true;
            interfaceMap[ e ] = true;
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
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( MeshFunctionType& u,
        const MeshEntity& cell,   
        const RealType v)
{
  const auto& neighborEntities = cell.template getNeighborEntities< 2 >();
  const MeshType& mesh = cell.getMesh();
  const RealType& hx = mesh.getSpaceSteps().x();
  const RealType& hy = mesh.getSpaceSteps().y();
  const RealType value = u( cell );
  RealType a, b, tmp = std::numeric_limits< RealType >::max();
  
  if( cell.getCoordinates().x() == 0 )
    a = u[ neighborEntities.template getEntityIndex< 1,  0 >() ];
  else if( cell.getCoordinates().x() == mesh.getDimensions().x() - 1 )
    a = u[ neighborEntities.template getEntityIndex< -1,  0 >() ];
  else
  {
    a = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< -1,  0 >() ],
            u[ neighborEntities.template getEntityIndex<  1,  0 >() ] );
  }
  
  if( cell.getCoordinates().y() == 0 )
    b = u[ neighborEntities.template getEntityIndex< 0,  1 >()];
  else if( cell.getCoordinates().y() == mesh.getDimensions().y() - 1 )
    b = u[ neighborEntities.template getEntityIndex< 0,  -1 >() ];
  else
  {
    b = TNL::argAbsMin( u[ neighborEntities.template getEntityIndex< 0,  -1 >() ],
            u[ neighborEntities.template getEntityIndex< 0,   1 >() ] );
  }
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() && 
          fabs( b ) == std::numeric_limits< RealType >::max() )
    return false;
  
  RealType pom[6] = { a, b, std::numeric_limits< RealType >::max(), hx, hy, 0.0 };
  
  tmp = getNewValue( pom , value, v );
  
  u[ cell.getIndex() ] = tmp;
  
  
  tmp = value - u[ cell.getIndex() ];
  
  if ( fabs( tmp ) >  0.001*hx )
    return true;
  else
    return false;
  
}

template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
__cuda_callable__
bool
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateCell( volatile Real *sArray, int thri, int thrj, const Real hx, const Real hy,
        const Real v )
{
  const RealType value = sArray[ thrj * sizeSArray + thri ];
  RealType a, b, tmp = std::numeric_limits< RealType >::max();
  
  b = TNL::argAbsMin( sArray[ (thrj+1) * sizeSArray + thri ],
          sArray[ (thrj-1) * sizeSArray + thri ] );
  
  a = TNL::argAbsMin( sArray[ thrj * sizeSArray + thri+1 ],
          sArray[ thrj * sizeSArray + thri-1 ] );
  
  if( fabs( a ) == std::numeric_limits< RealType >::max() && 
          fabs( b ) == std::numeric_limits< RealType >::max() )
    return false;
  
  RealType pom[6] = { a, b, std::numeric_limits< RealType >::max(), (RealType)hx, (RealType)hy, 0.0 };
  
  tmp = getNewValue( pom , value, v );
  
  sArray[ thrj * sizeSArray + thri ] = tmp;
  tmp = value - sArray[ thrj * sizeSArray + thri ];
  if ( fabs( tmp ) >  0.001*hx )
    return true;
  else
    return false;
}

template< typename Real,
        typename Device,
        typename Index >
__cuda_callable__
Real
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
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
    newValue = argAbsMin( originalValue, newValue );
  }
  
  return newValue;
}


template < typename T1 >
__cuda_callable__ void sortMinims( T1 pom[] )
{
  T1 tmp[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; 
  if( fabs(pom[0]) <= fabs(pom[1]) && fabs(pom[1]) <= fabs(pom[2])){
    tmp[0] = pom[0]; tmp[1] = pom[1]; tmp[2] = pom[2];
    tmp[3] = pom[3]; tmp[4] = pom[4]; tmp[5] = pom[5];
    
  }
  else if( fabs(pom[0]) <= fabs(pom[2]) && fabs(pom[2]) <= fabs(pom[1]) ){
    tmp[0] = pom[0]; tmp[1] = pom[2]; tmp[2] = pom[1];
    tmp[3] = pom[3]; tmp[4] = pom[5]; tmp[5] = pom[4];
  }
  else if( fabs(pom[1]) <= fabs(pom[0]) && fabs(pom[0]) <= fabs(pom[2]) ){
    tmp[0] = pom[1]; tmp[1] = pom[0]; tmp[2] = pom[2];
    tmp[3] = pom[4]; tmp[4] = pom[3]; tmp[5] = pom[5];
  }
  else if( fabs(pom[1]) <= fabs(pom[2]) && fabs(pom[2]) <= fabs(pom[0]) ){
    tmp[0] = pom[1]; tmp[1] = pom[2]; tmp[2] = pom[0];
    tmp[3] = pom[4]; tmp[4] = pom[5]; tmp[5] = pom[3];
  }
  else if( fabs(pom[2]) <= fabs(pom[0]) && fabs(pom[0]) <= fabs(pom[1]) ){
    tmp[0] = pom[2]; tmp[1] = pom[0]; tmp[2] = pom[1];
    tmp[3] = pom[5]; tmp[4] = pom[3]; tmp[5] = pom[4];
  }
  else if( fabs(pom[2]) <= fabs(pom[1]) && fabs(pom[1]) <= fabs(pom[0]) ){
    tmp[0] = pom[2]; tmp[1] = pom[1]; tmp[2] = pom[0];
    tmp[3] = pom[5]; tmp[4] = pom[4]; tmp[5] = pom[3];
  }
  
  for( unsigned int i = 0; i < 6; i++ )
  {
    pom[ i ] = tmp[ i ];
  }   
}

#ifdef HAVE_CUDA
template < typename Real, typename Device, typename Index >
__global__ void CudaInitCaller( const Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& input, 
        Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& output,
        Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
        const Containers::StaticVector< 2, Index > vecLowerOverlaps, 
        const Containers::StaticVector< 2, Index > vecUpperOverlaps ) 
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  const Meshes::Grid< 2, Real, Device, Index >& mesh = input.template getMesh< Devices::Cuda >();
  
  if( i < mesh.getDimensions().x() && j < mesh.getDimensions().y() )
  {
    typedef typename Meshes::Grid< 2, Real, Device, Index >::Cell Cell;
    Cell cell( mesh );
    cell.getCoordinates().x() = i; cell.getCoordinates().y() = j;
    cell.refresh();
    const Index cind = cell.getIndex();
    
    
    output[ cind ] =
            input( cell ) >= 0 ? std::numeric_limits< Real >::max() :
              - std::numeric_limits< Real >::max();
    interfaceMap[ cind ] = false; 
    
    if( i < mesh.getDimensions().x() - vecUpperOverlaps[ 0 ] &&
            j < mesh.getDimensions().y() - vecUpperOverlaps[ 1 ] &&
            i>vecLowerOverlaps[ 0 ] -1 && j> vecLowerOverlaps[ 1 ]-1 )
    {
      const Real& hx = mesh.getSpaceSteps().x();
      const Real& hy = mesh.getSpaceSteps().y();
      cell.refresh();
      const Real& c = input( cell );
      if( ! cell.isBoundaryEntity()  )
      {
        auto neighbors = cell.getNeighborEntities();
        Real tmp = 0;
        const Index e = neighbors.template getEntityIndex<  1,  0 >();
        const Index w = neighbors.template getEntityIndex<  -1,  0 >();
        const Index n = neighbors.template getEntityIndex<  0,  1 >();
        const Index s = neighbors.template getEntityIndex<  0,  -1 >();
        
        if( c * input[ n ] <= 0 )
        {
          tmp = TNL::sign( c )*( hy * c )/( c - input[ n ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( tmp ) )
              output[ cind ] = tmp;
          
          interfaceMap[ cell.getIndex() ] = true;
        }
        
        
        if( c * input[ e ] <= 0 )
        {
          tmp = TNL::sign( c )*( hx * c )/( c - input[ e ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( tmp ) )
            output[ cind ] = tmp;                       
          
          interfaceMap[ cind ] = true;
        }
        if( c * input[ w ] <= 0 )
        {
          tmp = TNL::sign( c )*( hx * c )/( c - input[ w ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( tmp ) ) 
            output[ cind ] = tmp;
          
          interfaceMap[ cind ] = true;
        }
        if( c * input[ s ] <= 0 )
        {
          tmp = TNL::sign( c )*( hy * c )/( c - input[ s ]);
          if( TNL::abs( output[ cind ] ) > TNL::abs( tmp ) ) 
            output[ cind ] = tmp;
          
          interfaceMap[ cind ] = true;
        }
      }
    }
  }
}


template < typename Index >
__global__ void GetNeighbours( const TNL::Containers::ArrayView< int, Devices::Cuda, Index > blockCalculationIndicator,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > blockCalculationIndicatorHelp, int numBlockX, int numBlockY )
{
  int i = blockIdx.x * 1024 + threadIdx.x;
  
  if( i < numBlockX * numBlockY )
  {
    int pom = 0;//BlockIterPom[ i ] = 0;
    int m=0, k=0;
    m = i%numBlockX;
    k = i/numBlockX;
    if( m > 0 && blockCalculationIndicator[ i - 1 ] ){
      pom = 1;//blockCalculationIndicatorHelp[ i ] = 1;
    }else if( m < numBlockX -1 && blockCalculationIndicator[ i + 1 ] ){
      pom = 1;//blockCalculationIndicatorHelp[ i ] = 1;
    }else if( k > 0 && blockCalculationIndicator[ i - numBlockX ] ){
      pom = 1;// blockCalculationIndicatorHelp[ i ] = 1;
    }else if( k < numBlockY -1 && blockCalculationIndicator[ i + numBlockX ] ){
      pom = 1;//blockCalculationIndicatorHelp[ i ] = 1;
    }
    
    if( blockCalculationIndicator[ i ] != 1 )
      blockCalculationIndicatorHelp[ i ] = pom;//BlockIterPom[ i ];
    else
      blockCalculationIndicatorHelp[ i ] = 1;
  }
}




template < int sizeSArray, typename Real, typename Device, typename Index >
__global__ void CudaUpdateCellCaller( tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > > ptr,
        const Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index >, 2, bool >& interfaceMap,
        const Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& aux,
        Functions::MeshFunctionView< Meshes::Grid< 2, Real, Device, Index > >& helpFunc,
        TNL::Containers::ArrayView< int, Devices::Cuda, Index > blockCalculationIndicator,
        const Containers::StaticVector< 2, Index > vecLowerOverlaps, 
        const Containers::StaticVector< 2, Index > vecUpperOverlaps, int oddEvenBlock )
{
  // Setting up threads
  int thri = threadIdx.x; int thrj = threadIdx.y;
  int i = threadIdx.x + blockDim.x*blockIdx.x + vecLowerOverlaps[0];
  int j = blockDim.y*blockIdx.y + threadIdx.y + vecLowerOverlaps[1];
  const Meshes::Grid< 2, Real, Device, Index >& mesh = aux.template getMesh< Devices::Cuda >();
  
/** FOR CHESS METHOD */
  //if( (blockIdx.y%2  + blockIdx.x) % 2 == oddEvenBlock )
  //{
/**------------------------------------------*/
  
/** FOR FIM METHOD */
  if( blockCalculationIndicator[ blockIdx.y * gridDim.x + blockIdx.x ] )
  { 
    __syncthreads();
/**-----------------------------------------*/
    
    const int dimX = mesh.getDimensions().x(); const int dimY = mesh.getDimensions().y();
    const Real hx = mesh.getSpaceSteps().x(); const Real hy = mesh.getSpaceSteps().y();
    if( thri==0 && thrj == 0)
    {
      blockCalculationIndicator[ blockIdx.y * gridDim.x + blockIdx.x ] = 0;
    }
    __syncthreads();
    int maxThreadsInXDirection;
    int maxThreadsInYDirection;
    
    // Maximum threads in each direction can differ
    // e.g. cudaBlockSize = 16, dimX = 50, then:
    // blockIdx   maxThreadsInXDirection   calculation [from, to]  sArray [from, to] 
    //    0                 16                    [ 0,15]             [ 0,16]   //"-1" set to inf
    //    1                 16                    [16,31]             [15,32]
    //    2                 16                    [32,47]             [31,48]
    //    3                  2                    [48,50]             [47,50]   // rest set to inf
    // same for YDirection because blocks are squared 
    maxThreadsInXDirection = blockDim.x + 1;
    maxThreadsInYDirection = blockDim.y + 1;
    
    if( gridDim.x - 1 == blockIdx.x ) // care about number of values if we are in last block
      maxThreadsInXDirection = (dimX-vecUpperOverlaps[0]-vecLowerOverlaps[0]) - (blockIdx.x)*blockDim.x+1;
    
    if( gridDim.y - 1 == blockIdx.y ) // care about number of values if we are in last block
      maxThreadsInYDirection = (dimY-vecUpperOverlaps[1]-vecLowerOverlaps[1]) - (blockIdx.y)*blockDim.y+1;
    __syncthreads();
    
    // Setting changed array that contains info: "Did the value of this thread changed in last passage?"
    // Will be used in parallel reduction ( inside block level )
    int currentIndex = thrj * blockDim.x + thri;
    __shared__ volatile bool changed[ ( sizeSArray - 2 ) * ( sizeSArray - 2 ) ];
    changed[ currentIndex ] = false;
    if( thrj == 0 && thri == 0 )
      changed[ 0 ] = true; // fist must be true to start while cycle
    
    
    //__shared__ volatile Real sArray[ blockDim.y+2 ][ blockDim.x+2 ];
    __shared__ volatile Real sArray[ sizeSArray * sizeSArray ];
    sArray[ (thrj+1) * sizeSArray + thri +1 ] = std::numeric_limits< Real >::max();
    
       
    //filling sArray edges
    if( thri == 0 ) // 
    {      
      if( dimX - vecLowerOverlaps[ 0 ] > (blockIdx.x+1) * blockDim.x  && thrj+1 < maxThreadsInYDirection )
        sArray[ (thrj+1)*sizeSArray + maxThreadsInXDirection ] = 
                aux[ (blockIdx.y*blockDim.y+vecLowerOverlaps[1])*dimX - dimX + blockIdx.x*blockDim.x - 1 // this to get to right possition
                + (thrj+1)*dimX + maxThreadsInXDirection + vecLowerOverlaps[0] ];                        // rest to get the right sArray overlap
      else
        sArray[ (thrj+1)*sizeSArray + maxThreadsInXDirection ] = std::numeric_limits< Real >::max();
    }
        
    if( thri == 1 )
    { 
      if( ( blockIdx.x != 0 || vecLowerOverlaps[0] != 0 ) && thrj+1 < maxThreadsInYDirection )
        sArray[(thrj+1)*sizeSArray + 0] = 
                aux[ (blockIdx.y*blockDim.y+vecLowerOverlaps[1])*dimX - dimX + blockIdx.x*blockDim.x - 1
                + (thrj+1)*dimX  + vecLowerOverlaps[0] ];
      else
        sArray[(thrj+1)*sizeSArray + 0] = std::numeric_limits< Real >::max();
    }
    
    if( thri == 2 )
    {
      if( dimY - vecLowerOverlaps[ 1 ] > (blockIdx.y+1) * blockDim.y  && thrj+1 < maxThreadsInXDirection )
        sArray[ maxThreadsInYDirection * sizeSArray + thrj+1 ] = 
                aux[ ( blockIdx.y * blockDim.y + vecLowerOverlaps[ 1 ] ) * dimX - dimX + blockIdx.x * blockDim.x - 1
                + maxThreadsInYDirection * dimX + thrj + 1 + vecLowerOverlaps[0] ];
      else
        sArray[ maxThreadsInYDirection*sizeSArray + thrj+1 ] = std::numeric_limits< Real >::max();
      
    }
        
    if( thri == 3 )
    {
      if( ( blockIdx.y != 0 || vecLowerOverlaps[1] != 0 ) && thrj+1 < maxThreadsInXDirection )
        sArray[0*sizeSArray + thrj+1] = 
                aux[ ( blockIdx.y * blockDim.y + vecLowerOverlaps[ 1 ] ) * dimX - dimX + blockIdx.x * blockDim.x - 1
                + thrj + 1 + vecLowerOverlaps[ 0 ] ];
      else
        sArray[0*sizeSArray + thrj+1] = std::numeric_limits< Real >::max();
    }
    
    // Filling sArray inside
    if( i - vecLowerOverlaps[ 0 ] < dimX && j - vecLowerOverlaps[ 1 ] < dimY &&
            thri + 1 < maxThreadsInXDirection + vecUpperOverlaps[ 0 ] && 
            thrj + 1 < maxThreadsInYDirection + vecUpperOverlaps[ 1 ] )
    {
      sArray[ ( thrj + 1 ) * sizeSArray + thri + 1 ] = aux[ j * dimX + i ];
    }
    __syncthreads();  

    //main while cycle ( CALCULATES TILL VALUES ARE CHANGING )
    while( changed[ 0 ] )
    {
      __syncthreads();
      
      changed[ currentIndex] = false;
      
      //calculation of update cell
      if( i < dimX - vecUpperOverlaps[ 0 ] && j < dimY - vecUpperOverlaps[ 1 ] )
      {
        if( ! interfaceMap[ j * dimX + i ] )
        {
          changed[ currentIndex ] = ptr.updateCell<sizeSArray>( sArray, thri + 1, thrj + 1, hx, hy );
        }
      }
      __syncthreads();
      
      //pyramid reduction
      if( blockDim.x * blockDim.y == 1024 )
      {
        if( currentIndex < 512 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 512 ];
        }
      }
      __syncthreads();
      if( blockDim.x * blockDim.y >= 512 )
      {
        if( currentIndex < 256 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 256 ];
        }
      }
      __syncthreads();
      if( blockDim.x * blockDim.y >= 256 )
      {
        if( currentIndex < 128 )
        {
          changed[ currentIndex ] = changed[ currentIndex ] || changed[ currentIndex + 128 ];
        }
      }
      __syncthreads();
      if( blockDim.x * blockDim.y >= 128 )
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
      // result of reduction is in changed[ 0 ]
      
      // If we calculated in passage, then the blockCalculationIndicator for this block has to be 1
      // means that we calculated in this block
      if( thri == 0 && thrj == 0 && changed[ 0 ] ){
        blockCalculationIndicator[ blockIdx.y * gridDim.x + blockIdx.x ] = 1;
      }
      __syncthreads();
    }
    
    
      
    if( i < dimX && j < dimY && thri+1 < maxThreadsInXDirection && thrj+1 < maxThreadsInYDirection )
      helpFunc[ j * dimX + i ] = sArray[ ( thrj + 1 ) * sizeSArray + thri + 1 ];
    __syncthreads();
  }
  else
  {
    if( i < mesh.getDimensions().x() - vecUpperOverlaps[0] && j < mesh.getDimensions().y() - vecUpperOverlaps[1] )
      helpFunc[ j * mesh.getDimensions().x() + i ] = aux[ j * mesh.getDimensions().x() + i ];
  }
}
#endif



/// ====================OPEN=MP============================================
template< typename Real,
        typename Device,
        typename Index >
template< int sizeSArray >
void
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
updateBlocks( InterfaceMapType interfaceMap,
        MeshFunctionType aux,
        MeshFunctionType helpFunc,
        ArrayContainerView BlockIterHost, int numThreadsPerBlock/*, Real **sArray*/ )
{
#pragma omp parallel for schedule( dynamic )
  for( IndexType i = 0; i < BlockIterHost.getSize(); i++ )
  {
    if( BlockIterHost[ i ] )
    {
      MeshType mesh = interfaceMap.template getMesh< Devices::Host >();
      
      int dimX = mesh.getDimensions().x(); int dimY = mesh.getDimensions().y();
      //std::cout << "dimX = " << dimX << " ,dimY = " << dimY << std::endl;
      int numOfBlocky = dimY/numThreadsPerBlock + ((dimY%numThreadsPerBlock != 0) ? 1:0);
      int numOfBlockx = dimX/numThreadsPerBlock + ((dimX%numThreadsPerBlock != 0) ? 1:0);
      //std::cout << "numOfBlockx = " << numOfBlockx << " ,numOfBlocky = " << numOfBlocky << std::endl;
      int xkolik = numThreadsPerBlock + 1;
      int ykolik = numThreadsPerBlock + 1;
      
      int blIdx = i%numOfBlockx;
      int blIdy = i/numOfBlockx;
      //std::cout << "blIdx = " << blIdx << " ,blIdy = " << blIdy << std::endl;
      
      if( numOfBlockx - 1 == blIdx )
        xkolik = dimX - (blIdx)*numThreadsPerBlock+1;
      
      if( numOfBlocky -1 == blIdy )
        ykolik = dimY - (blIdy)*numThreadsPerBlock+1;
      //std::cout << "xkolik = " << xkolik << " ,ykolik = " << ykolik << std::endl;
      
      
      /*bool changed[numThreadsPerBlock*numThreadsPerBlock];
       changed[ 0 ] = 1;*/
      Real hx = mesh.getSpaceSteps().x();
      Real hy = mesh.getSpaceSteps().y();
      
      bool changed = false;
      BlockIterHost[ blIdy * numOfBlockx + blIdx ] = 0;
      
      
      Real *sArray;
      sArray = new Real[ sizeSArray * sizeSArray ];
      if( sArray == nullptr )
        std::cout << "Error while allocating memory for sArray." << std::endl;
      
      for( IndexType thri = 0; thri < sizeSArray; thri++ ){
        for( IndexType thrj = 0; thrj < sizeSArray; thrj++ )
          sArray[ thri * sizeSArray + thrj ] = std::numeric_limits< Real >::max();
      }
      
      
      //printf("numThreadsPerBlock = %d\n", numThreadsPerBlock);
      for( IndexType thrj = 0; thrj < numThreadsPerBlock + 1; thrj++ )
      {        
        if( dimX > (blIdx+1) * numThreadsPerBlock  && thrj+1 < ykolik )
          sArray[ ( thrj+1 )* sizeSArray +xkolik] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + (thrj+1)*dimX + xkolik ];
        
        
        if( blIdx != 0 && thrj+1 < ykolik )
          sArray[(thrj+1)* sizeSArray] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + (thrj+1)*dimX ];
        
        if( dimY > (blIdy+1) * numThreadsPerBlock  && thrj+1 < xkolik )
          sArray[ykolik * sizeSArray + thrj+1] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + ykolik*dimX + thrj+1 ];
        
        if( blIdy != 0 && thrj+1 < xkolik )
          sArray[thrj+1] = aux[ blIdy*numThreadsPerBlock*dimX - dimX + blIdx*numThreadsPerBlock - 1 + thrj+1 ];
      }
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){
        for( IndexType l = 0; l < numThreadsPerBlock; l++ )
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
            sArray[(k+1) * sizeSArray + l+1] = aux[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ];
      }
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ){
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX ){
            //std::cout << "proslo i = " << k * numThreadsPerBlock + l << std::endl;
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              changed = this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx,hy) || changed;
              
            }
          }
        }
      }
      /*aux.save( "aux-1pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ) 
        for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ) { 
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
          {
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx,hy);
            }
          }
        }
      /*aux.save( "aux-2pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ) 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ) {
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
          {
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx,hy);
            }
          }
        }
      /*aux.save( "aux-3pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      for( IndexType k = numThreadsPerBlock-1; k > -1; k-- ){
        for( IndexType l = numThreadsPerBlock-1; l >-1; l-- ) { 
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )
          {
            if( ! interfaceMap[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] )
            {
              this->template updateCell< sizeSArray >( sArray, l+1, k+1, hx, hy, 1.0);
            }
          }
        }
      }
      /*aux.save( "aux-4pruch.tnl" );
       for( int k = 0; k < sizeSArray; k++ ){ 
       for( int l = 0; l < sizeSArray; l++ ) {
       std::cout << sArray[ k * sizeSArray + l] << " ";
       }
       std::cout << std::endl;
       }*/
      
      
      if( changed ){
        BlockIterHost[ blIdy * numOfBlockx + blIdx ] = 1;
      }
      
      
      for( IndexType k = 0; k < numThreadsPerBlock; k++ ){ 
        for( IndexType l = 0; l < numThreadsPerBlock; l++ ) {
          if( blIdy * numThreadsPerBlock + k < dimY && blIdx * numThreadsPerBlock + l < dimX )      
            helpFunc[ blIdy * numThreadsPerBlock * dimX + numThreadsPerBlock * blIdx  + k*dimX + l ] = sArray[ (k + 1)* sizeSArray + l + 1 ];
          //std::cout<< sArray[k+1][l+1];
        }
        //std::cout<<std::endl;
      }
      delete []sArray;
    }
  }
}

template< typename Real,
        typename Device,
        typename Index >
void 
tnlDirectEikonalMethodsBase< Meshes::Grid< 2, Real, Device, Index > >::
getNeighbours( ArrayContainerView BlockIterHost, int numBlockX, int numBlockY )
{
  int* BlockIterPom; 
  BlockIterPom = new int [numBlockX * numBlockY];
  
  for(int i = 0; i < numBlockX * numBlockY; i++)
  {
    BlockIterPom[ i ] = 0;//BlockIterPom[ i ] = 0;
    int m=0, k=0;
    m = i%numBlockX;
    k = i/numBlockX;
    if( m > 0 && BlockIterHost[ i - 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( m < numBlockX -1 && BlockIterHost[ i + 1 ] ){
      BlockIterPom[ i ] = 1;
    }else if( k > 0 && BlockIterHost[ i - numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }else if( k < numBlockY -1 && BlockIterHost[ i + numBlockX ] ){
      BlockIterPom[ i ] = 1;
    }
  }
  
  for(int i = 0; i < numBlockX * numBlockY; i++)
  {
    if( !BlockIterHost[ i ] )
      BlockIterHost[ i ] = BlockIterPom[ i ];
  }
  delete[] BlockIterPom;
}



