Array a;
cudaKernel<<< gridSize, blockSize >>>( a );