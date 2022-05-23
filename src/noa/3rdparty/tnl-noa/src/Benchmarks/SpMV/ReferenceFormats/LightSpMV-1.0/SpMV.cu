/*
 * SpMV.cu
 *
 *  Created on: Nov 21, 2014
 *      Author: yongchao
 */
#include "SpMV.h"
#include "SpMVCSR.h"

extern __constant__ uint32_t _cudaNumRows;

SpMV::SpMV(Options* opt) {
	_opt = opt;

	/*the number of GPUs*/
	_numGPUs = _opt->_numGPUs;

	/*compute the mean number of elements per row*/
	_meanElementsPerRow = (int32_t) rint(
			(double) _opt->_numValues / _opt->_numRows);

	/*create row counter*/
	_cudaRowCounters.resize(_numGPUs, NULL);

	/*create streams*/
	_streams.resize(_numGPUs, 0);

	for (int32_t i = 0; i < _numGPUs; ++i) {
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		cudaStreamCreate(&_streams[i]);
		CudaCheckError();
	}
#if defined(FLOAT_USE_TEXTURE_MEMORY) || defined(DOUBLE_USE_TEXTURE_MEMORY)
	_texVectorX.resize(_numGPUs, 0);
#endif
}
SpMV::~SpMV() {
	/*destroy the streams*/
	for (int32_t i = 0; i < _numGPUs; ++i) {

		/*set device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		cudaStreamDestroy(_streams[i]);
		CudaCheckError();

#if defined(FLOAT_USE_TEXTURE_MEMORY) || defined(DOUBLE_USE_TEXTURE_MEMORY)
		if (_texVectorX[i]) {
			cudaDestroyTextureObject(_texVectorX[i]);
		}
		CudaCheckError();
#endif
	}
}

/*invoke kernel*/
void SpMV::spmvKernel() {

	/*initialize the counter*/
	cudaMemset(_cudaRowCounters[0], 0, sizeof(uint32_t));

	/*invoke kernel*/
	if (_opt->_formula == 0) {
		invokeKernel(0);
	} else {
		invokeKernelBLAS(0);
	}
}
void SpMV::invokeKernel(const int32_t i) {
	/*do nothing*/
}
void SpMV::invokeKernelBLAS(const int32_t i) {
	/*do nothing*/
}

/*single-precision floating point*/
SpMVFloatVector::SpMVFloatVector(Options* opt) :
		SpMV(opt) {

	_rowOffsets.resize(_numGPUs, NULL);
	_colIndexValues.resize(_numGPUs, NULL);
	_numericalValues.resize(_numGPUs, NULL);
	_vectorY.resize(_numGPUs, NULL);
	_vectorX.resize(_numGPUs, NULL);

	_alpha = _opt->_alpha;
	_beta = _opt->_beta;
}
SpMVFloatVector::~SpMVFloatVector() {
	/*release matrix data*/
	for (int32_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*release the resources*/
		if (_rowOffsets[i]) {
			cudaFree(_rowOffsets[i]);
		}
		if (_colIndexValues[i]) {
			cudaFree(_colIndexValues[i]);
		}

		if (_numericalValues[i]) {
			cudaFree(_numericalValues[i]);
		}
		if (i == 0 && _vectorY[i]) {
			cudaFree(_vectorY[i]);
		}
		if (_vectorX[i]) {
			cudaFree(_vectorX[i]);
		}
	}
}
void SpMVFloatVector::loadData() {
	size_t numBytes;

#ifdef FLOAT_USE_TEXTURE_MEMORY
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	/*specify the texture object parameters*/
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
#endif

	/*iterate each GPU*/
	for (int32_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*allocate counter buffers*/
		cudaMalloc(&_cudaRowCounters[i], sizeof(uint32_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumRows, &_opt->_numRows, sizeof(uint32_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumCols, &_opt->_numCols, sizeof(uint32_t));
		CudaCheckError();

		/******************************************************
		 * Load matrix data
		 ******************************************************/
		numBytes = (_opt->_numRows + 1) * sizeof(uint32_t);
		cudaMalloc(&_rowOffsets[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_rowOffsets[i], _opt->_rowOffsets, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		numBytes = _opt->_numValues * sizeof(uint32_t);
		cudaMalloc(&_colIndexValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_colIndexValues[i], _opt->_colIndexValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		/*load the numerical values*/
		numBytes = _opt->_numValues * sizeof(float);
		cudaMalloc(&_numericalValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_numericalValues[i], _opt->_numericalValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		/*****************************************************
		 * Load vector X data
		 ******************************************************/
		numBytes = _opt->_numCols * sizeof(float);
		cudaMalloc(&_vectorX[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_vectorX[i], _opt->_vectorX, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

#ifdef FLOAT_USE_TEXTURE_MEMORY
		/*specify texture and texture object*/
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = _vectorX[i];
		resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
				cudaChannelFormatKindFloat);
		resDesc.res.linear.sizeInBytes = numBytes;
		cudaCreateTextureObject(&_texVectorX[i], &resDesc, &texDesc, NULL);
		CudaCheckError();
#endif

		/*****************************************************
		 * vector Y data
		 ******************************************************/
		numBytes = _opt->_numRows * sizeof(float);
		cudaMalloc(&_vectorY[i], numBytes);
		CudaCheckError();

		/*copy the data*/
		cudaMemcpy(_vectorY[i], _opt->_vectorY, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();
	}
}
void SpMVFloatVector::storeData() {
	/*transfer back vector Y*/
	uint64_t numBytes = _opt->_numRows * sizeof(float);

	/*select the device*/
	cudaSetDevice(_opt->_gpus[0].first);
	CudaCheckError();

	/*copy back the data*/
	cudaMemcpy(_opt->_vectorY, _vectorY[0], numBytes, cudaMemcpyDeviceToHost);
	CudaCheckError();

	/*open the file*/
	FILE* file;
	if (_opt->_outFileName.length() == 0) {
		return;
	}

	file = fopen(_opt->_outFileName.c_str(), "w");
	if (!file) {
		cerr << "Failed to open file: " << _opt->_outFileName << endl;
		return;
	}

	/*write to the file*/
	float* ptr = (float*) _opt->_vectorY;
	for (uint32_t i = 0; i < _opt->_numRows; ++i) {
		fprintf(file, "%f\n", ptr[i]);
	}

	/*close the file*/
	if (file != stdout) {
		fclose(file);
	}
}
void SpMVFloatVector::invokeKernel(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef FLOAT_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicVector<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicVector<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicVector<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr32DynamicVector<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicVector<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicVector<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicVector<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr32DynamicVector<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}

#endif
}

void SpMVFloatVector::invokeKernelBLAS(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef FLOAT_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicVectorBLAS<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicVectorBLAS<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicVectorBLAS<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr32DynamicVectorBLAS<float, 32,
				MAX_NUM_THREADS_PER_BLOCK / 32><<<numThreadBlocks,
				numThreadsPerBlock>>>(_cudaRowCounters[i], _rowOffsets[i],
				_colIndexValues[i], _numericalValues[i], _texVectorX[i],
				_vectorY[i], _alpha, _beta);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicVectorBLAS<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicVectorBLAS<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicVectorBLAS<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr32DynamicVectorBLAS<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	}

#endif
}

void SpMVFloatWarp::invokeKernel(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef FLOAT_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicWarp<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicWarp<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicWarp<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr32DynamicWarp<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicWarp<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i],_vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicWarp<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicWarp<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr32DynamicWarp<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}

#endif
}

void SpMVFloatWarp::invokeKernelBLAS(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef FLOAT_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicWarpBLAS<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicWarpBLAS<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicWarpBLAS<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr32DynamicWarpBLAS<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _alpha, _beta);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr32DynamicWarpBLAS<float, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i],_vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr32DynamicWarpBLAS<float, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr32DynamicWarpBLAS<float, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr32DynamicWarpBLAS<float, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _alpha, _beta);
	}

#endif
}

/*double-precision floating point*/
SpMVDoubleVector::SpMVDoubleVector(Options* opt) :
		SpMV(opt) {

	_rowOffsets.resize(_numGPUs, NULL);
	_colIndexValues.resize(_numGPUs, NULL);
	_numericalValues.resize(_numGPUs, NULL);
	_vectorY.resize(_numGPUs, NULL);

	_vectorX.resize(_numGPUs, NULL);

	_alpha = _opt->_alpha;
	_beta = _opt->_beta;

}
SpMVDoubleVector::~SpMVDoubleVector() {
	/*release matrix data*/
	for (int32_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*release the resources*/
		if (_rowOffsets[i]) {
			cudaFree(_rowOffsets[i]);
		}
		if (_colIndexValues[i]) {
			cudaFree(_colIndexValues[i]);
		}

		if (_numericalValues[i]) {
			cudaFree(_numericalValues[i]);
		}
		if (i == 0 && _vectorY[i]) {
			cudaFree(_vectorY[i]);
		}
		if (_vectorX[i]) {
			cudaFree(_vectorX[i]);
		}
	}
}
void SpMVDoubleVector::loadData() {
	size_t numBytes;

#ifdef DOUBLE_USE_TEXTURE_MEMORY
	cudaTextureDesc texDesc;
	cudaResourceDesc resDesc;

	/*specify the texture object parameters*/
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
#endif

	/*iterate each GPU*/
	for (int32_t i = 0; i < _numGPUs; ++i) {

		/*select the device*/
		cudaSetDevice(_opt->_gpus[i].first);
		CudaCheckError();

		/*allocate counter buffers*/
		cudaMalloc(&_cudaRowCounters[i], sizeof(uint32_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumRows, &_opt->_numRows, sizeof(uint32_t));
		CudaCheckError();

		cudaMemcpyToSymbol(_cudaNumCols, &_opt->_numCols, sizeof(uint32_t));
		CudaCheckError();

		/******************************************************
		 * Load matrix data
		 ******************************************************/
		numBytes = (_opt->_numRows + 1) * sizeof(uint32_t);
		cudaMalloc(&_rowOffsets[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_rowOffsets[i], _opt->_rowOffsets, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		numBytes = _opt->_numValues * sizeof(uint32_t);
		cudaMalloc(&_colIndexValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_colIndexValues[i], _opt->_colIndexValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		/*load the numerical values*/
		numBytes = _opt->_numValues * sizeof(double);
		cudaMalloc(&_numericalValues[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_numericalValues[i], _opt->_numericalValues, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

		/*****************************************************
		 * Load vector X data
		 ******************************************************/
		numBytes = _opt->_numCols * sizeof(double);
		cudaMalloc(&_vectorX[i], numBytes);
		CudaCheckError();

		cudaMemcpy(_vectorX[i], _opt->_vectorX, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();

#ifdef DOUBLE_USE_TEXTURE_MEMORY
		/*specify texture and texture object*/
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = _vectorX[i];
		resDesc.res.linear.desc = cudaCreateChannelDesc(32, 32, 0, 0,
				cudaChannelFormatKindSigned);
		resDesc.res.linear.sizeInBytes = numBytes;
		cudaCreateTextureObject(&_texVectorX[i], &resDesc, &texDesc, NULL);
		CudaCheckError();
#endif
		/*****************************************************
		 * vector Y data
		 ******************************************************/
		numBytes = _opt->_numRows * sizeof(double);
		/*allocate space on the first GPU*/
		cudaMalloc(&_vectorY[i], numBytes);
		CudaCheckError();

		/*copy the data*/
		cudaMemcpy(_vectorY[i], _opt->_vectorY, numBytes,
				cudaMemcpyHostToDevice);
		CudaCheckError();
	}
}
void SpMVDoubleVector::storeData() {
	/*transfer back vector Y*/
	uint64_t numBytes = _opt->_numRows * sizeof(double);

	/*select the device*/
	cudaSetDevice(_opt->_gpus[0].first);
	CudaCheckError();

	/*copy back the data*/
	cudaMemcpy(_opt->_vectorY, _vectorY[0], numBytes, cudaMemcpyDeviceToHost);
	CudaCheckError();

	/*open the file*/
	FILE* file;
	if (_opt->_outFileName.length() == 0) {
		return;
	}

	file = fopen(_opt->_outFileName.c_str(), "w");
	if (!file) {
		cerr << "Failed to open file: " << _opt->_outFileName << endl;
		return;
	}

	/*write to the file*/
	double* ptr = (double*) _opt->_vectorY;
	for (uint32_t i = 0; i < _opt->_numRows; ++i) {
		fprintf(file, "%lf\n", ptr[i]);
	}

	/*close the file*/
	if (file != stdout) {
		fclose(file);
	}
}
void SpMVDoubleVector::invokeKernel(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef DOUBLE_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicVector<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicVector<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicVector<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr64DynamicVector<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicVector<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicVector<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicVector<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr64DynamicVector<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}

#endif
}

void SpMVDoubleVector::invokeKernelBLAS(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef DOUBLE_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicVectorBLAS<double, 2,
				MAX_NUM_THREADS_PER_BLOCK / 2><<<numThreadBlocks,
				numThreadsPerBlock>>>(_cudaRowCounters[i], _rowOffsets[i],
				_colIndexValues[i], _numericalValues[i], _texVectorX[i],
				_vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicVectorBLAS<double, 4,
				MAX_NUM_THREADS_PER_BLOCK / 4><<<numThreadBlocks,
				numThreadsPerBlock>>>(_cudaRowCounters[i], _rowOffsets[i],
				_colIndexValues[i], _numericalValues[i], _texVectorX[i],
				_vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicVectorBLAS<double, 8,
				MAX_NUM_THREADS_PER_BLOCK / 8><<<numThreadBlocks,
				numThreadsPerBlock>>>(_cudaRowCounters[i], _rowOffsets[i],
				_colIndexValues[i], _numericalValues[i], _texVectorX[i],
				_vectorY[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr64DynamicVectorBLAS<double, 32,
				MAX_NUM_THREADS_PER_BLOCK / 32><<<numThreadBlocks,
				numThreadsPerBlock>>>(_cudaRowCounters[i], _rowOffsets[i],
				_colIndexValues[i], _numericalValues[i], _texVectorX[i],
				_vectorY[i], _vectorY[i], _alpha, _beta);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicVectorBLAS<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicVectorBLAS<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicVectorBLAS<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr64DynamicVectorBLAS<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	}
#endif
}

void SpMVDoubleWarp::invokeKernel(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef DOUBLE_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicWarp<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicWarp<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicWarp<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr64DynamicWarp<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i]);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicWarp<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicWarp<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicWarp<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	} else {
		spmv_csr::csr64DynamicWarp<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i]);
	}

#endif
}

void SpMVDoubleWarp::invokeKernelBLAS(const int32_t i) {
	int32_t numThreadsPerBlock;
	int32_t numThreadBlocks;

	/*get the number of threads per block*/
	getKernelGridInfo(i, numThreadsPerBlock, numThreadBlocks);

	/*invoke the kernel*/
#ifdef DOUBLE_USE_TEXTURE_MEMORY
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicWarpBLAS<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicWarpBLAS<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicWarpBLAS<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
				numThreadBlocks, numThreadsPerBlock>>>(_cudaRowCounters[i],
				_rowOffsets[i], _colIndexValues[i], _numericalValues[i],
				_texVectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr64DynamicWarpBLAS<double, 32,
				MAX_NUM_THREADS_PER_BLOCK / 32><<<numThreadBlocks,
				numThreadsPerBlock>>>(_cudaRowCounters[i], _rowOffsets[i],
				_colIndexValues[i], _numericalValues[i], _texVectorX[i],
				_vectorY[i], _vectorY[i], _alpha, _beta);
	}
#else
	if (_meanElementsPerRow <= 2) {
		spmv_csr::csr64DynamicWarpBLAS<double, 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if (_meanElementsPerRow <= 4) {
		spmv_csr::csr64DynamicWarpBLAS<double, 4, MAX_NUM_THREADS_PER_BLOCK / 4><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else if(_meanElementsPerRow <= 64) {
		spmv_csr::csr64DynamicWarpBLAS<double, 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	} else {
		spmv_csr::csr64DynamicWarpBLAS<double, 32, MAX_NUM_THREADS_PER_BLOCK / 32><<<
		numThreadBlocks, numThreadsPerBlock>>>(
				_cudaRowCounters[i], _rowOffsets[i], _colIndexValues[i],
				_numericalValues[i], _vectorX[i], _vectorY[i], _vectorY[i], _alpha, _beta);
	}

#endif
}
