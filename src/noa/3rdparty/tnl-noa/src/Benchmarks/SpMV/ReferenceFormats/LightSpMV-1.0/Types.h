/*
 * Types.h
 *
 *  Created on: Nov 21, 2014
 *      Author: yongchao
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
using namespace std;

/*program version*/
#define VERSION "v1.0"

/*macros for cuda array*/
#if !defined(SPMV_CUDA_ARRAY_WIDTH_SHIFT) || SPMV_CUDA_ARRAY_WIDTH_SHIFT < 10 || SPMV_CUDA_ARRAY_WIDTH_SHIFT > 16
#define SPMV_CUDA_ARRAY_WIDTH_SHIFT		15
#endif
#define SPMV_CUDA_ARRAY_WIDTH_MASK		((1 << SPMV_CUDA_ARRAY_WIDTH_SHIFT) - 1)
#define SPMV_CUDA_ARRAY_WIDTH 			(1 << SPMV_CUDA_ARRAY_WIDTH_SHIFT)

/*texture memory*/
#ifdef NO_FLOAT_TEXTURE_MEMORY
#undef FLOAT_USE_TEXTURE_MEMORY
#else
#define FLOAT_USE_TEXTURE_MEMORY
#endif

#ifdef NO_DOUBLE_TEXTURE_MEMORY
#undef DOUBLE_USE_TEXTURE_MEMORY
#else
#define DOUBLE_USE_TEXTURE_MEMORY
#endif

/*maximum number of threads per block*/
#define MAX_NUM_THREADS_PER_BLOCK			1024

/*error check*/
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError(const char* file, const int32_t line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		cerr << "cudaCheckError() failed at " << file << ":" << line << " : "
				<< cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

#endif /* TYPES_H_ */
