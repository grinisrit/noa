/*
 * main.cu
 *
 *  Created on: Nov 21, 2014
 *      Author: yongchao
 */
#include "Options.h"
#include "SpMV.h"

int32_t main(int32_t argc, char* argv[]) {
	Options opt;
	float runtime;
	double gflops;
	int32_t numIters;

	/*parse the parameters*/
	if (!opt.parseArgs(argc, argv)) {
		return -1;
	}
	numIters = opt._numIters;

	/*run the sparse matrix-vector multiplication kernel*/
	SpMV* spmv;
	if (opt._singlePrecision) {
		switch (opt._routine) {
		case 0:
			spmv = new SpMVFloatVector(&opt);
			break;
		case 1:
			spmv = new SpMVFloatWarp(&opt);
			break;
		default:
			cerr << "Error: unsupported routine number for FLOAT" << endl;
			return -1;
		}
	} else {
		switch (opt._routine) {
		case 0:
			spmv = new SpMVDoubleVector(&opt);
			break;
		case 1:
			spmv = new SpMVDoubleWarp(&opt);
			break;
		default:
			cerr << "Error: unsupported routine number for DOUBLE" << endl;
			return -1;
		}
	}

	/*set device cache*/
	if (opt._routine == 2) {
		cudaDeviceSetCacheConfig (cudaFuncCachePreferShared);
	} else {
		cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
	}

	if (opt._singlePrecision) {
		cerr << "Use single-precision floating point" << endl;
	} else {
		cerr << "Use double-precision floating point" << endl;
	}

	/*print out the statistical information of the sparse matrix*/
	opt.getRowSizeVariance();

	/*load the data*/
	spmv->loadData();

	/*run the kernel*/
	double stime = spmv->getSysTime();
	for (int32_t i = 0; i < numIters; ++i) {
		spmv->spmvKernel();
	}
	/*synchronize all kernels*/
	cudaDeviceSynchronize();
	double etime = spmv->getSysTime();

	runtime = etime - stime;
	runtime /= 1000.0 * (float) numIters;
	cerr << "Average runtime: " << runtime << " seconds (in " << numIters
			<< " iterations)" << endl;

	/*compute the GFLOPS*/
	gflops =
			opt._formula == 0 ?
					2 * opt._numValues - 1 :
					2 * (opt._numValues + opt._numRows);
	cerr << "Total FLOPs: " << (uint64_t) gflops << endl;
	gflops /= runtime * 1000000000;
	cerr << "GFLOPS: " << gflops << endl;

	/*store the data*/
	spmv->storeData();

	/*release the data*/
	delete spmv;

	return 0;
}
