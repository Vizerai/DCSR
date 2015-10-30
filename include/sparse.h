#ifndef SPARSE_H
#define SPARSE_H

#include "../thrust/sort.h"
#include "../thrust/reduce.h"
#include "../thrust/scan.h"
#include "../thrust/inner_product.h"
//#include <thrust-1.8/system/cuda/experimental/pinned_allocator.h>

//cusp and thrust
typedef cusp::array1d<char, cusp::device_memory> 	CuspVectorChar_d;
typedef cusp::array1d<char, cusp::host_memory> 		CuspVectorChar_h;
typedef cusp::array1d<short, cusp::device_memory> 	CuspVectorShort_d;
typedef cusp::array1d<short, cusp::host_memory> 	CuspVectorShort_h;
typedef cusp::array1d<int, cusp::device_memory> 	CuspVectorInt_d;
typedef cusp::array1d<int, cusp::host_memory> 		CuspVectorInt_h;
typedef cusp::array1d<float, cusp::device_memory> 	CuspVectorS_d;
typedef cusp::array1d<float, cusp::host_memory> 	CuspVectorS_h;
typedef cusp::array1d<double, cusp::device_memory> 	CuspVectorD_d;
typedef cusp::array1d<double, cusp::host_memory> 	CuspVectorD_h;

#define TPC(x)		thrust::raw_pointer_cast(x)

inline void safeSync()
{
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		fprintf( stderr, "!! GPU program execution error on line %d : cudaError=%d, (%s)\n", __LINE__, error, cudaGetErrorString(error));
}

#endif