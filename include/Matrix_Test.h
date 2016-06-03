#ifndef MATRIX_TEST_H
#define MATRIX_TEST_H

#define ARG_MAX                     512
#define BLOCKS                      26
#define BLOCK_THREAD_SIZE           256
#define BLOCK_THREADS_MAX		    512
#define DEFAULT_OVERFLOW            256
#define MAX_OFFSET                  4
#define __VECTOR_SIZE               4
#define RUN_HYB                     0
#define RUN_CSR                     1
#define RUN_DCSR                    1
#define SPMV_TEST                   0
#define SPMM_TEST                   1
#define USE_PARTIAL_VECTORS         0
#define RUN_ADD                     0
#define NUM_STREAMS                 8
#define PRECISION                   64

#define MEMORY_ALIGNMENT    4096
#define ALIGN_UP(x,size)    ( ((size_t)x+(size-1))&(~(size-1)) ) //works for size that is a power of 2
#define ROUND_UP(x,y)       ( (x + y-1) / y )
#define SAFE_DELETE(x)          if(x != NULL) delete x
#define SAFE_DELETE_ARRAY(x)    if(x != NULL) delete [] x

#define CPU             0
#define GPU             1
#define MULTI_GPU       0
#define NUM_DEVICES     2
#define BUILD_TYPE      GPU         //0 is CPU 1 is GPU

#define RADIX           1
#define BITONIC         2
#define SORT            BITONIC
 
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#if (BUILD_TYPE == GPU)
//CUDA
#include <cuda.h>
// #include <cublas_v2.h>
// #include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

cudaStream_t __streams[NUM_STREAMS];
cudaStream_t __multiStreams[NUM_DEVICES][NUM_STREAMS];
__constant__ int c_bin_offsets[8];
#endif

//openmp
#include <omp.h>

//cusp
//#include <cusp/dia_matrix.h>
//#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
//#include <cusp/hyb_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/sort.h>
#include <cusp/format_utils.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>
#include <cusp/blas/blas.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>

//dynamic ell
#include "dcsr_matrix.h"

struct is_non_negative
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return (x >= 0);
    }
};

struct is_positive
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return (x > 0);
    }
};

#endif
