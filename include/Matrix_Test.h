#ifndef MATRIX_TEST_H
#define MATRIX_TEST_H

#define ARG_MAX                     512
#define BLOCKS                      26
#define BLOCK_THREAD_SIZE           256
#define BLOCK_THREADS_MAX		    512
#define DEFAULT_OVERFLOW            256
#define MAX_OFFSET                  4
#define VECTOR_SIZE                 4
#define RUN_HYB                     0
#define RUN_CSR                     1
#define RUN_DCSR                    1
#define SPMV_TEST                   0
#define SPMM_TEST                   1
#define USE_PARTIAL_VECTORS         0
#define RUN_ADD                     0
#define NUM_STREAMS                 8
#define PRECISION                   32

#define MEMORY_ALIGNMENT    4096
#define ALIGN_UP(x,size)    ( ((size_t)x+(size-1))&(~(size-1)) ) //works for size that is a power of 2
#define ROUND_UP(x,y)       ( (x + y-1) / y )
#define SAFE_DELETE(x)      if(x != NULL) delete x

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

//dynamic ell
#include "dcsr_matrix.h"

#if(BUILD_TYPE == GPU)

// template <typename VALUE_TYPE>
// void AND_OP(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
//             const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
//             cusp::array1d<VALUE_TYPE, cusp::device_memory> &c);

// template <typename VALUE_TYPE>
// void get_indices(   const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
//                     cusp::array1d<VALUE_TYPE, cusp::device_memory> &b);

// template <typename VALUE_TYPE>
// void AccumVec(  cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
//                 const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b);

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void AccumMat(  const cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory> &mat,
//                 cusp::array1d<VALUE_TYPE, cusp::device_memory> &vec);

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void column_select( const cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory> &A,
//                     const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s,
//                     const INDEX_TYPE index,
//                     cusp::array1d<VALUE_TYPE, cusp::device_memory> &y);

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void OuterProduct(  const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
//                     const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
//                     cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat);

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void ell_add(   cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
//                 cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &B,
//                 cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &C);

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void ell_spmv(  const cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
//                 const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
//                 cusp::array1d<VALUE_TYPE, cusp::device_memory> &y);

#endif      //GPU

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
