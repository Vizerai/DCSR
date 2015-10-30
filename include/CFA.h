#ifndef CFA_H
#define CFA_H

#define NUM_GPUS                1
#define ARG_MAX                 512
#define BLOCKS                  32
#define BLOCK_THREADS           256
#define NUM_STREAMS             9
#define NUM_STREAMS             8
#define NUM_TRANSFER_FUNCS      7

#define MEMORY_ALIGNMENT    4096
#define ALIGN_UP(x,size)    ( ((size_t)x+(size-1))&(~(size-1)) )
#define ROUND_UP(x,y)       ( (x + y-1) / y )

#define STREAM_CALL     0
#define STREAM_LIST     1
#define STREAM_SET      2
#define STREAM_IF       3
#define STREAM_BOOL     4
#define STREAM_NUM      5
#define STREAM_VOID     6
#define STREAM_UPDATE   7

#define CPU             0
#define GPU             1
//#define MULTI_GPU     2
#define BUILD_TYPE      GPU         //0 is CPU 1 is GPU
#define GPU_HYBRID      0
#define GPU_DYNAMIC     1
#define GPU_MATRIX      GPU_HYBRID

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#if BUILD_TYPE == GPU
//CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#endif

//openmp
#include <omp.h>

//cusp
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/multiply.h>
#include <cusp/elementwise.h>
#include <cusp/transpose.h>
#include <cusp/blas.h>
#include <cusp/print.h>

//sparse matrix
#include "matrix_types.h"
#include "dell_matrix.h"
#include "sparse.h"
#include "matrix_types.h"

bool compare_entry(const std::pair<int,int> &a, const std::pair<int,int> &b)
{
    //first -> row
    //second -> column
    if(a.first < b.first)
        return true;
    else if(b.first < a.first)
        return false;
    else
        return (a.second < b.second);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
struct dell_matrix         //dynamic ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> column_indicesA;
    cusp::array1d<INDEX_TYPE, MEM_TYPE> column_indicesB;
    cusp::array1d<INDEX_TYPE, MEM_TYPE> *column_indices;
    cusp::coo_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> coo;
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_offsetsA;
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_offsetsB;
    cusp::array1d<INDEX_TYPE, MEM_TYPE> *row_offsets;
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_sizes;

    size_t mem_size;
    size_t num_rows;
    size_t num_cols;
    size_t num_entries;

    void resize(const size_t rows, const size_t cols, const size_t entries, const size_t size, const size_t coo_size)
    {
        num_rows = rows;
        num_cols = cols;
        num_entries = entries;
        mem_size = size;

        column_indicesA.resize(size, -1);
        column_indicesB.resize(size, -1);
        row_offsetsA.resize(rows+1,0);
        row_offsetsB.resize(rows+1,0);
        row_sizes.resize(rows,0);
		coo.row_indices.resize(coo_size,-1);
		coo.column_indices.resize(coo_size,-1);
        coo.resize(rows, cols, coo_size);
        coo.row_indices[0] = 0;
        coo.column_indices[0] = 0;

        row_offsets = &row_offsetsA;
        column_indices = &column_indicesA;
    }
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
struct hyb_matrix         //dynamic ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_sizes;
    cusp::hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> matrix;

    size_t num_rows;
    size_t num_cols;
    size_t num_entries;

    void resize(const size_t rows, const size_t cols, const size_t num_ell_entries, const size_t num_cols_per_row)
    {
        num_rows = rows;
        num_cols = cols;
        num_entries = num_ell_entries;

        matrix.resize(rows, cols, num_ell_entries, 0, num_cols_per_row);
        row_sizes.resize(rows, 0);
    }

    void resize(const size_t rows, const size_t cols, const size_t num_ell_entries, const size_t num_coo_entries, const size_t num_cols_per_row)
    {
        num_rows = rows;
        num_cols = cols;
        num_entries = num_ell_entries;

        matrix.resize(rows, cols, num_ell_entries, num_coo_entries, num_cols_per_row);
        row_sizes.resize(rows, 0);
    }
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
class CFA
{
private:
    int m_maxCall;
    int m_maxList;

    bool debug;

#if BUILD_TYPE == CPU
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> temp_Mat[8];
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> sigma;
#elif BUILD_TYPE == GPU
    //dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> sigma;
    hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> sigma;
    dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> sigma;
    //hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> sigma;

    INDEX_TYPE *entry_count_host, *entry_count_device;

    cudaStream_t stream_Call;
    cudaStream_t stream_List;
    cudaStream_t stream_Set;
    cudaStream_t stream_If;
    cudaStream_t stream_Num;
    cudaStream_t stream_Bool;
    cudaStream_t stream_Void;
    cudaStream_t stream_Update;
#endif
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> CondTrue;
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> CondFalse;
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> Body;
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> Fun;
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> Arg[ARG_MAX];
    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> Var[ARG_MAX];

    //const vectors
    cusp::array1d<VALUE_TYPE, MEM_TYPE> IF;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> SET;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> PrimBool;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> PrimVoid;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> PrimNum;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> PrimList[ARG_MAX];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Call[ARG_MAX];
    std::vector<bool> valid_Call, valid_List;

    cusp::array1d<VALUE_TYPE, MEM_TYPE> r;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> r_prime;

#if BUILD_TYPE == GPU
    cusp::array1d<VALUE_TYPE, MEM_TYPE> s[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> s_old[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> s_indices[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> update_queue[NUM_STREAMS];
    int update_flags[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> update_queue[NUM_STREAMS][2];
    int update_flags[NUM_STREAMS][2];
#else
    cusp::array1d<VALUE_TYPE, MEM_TYPE> s;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> s_indices;
#endif
    
    cusp::array1d<VALUE_TYPE, MEM_TYPE> a[ARG_MAX];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> a_set;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> v[ARG_MAX];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> v_set;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> v_cond;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> v_list;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> AND_vec1;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> AND_vec2;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Cond_vec;

#if BUILD_TYPE == GPU
    cusp::array1d<VALUE_TYPE, MEM_TYPE> a_indices;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> v_indices;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> index_count[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> temp_row_indices[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> temp_col_indices[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> a_var[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> vf[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Fun_vec[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Body_vec[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Arg_vec[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> accum_var_vec[NUM_STREAMS];
    cusp::array1d<VALUE_TYPE, MEM_TYPE> accum_vf_vec[NUM_STREAMS];
#else
    cusp::array1d<VALUE_TYPE, MEM_TYPE> a_var;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> vf;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Fun_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Body_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> Arg_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> accum_var_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> accum_vf_vec;
#endif

    void f_call();
    void f_list();
    void f_set();
    void f_if();
    void f_primBool();
    void f_primNum();
    void f_primVoid();
    void f_UpdateStore();

    unsigned int CountEntries(cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat);
    unsigned int CountEntries(dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat);
    unsigned int CountEntries(hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat);

#if(BUILD_TYPE == CPU)
    void f_call_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s, const int j);
    void f_list_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s, const int j);
    void f_set_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s);
    void f_if_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s);
    void f_primBool_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s);
    void f_primNum_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s);
    void f_primVoid_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s);
#elif(BUILD_TYPE == GPU)
    void f_call_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int j, const int stream_ID, cudaStream_t stream);
    void f_list_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int j, const int stream_ID, cudaStream_t stream);
    void f_set_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int stream_ID, cudaStream_t stream);
    void f_if_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int stream_ID, cudaStream_t stream);
    void f_primBool_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int stream_ID, cudaStream_t stream);
    void f_primNum_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int stream_ID, cudaStream_t stream);
    void f_primVoid_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int stream_ID, cudaStream_t stream);
    void f_UpdateStore_device(const int stream_ID, cudaStream_t stream);

    void LoadMatrix(cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
                    cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst);
    void LoadMatrix(cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
                    hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst);
    void LoadMatrix(cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
                    dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst);

    void RebuildMatrix(dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat);
#endif

    cusp::array1d<VALUE_TYPE, MEM_TYPE> temp_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> VOID_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> NOT_FALSE_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> FALSE_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> NUM_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> BOOL_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> LIST_vec;
    cusp::array1d<VALUE_TYPE, MEM_TYPE> tb, fb;

public:
    CFA() : m_maxCall(0), m_maxList(0) {};
    CFA(char *filename) : m_maxCall(0), m_maxList(0)
    {
        ReadTestFile(filename);
        Init();
    };
    ~CFA() {};

    //setup and runtime functions
    void ReadTestFile(const char* filename);
    void Init();
    void Run_Analysis();
    void WriteStore();
    void CopyStore( cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat,
                    cusp::coo_matrix<int, VALUE_TYPE, cusp::host_memory> &store);
    void CopyStore( hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat,
                    cusp::coo_matrix<int, VALUE_TYPE, cusp::host_memory> &store);
    void CopyStore( dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat,
                    cusp::coo_matrix<int, VALUE_TYPE, cusp::host_memory> &store);

    //GPU calls
    void Init_CPU();

    //CPU calls
    void Init_GPU();

    //Debug functions
};

#if(BUILD_TYPE == CPU)
template <typename VALUE_TYPE>
void AND_OP(    const cusp::array1d<VALUE_TYPE, cusp::host_memory> &A, 
                const cusp::array1d<VALUE_TYPE, cusp::host_memory> &B,
                std::vector<VALUE_TYPE> &vec);

template <typename VALUE_TYPE>
void AccumVec(  cusp::array1d<VALUE_TYPE, cusp::host_memory> &a,
                const cusp::array1d<VALUE_TYPE, cusp::host_memory> &b);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProduct(  const cusp::array1d<VALUE_TYPE, cusp::host_memory> &a,
                    const cusp::array1d<VALUE_TYPE, cusp::host_memory> &b,
                    cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> &mat);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProduct(  const cusp::array1d<VALUE_TYPE, cusp::host_memory> &a,
                    const cusp::array1d<VALUE_TYPE, cusp::host_memory> &b,
                    cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> &mat);
#endif      //CPU

#if(BUILD_TYPE == GPU)

template <typename VALUE_TYPE>
void AND_OP(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
            const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
            cusp::array1d<VALUE_TYPE, cusp::device_memory> &c);

template <typename VALUE_TYPE>
void get_indices(   const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
                    cusp::array1d<VALUE_TYPE, cusp::device_memory> &b);

template <typename VALUE_TYPE>
void AccumVec(  cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
                const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void AccumMat(  const cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory> &mat,
                cusp::array1d<VALUE_TYPE, cusp::device_memory> &vec);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void column_select( const cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory> &A,
                    const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s,
                    const INDEX_TYPE index,
                    cusp::array1d<VALUE_TYPE, cusp::device_memory> &y);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProduct(  const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
                    const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
                    cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void ell_add(   cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
                cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &B,
                cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &C);

template <typename INDEX_TYPE, typename VALUE_TYPE>
void ell_spmv(  const cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
                const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
                cusp::array1d<VALUE_TYPE, cusp::device_memory> &y);

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

template <typename INDEX_TYPE>
struct mat_info
{
    INDEX_TYPE num_rows;
    INDEX_TYPE num_cols;
    INDEX_TYPE num_entries;
    INDEX_TYPE num_entries_coo;
    INDEX_TYPE num_cols_per_row;
    INDEX_TYPE pitch;
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void get_matrix_info(const dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat, mat_info<INDEX_TYPE> &info)
{
    info.num_rows = mat.num_rows;
    info.num_cols = mat.num_cols;
    info.num_entries = mat.num_entries;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void get_matrix_info(const hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat, mat_info<INDEX_TYPE> &info)
{
    info.num_rows = mat.num_rows;
    info.num_cols = mat.num_cols;
    info.num_entries = mat.num_entries;
    info.num_entries_coo = mat.matrix.coo.row_indices.size();
    info.num_cols_per_row = mat.matrix.ell.column_indices.num_cols;
    info.pitch = mat.matrix.ell.column_indices.pitch;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void get_matrix_info(const cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat, mat_info<INDEX_TYPE> &info)
{
    info.num_rows = mat.num_rows;
    info.num_cols = mat.num_cols;
    info.num_entries = mat.num_entries;
    info.num_cols_per_row = mat.column_indices.num_cols;
    info.pitch = mat.column_indices.pitch;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void get_matrix_info(const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat, mat_info<INDEX_TYPE> &info)
{
    info.num_rows = mat.num_rows;
    info.num_cols = mat.num_cols;
    info.num_entries = mat.num_entries;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void print_matrix_info(hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat)
{
    mat_info<INDEX_TYPE> info;
    get_matrix_info(mat, info);
    fprintf(stderr, "hyb matrix - rows: %d  cols: %d  num_entries: %d  num_entries_coo: %d  num_cols_per_row: %d  pitch: %d\n", 
        info.num_rows, info.num_cols, info.num_entries, info.num_entries_coo, info.num_cols_per_row, info.pitch);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void print_matrix_info(cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat)
{
    mat_info<INDEX_TYPE> info;
    get_matrix_info(mat, info);
    fprintf(stderr, "ell matrix - rows: %d  cols: %d  num_entries: %d  num_cols_per_row: %d  pitch: %d\n", 
        info.num_rows, info.num_cols, info.num_entries, info.num_cols_per_row, info.pitch);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void print_matrix_info(cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat)
{
    mat_info<INDEX_TYPE> info;
    get_matrix_info(mat, info);
    fprintf(stderr, "csr matrix - rows: %d  cols: %d  num_entries: %d\n", 
        info.num_rows, info.num_cols, info.num_entries);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void print_matrix_info(dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat)
{
    mat_info<INDEX_TYPE> info;
    get_matrix_info(mat, info);
    fprintf(stderr, "dell matrix - rows: %d  cols: %d  num_entries: %d\n", 
        info.num_rows, info.num_cols, info.num_entries);
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void debug_print(const cusp::hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src_mat)
{
    cusp::hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> mat(src_mat);

    mat_info<INDEX_TYPE> info;
    get_matrix_info(mat, info);
    const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory>::invalid_index;

    int count = 0;
    for(int row=0; row<info.num_rows; ++row)
    {
        INDEX_TYPE offset = row+info.pitch;
        for(int n=1; n<info.num_cols_per_row; ++n, offset+=info.pitch)
        {
            if(mat.ell.column_indices.values[offset] != invalid_index)
            {
                fprintf(stderr, "\t%d\t%d\t%d\n", mat.ell.column_indices.values[row], row, mat.ell.column_indices.values[offset]);
                if(n == info.num_cols_per_row-1)
                    fprintf(stderr, "***FULL ROW***\n");
                count++;
            }
        }

        for(int n=1; n<mat.coo.column_indices[0]; ++n)
        {
            if(mat.coo.row_indices[n] == row)
            {
                fprintf(stderr, "\t%d\t%d\n", mat.coo.row_indices[n], mat.coo.column_indices[n]);
                count++;
            }
        }
    }

    fprintf(stderr, "number of entries: %d\n", count);
}

#endif
