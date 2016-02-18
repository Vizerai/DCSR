#ifndef DCSR_MATRIX_H
#define DCSR_MATRIX_H

#include "sparse.h"

#define ORDERING    ROW_MAJOR

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
struct dcsr_matrix         //dynamic ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_sizes;              //row sizes
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_offsets;            //row offsets
    cusp::array1d<INDEX_TYPE, MEM_TYPE> *column_indices;         //column indices
    cusp::array1d<VALUE_TYPE, MEM_TYPE> *values;                 //values

    //binning arrays
    cusp::array1d<INDEX_TYPE, MEM_TYPE> bins;                   //bins sizes
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_ids;                //row IDs
    cusp::array1d<INDEX_TYPE, MEM_TYPE> bin_offsets;            //bin offsets
    cusp::array1d<INDEX_TYPE, cusp::host_memory> bin_offsets_H; //host side bin offsets
    cusp::array1d<INDEX_TYPE, MEM_TYPE> Prow_sizes;             //permuted row sizes
    cusp::array1d<INDEX_TYPE, MEM_TYPE> permutation;            //permutation

    float alpha;            //alpha threshold
    size_t pitch;
    size_t bin_length;      //chunk length
    size_t mem_size;        //total memory used
    size_t num_rows;
    size_t num_cols;
    size_t num_entries;
    size_t num_layers;

    dcsr_matrix() : column_indices(NULL), values(NULL)
    {}
    ~dcsr_matrix()
    {
        SAFE_DELETE(column_indices);
        SAFE_DELETE(values);
    }

    void resize(const size_t n_rows, const size_t n_cols, const size_t bin_size, const float factor)
    {
        num_rows = n_rows;
        num_cols = n_cols;
        num_entries = 0;
        num_layers = BINS;
        bin_length = bin_size;

        pitch = ALIGN_UP(num_rows*2, 32);
        mem_size = bin_length * num_rows * factor; //(BINS/4);

        fprintf(stderr, "memsize:  %d\n", mem_size);

        row_sizes.resize(num_rows+1);           //1 extra for ending index (replaces MD)
        row_sizes.assign(num_rows+1, 0);
        row_offsets.resize(pitch*BINS);
        row_offsets.assign(pitch*BINS, -1);
        SAFE_DELETE(column_indices);
        SAFE_DELETE(values);
        column_indices = new cusp::array1d<INDEX_TYPE, MEM_TYPE>(mem_size, -1);
        values = new cusp::array1d<VALUE_TYPE, MEM_TYPE>(mem_size);

        bins.resize(num_rows);
        bins.assign(num_rows, 0);
        row_ids.resize(num_rows);
        row_ids.assign(num_rows, 0);
        Prow_sizes.resize(num_rows+1);
        Prow_sizes.assign(num_rows+1, 0);
        bin_offsets.resize(8);                  //1, 2, 4, 8, 16, 32, 512+
        bin_offsets.assign(8, 0);
        bin_offsets_H.resize(8);
        bin_offsets_H.assign(8, 0);
    }

    //use exact NNZ size
    void resize(const size_t n_rows, const size_t n_cols, const size_t NNZ)
    {
        num_rows = n_rows;
        num_cols = n_cols;
        num_entries = 0;
        num_layers = BINS;
        bin_length = 0;

        pitch = ALIGN_UP(num_rows*2, 32);
        mem_size = NNZ;

        fprintf(stderr, "NNZ:  %d\n", NNZ);

        row_sizes.resize(num_rows+1);           //1 extra for ending index (replaces MD)
        row_sizes.assign(num_rows+1, 0);
        row_offsets.resize(pitch*BINS);
        row_offsets.assign(pitch*BINS, -1);
        SAFE_DELETE(column_indices);
        SAFE_DELETE(values);
        column_indices = new cusp::array1d<INDEX_TYPE, MEM_TYPE>(mem_size, -1);
        values = new cusp::array1d<VALUE_TYPE, MEM_TYPE>(mem_size);

        bins.resize(num_rows);
        bins.assign(num_rows, 0);
        row_ids.resize(num_rows);
        row_ids.assign(num_rows, 0);
        Prow_sizes.resize(num_rows+1);
        Prow_sizes.assign(num_rows+1, 0);
        bin_offsets.resize(8);                  //1, 2, 4, 8, 16, 32, 512+
        bin_offsets.assign(8, 0);
        bin_offsets_H.resize(8);
        bin_offsets_H.assign(8, 0);
    }
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
struct dcsr_matrix_B         //dynamic ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_sizes;              //row sizes
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_offsets;            //row offsets
    cusp::array1d<INDEX_TYPE, MEM_TYPE> column_indices;         //column indices
    cusp::array1d<INDEX_TYPE, MEM_TYPE> Matrix_MD;

    float alpha;                //alpha threshold
    size_t pitch;
    size_t bin_length;          //chunk length
    size_t mem_size;            //total memory used
    size_t num_rows;
    size_t num_cols;
    size_t num_entries;
    size_t num_layers;

    void resize(const size_t n_rows, const size_t n_cols, const size_t bin_size)
    {
        num_rows = n_rows;
        num_cols = n_cols;
        num_entries = 0;
        num_layers = BINS;
        bin_length = bin_size;

        pitch = ALIGN_UP(num_rows*2, 32);
        mem_size = bin_length * num_rows * (BINS/4);

        fprintf(stderr, "memsize:  %d\n", mem_size);

        Matrix_MD.resize(4);
        row_sizes.resize(num_rows);
        row_offsets.resize(pitch*BINS);
        column_indices.resize(mem_size);
    }
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
struct hyb_matrix         //hyb ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> row_sizes;             //row sizes
    cusp::hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> matrix;

    size_t num_rows;
    size_t num_cols;
    size_t num_entries;
    size_t num_overflow;

    void resize(const size_t rows, const size_t cols, const size_t num_coo_entries, const size_t num_cols_per_row)
    {
        num_rows = rows;
        num_cols = cols;
        num_entries = 0;
        num_overflow = 0;

        matrix.resize(rows, cols, 0, num_coo_entries*2, num_cols_per_row);
        row_sizes.resize(rows+2,0);
        matrix.num_entries = 0;
        matrix.ell.num_entries = 0;
        matrix.coo.num_entries = 0;
    }
};

// template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, unsigned int BINS>
// struct dell_matrix         //dynamic hyb ELL matrix
// {
//     cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> ell;     //ell matrix
//     cusp::array1d<INDEX_TYPE, MEM_TYPE> row_sizes;              //row sizes
//     cusp::array1d<INDEX_TYPE, MEM_TYPE> row_offsets;            //row offsets
//     cusp::array1d<INDEX_TYPE, MEM_TYPE> column_indices;         //column indices
//     cusp::array1d<VALUE_TYPE, MEM_TYPE> values;                 //values
//     cusp::array1d<INDEX_TYPE, MEM_TYPE> Matrix_MD;

//     float alpha;                //alpha threshold
//     size_t pitch;
//     size_t chunk_length;        //chunk length
//     size_t mem_size;            //total memory used
//     size_t num_rows;
//     size_t num_cols;
//     size_t num_entries;
//     size_t num_layers;

//     void resize(const size_t n_rows, const size_t n_cols, const size_t num_cols_per_row, const size_t c_length)
//     {
//         num_rows = n_rows;
//         num_cols = n_cols;
//         num_entries = 0;
//         num_layers = BINS;
//         chunk_length = c_length;

//         pitch = ALIGN_UP(num_rows*2, 32);
//         mem_size = chunk_length * num_rows; //(BINS/4);

//         fprintf(stderr, "memsize:  %d\n", mem_size);

//         Matrix_MD.resize(4);
//         row_sizes.resize(num_rows+1);
//         row_offsets.resize(pitch*BINS);
//         column_indices.resize(mem_size);
//         values.resize(mem_size);

//         ell.resize(num_rows, num_cols, 0, num_cols_per_row);
//     }
// };


#include "primitives.h"
#include "scan.h"
#include "load.h"
//#include "spmv.h"
#include "spmm.h"

#endif