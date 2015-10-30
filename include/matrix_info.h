#ifndef MATRIX_INFO_H
#define MATRIX_INFO_H

template <typename INDEX_TYPE>
struct mat_info
{
    INDEX_TYPE num_rows;
    INDEX_TYPE num_cols;
    INDEX_TYPE num_entries;
    INDEX_TYPE num_entries_coo;
    INDEX_TYPE num_cols_per_row;
    INDEX_TYPE pitch;
    INDEX_TYPE bin_length;
    float alpha;
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
inline void get_matrix_info(const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE, BINS> &mat, mat_info<INDEX_TYPE> &info)
{
    info.num_rows = mat.num_rows;
    info.num_cols = mat.num_cols;
    info.num_entries = mat.num_entries;
    info.bin_length = mat.bin_length;
    info.pitch = mat.pitch;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
inline void get_matrix_info(const dcsr_matrix_B<INDEX_TYPE, VALUE_TYPE, MEM_TYPE, BINS> &mat, mat_info<INDEX_TYPE> &info)
{
    info.num_rows = mat.num_rows;
    info.num_cols = mat.num_cols;
    info.num_entries = mat.num_entries;
    info.bin_length = mat.bin_length;
    info.pitch = mat.pitch;
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

// template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
// inline void get_matrix_info(const dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE, BINS> &mat, mat_info<INDEX_TYPE> &info)
// {
//     info.num_rows = mat.num_rows;
//     info.num_cols = mat.num_cols;
//     info.num_entries = mat.num_entries;
//     info.num_entries_coo = mat.matrix.coo.row_indices.size();
//     info.chunk_length = mat.chunk_length;
//     info.num_cols_per_row = mat.matrix.ell.column_indices.num_cols;
//     info.pitch = mat.matrix.ell.column_indices.pitch;
// }

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

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
inline void print_matrix_info(dcsr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE, BINS> &mat)
{
    mat_info<INDEX_TYPE> info;
    get_matrix_info(mat, info);
    fprintf(stderr, "dcsr matrix - rows: %d  cols: %d  num_entries: %d  pitch: %d\n", 
        info.num_rows, info.num_cols, info.num_entries, info.pitch);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
inline void print_matrix_info(cusp::hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat)
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