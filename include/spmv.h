#ifndef SPMV_H
#define SPMV_H

#include "spmv.inl"

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

	spmv_ellb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
            infoA.num_cols_per_row,
            infoA.pitch,
            TPC(&A.column_indices.values[0]),
            TPC(&x[0]), 
        	TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

	spmv_hyb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
    	    infoA.num_cols_per_row,
   			infoA.pitch,
        	TPC(&A.matrix.ell.column_indices.values[0]),
        	TPC(&A.matrix.ell.values.values[0]),
        	TPC(&A.matrix.coo.row_indices[0]),
        	TPC(&A.matrix.coo.column_indices[0]),
        	TPC(&A.matrix.coo.values[0]),
        	TPC(&A.rs[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

	spmv_hyb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0>>> (
			infoA.num_rows,
    	    infoA.num_cols_per_row,
   			infoA.pitch,
        	TPC(&A.matrix.ell.column_indices.values[0]),
        	TPC(&A.matrix.ell.values.values[0]),
        	TPC(&A.matrix.coo.row_indices[0]),
        	TPC(&A.matrix.coo.column_indices[0]),
        	TPC(&A.matrix.coo.values[0]),
        	TPC(&A.rs[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

	spmv_csr<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
        	TPC(&A.row_offsets[0]),
        	TPC(&A.column_indices[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void spmv(	const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			const cudaStream_t *streams)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);
	// fprintf(stderr, "dcsr matrix - rows: %d  cols: %d  num_entries: %d  pitch: %d\n", 
 	//        infoA.num_rows, infoA.num_cols, infoA.num_entries, infoA.pitch);

	const size_t BLOCK_SIZE = 128;
	const size_t BLOCK_SIZE_LARGE = 256;

	// const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
	// const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
	// const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
	// const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(A.num_rows, BLOCK_SIZE));

	// //fprintf(stderr, "MAX_BLOCKS: %d  NUM_BLOCKS: %d\n", MAX_BLOCKS, NUM_BLOCKS);
	// spmv_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE, 0, streams[0]>>> (
	// 		infoA.num_rows,
	// 		infoA.pitch,
	// 		TPC(&A.row_offsets[0]),
	// 		TPC(&A.row_sizes[0]),
	// 		TPC(&(*A.column_indices)[0]),
	// 		TPC(&(*A.values)[0]),
	// 		TPC(&x[0]), 
	// 		TPC(&y[0]));

	const size_t MAX_BLOCKS1 = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_bin_scalar<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS1 = std::min<size_t>(MAX_BLOCKS1, ROUND_UP(A.num_rows, BLOCK_SIZE));
	const size_t MAX_BLOCKS2 = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/2, 2>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS2 = std::min<size_t>(MAX_BLOCKS2, ROUND_UP(A.num_rows, BLOCK_SIZE));
	const size_t MAX_BLOCKS4 = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/4, 4>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS4 = std::min<size_t>(MAX_BLOCKS4, ROUND_UP(A.num_rows, BLOCK_SIZE));
	const size_t MAX_BLOCKS8 = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/8, 8>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS8 = std::min<size_t>(MAX_BLOCKS8, ROUND_UP(A.num_rows, BLOCK_SIZE));
	const size_t MAX_BLOCKS16 = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/16, 16>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS16 = std::min<size_t>(MAX_BLOCKS16, ROUND_UP(A.num_rows, BLOCK_SIZE));
	const size_t MAX_BLOCKS32 = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/32, 32>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS32 = std::min<size_t>(MAX_BLOCKS32, ROUND_UP(A.num_rows, BLOCK_SIZE));
	const size_t MAX_BLOCKS_L = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_bin_large<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE_LARGE>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_L = std::min<size_t>(MAX_BLOCKS_L, ROUND_UP(A.num_rows, BLOCK_SIZE_LARGE));

	//fprintf(stderr, "NUM_BLOCKS1: %d  MAX_BLOCKS1: %d\n", NUM_BLOCKS1, MAX_BLOCKS1);
	spmv_dcsr_bin_scalar<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE> <<<NUM_BLOCKS1, BLOCK_SIZE, 0, streams[0]>>> (
			infoA.num_rows,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.Prow_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&x[0]), 
			TPC(&y[0]),
			TPC(&A.row_ids[0]));

	//fprintf(stderr, "NUM_BLOCKS2: %d  MAX_BLOCKS2: %d\n", NUM_BLOCKS2, MAX_BLOCKS2);
	spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/2, 2> <<<NUM_BLOCKS2, BLOCK_SIZE, 0, streams[1]>>> (
			infoA.num_rows,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.Prow_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&x[0]), 
			TPC(&y[0]),
			TPC(&A.row_ids[0]));

	//fprintf(stderr, "NUM_BLOCKS4: %d  MAX_BLOCKS4: %d\n", NUM_BLOCKS4, MAX_BLOCKS4);
	spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/4, 4> <<<NUM_BLOCKS4, BLOCK_SIZE, 0, streams[2]>>> (
			infoA.num_rows,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.Prow_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&x[0]), 
			TPC(&y[0]),
			TPC(&A.row_ids[0]));

	//fprintf(stderr, "NUM_BLOCKS8: %d  MAX_BLOCKS8: %d\n", NUM_BLOCKS8, MAX_BLOCKS8);
	spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/8, 8> <<<NUM_BLOCKS8, BLOCK_SIZE, 0, streams[3]>>> (
			infoA.num_rows,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.Prow_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&x[0]), 
			TPC(&y[0]),
			TPC(&A.row_ids[0]));

	//fprintf(stderr, "NUM_BLOCKS16: %d  MAX_BLOCKS16: %d\n", NUM_BLOCKS16, MAX_BLOCKS16);
	spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/16, 16> <<<NUM_BLOCKS16, BLOCK_SIZE, 0, streams[4]>>> (
			infoA.num_rows,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.Prow_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&x[0]), 
			TPC(&y[0]),
			TPC(&A.row_ids[0]));

	//fprintf(stderr, "NUM_BLOCKS32: %d  MAX_BLOCKS32: %d\n", NUM_BLOCKS32, MAX_BLOCKS32);
	spmv_dcsr_bin<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE/32, 32> <<<NUM_BLOCKS32, BLOCK_SIZE, 0, streams[5]>>> (
			infoA.num_rows,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.Prow_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&x[0]), 
			TPC(&y[0]),
			TPC(&A.row_ids[0]));

	spmv_dcsr_bin_large<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE_LARGE> <<<NUM_BLOCKS_L, BLOCK_SIZE_LARGE, 0, streams[6]>>> (
			infoA.num_rows,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.Prow_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&x[0]), 
			TPC(&y[0]),
			TPC(&A.row_ids[0]));
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS>
// void spmv_sorted(	const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &A,
//      				const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
// 					cusp::array1d<VALUE_TYPE, cusp::device_memory> &y)
// {
// 	mat_info<INDEX_TYPE> infoA;
// 	get_matrix_info(A, infoA);

// 	const size_t BLOCK_SIZE = 128;
// 	const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
// 	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

// 	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr_sorted<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
// 	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(A.num_rows, VECTORS_PER_BLOCK));

// 	spmv_dcsr_sorted<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
// 			infoA.num_rows,
// 			infoA.pitch,
// 			TPC(&A.row_offsets[0]),
// 			TPC(&A.row_sizes[0]),
// 			TPC(&A.column_indices[0]),
// 			TPC(&A.values[0]),
// 			TPC(&x[0]), 
// 			TPC(&y[0]));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS>
// void spmv(	const dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &A,
//      		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
// 			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y)
// {
// 	mat_info<INDEX_TYPE> infoA;
// 	get_matrix_info(A, infoA);

// 	cusp::array1d<VALUE_TYPE, cusp::device_memory> T1(y.size()), T2(y.size());

// 	//multiply ell portion
// 	cusp::multiply(A.ell, x, T1);
	
// 	const size_t BLOCK_SIZE = 128;
// 	const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
// 	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

// 	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
// 	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(A.num_rows, VECTORS_PER_BLOCK));
	
// 	//multiply dcsr portion
// 	spmv_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
// 			infoA.num_rows,
// 			infoA.pitch,
// 			TPC(&A.row_offsets[0]),
// 			TPC(&A.row_sizes[0]),
// 			TPC(&A.column_indices[0]),
// 			TPC(&A.values[0]),
// 			TPC(&x[0]), 
// 			TPC(&T2[0]));

// 	cusp::add(T1, T2, y);
// }

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info(A, infoA);

	const size_t BLOCK_SIZE = 128;
	const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_csr_vector_kernel<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(A.num_rows, VECTORS_PER_BLOCK));

	spmv_csr_vector_kernel<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			A.num_rows,
			TPC(&A.row_offsets[0]),
        	TPC(&A.column_indices[0]),
        	TPC(&A.values[0]),
        	TPC(&x[0]), 
    		TPC(&y[0]));
}

}

#endif