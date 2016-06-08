#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "matrix_info.h"
#include "primitives_device.h"

#include "sparse_update.inl"

namespace device
{

/////////////////////////////////////////////////////////////////////////
/////////////////  Entry Wrapper Functions  /////////////////////////////
/////////////////////////////////////////////////////////////////////////
// void InitCuPrint()
// {
// 	cudaPrintfInit();
// }

// template <typename VALUE_TYPE>
// void FILL(	cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 			const VALUE_TYPE value,
// 			cudaStream_t &stream)
// {
// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	FILL<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
// 			TPC(&a[0]),
// 			value,
// 			int(a.size()));
// }

// template <typename VALUE_TYPE>
// void AND_OP(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 			const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 			cusp::array1d<VALUE_TYPE, cusp::device_memory> &c,
// 			cudaStream_t &stream)
// {
// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	AND_OP<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
// 			TPC(&a[0]),
// 			TPC(&b[0]),
// 			TPC(&c[0]),
// 			int(a.size()));
// }

// template <typename VALUE_TYPE>
// void get_indices(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 					cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 					cudaStream_t &stream)
// {
// 	const size_t NUM_BLOCKS = 1;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	get_indices<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
// 			TPC(&a[0]),
// 			TPC(&b[0]),
// 			int(a.size()));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void count(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 			const VALUE_TYPE val,
// 			INDEX_TYPE *h_res,
// 			INDEX_TYPE *d_res,
// 			cudaStream_t &stream)
// {
// 	const size_t NUM_BLOCKS = 1;
// 	const size_t BLOCK_SIZE = 512;

// 	count<INDEX_TYPE, VALUE_TYPE, 512> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
// 			TPC(&a[0]),
// 			val,
// 			d_res,
// 			int(a.size()));

// 	cudaMemcpyAsync(h_res, d_res, sizeof(INDEX_TYPE), cudaMemcpyDeviceToHost, stream);
// }

// template <typename VALUE_TYPE>
// void gather_reduce(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 					cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 					cusp::array1d<VALUE_TYPE, cusp::device_memory> &indices,
// 					const int index,
// 					cudaStream_t &stream)
// {
// 	const size_t NUM_BLOCKS = 1;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// #if DEBUG
// 	assert(a.size() == b.size());
// #endif

// 	gather_reduce<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
// 			TPC(&a[0]),
// 			TPC(&b[0]),
// 			TPC(&indices[index]),
// 			int(a.size()));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void column_select(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
// 					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s,
// 					const INDEX_TYPE index,
// 					cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
// 					cudaStream_t &stream)
// {
// 	mat_info<INDEX_TYPE> infoA;
// 	get_matrix_info(A, infoA);

// #if(DEBUG)
// 	assert(infoA.num_cols == s.size());
// 	assert(infoA.num_rows == y.size());
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	column_select<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
// 			infoA.num_rows,
// 			TPC(&A.row_offsets[0]),
// 			TPC(&A.column_indices[0]),
// 			TPC(&s[index]),
// 			TPC(&y[0]));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void column_select_if(	const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &cond,
// 						const INDEX_TYPE index,
// 						cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
// 						cudaStream_t &stream)
// {
// 	mat_info<INDEX_TYPE> infoA;
// 	get_matrix_info(A, infoA);

// #if(DEBUG)
// 	assert(infoA.num_cols == s.size());
// 	assert(infoA.num_rows == y.size());
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	column_select_if<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
// 			infoA.num_rows,
// 			TPC(&A.row_offsets[0]),
// 			TPC(&A.column_indices[0]),
// 			TPC(&s[index]),
// 			TPC(&cond[index]),
// 			TPC(&y[0]));
// }

// template <typename VALUE_TYPE>
// void AccumVec(	cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 				const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 				cudaStream_t &stream)
// {
// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	AccumVec<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
// 			TPC(&a[0]),
// 			TPC(&b[0]),
// 			int(a.size()));
// }

// template <typename VALUE_TYPE>
// void InnerProductStore(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 						cusp::array1d<VALUE_TYPE, cusp::device_memory> &c,
// 						const int index,
// 						cudaStream_t &stream)
// {
// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	InnerProductStore<VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
// 			TPC(&a[0]),
// 			TPC(&b[0]),
// 			int(a.size()),
// 			TPC(&c[index]));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void OuterProduct(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 					cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
// 					cudaStream_t &stream)
// {
// 	mat_info<INDEX_TYPE> info;
// 	get_matrix_info(mat, info);

// #if(DEBUG)
// 	assert(src.num_rows == infoDst.num_rows);
// 	assert(src.num_cols == infoDst.num_cols);
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	//OuterProduct<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>>
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
// 						cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
// 						cudaStream_t &stream)
// {
// 	mat_info<INDEX_TYPE> info;
// 	get_matrix_info(mat, info);

// #if(DEBUG)
// 	assert(info.num_rows == a.size());
// 	assert(info.num_cols == b.size());
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	OuterProductAdd_ELL_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
// 			TPC(&a[0]),
// 			TPC(&b[0]),
// 			TPC(&index_count[0]),
// 			info.num_rows,
// 			info.num_cols,
// 			info.num_cols_per_row,
// 			info.pitch,
// 			TPC(&mat.column_indices.values[0]),
// 			TPC(&mat.values.values[0]));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
// 						hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
// 						cudaStream_t &stream)
// {
// 	mat_info<INDEX_TYPE> info;
// 	get_matrix_info(mat, info);

// #if(DEBUG)
// 	assert(info.num_rows == a.size());
// 	assert(info.num_cols == b.size());
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	OuterProductAdd_HYB_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
// 			TPC(&a[0]),
// 			TPC(&b[0]),
// 			TPC(&index_count[0]),
// 			info.num_rows,
// 			info.num_cols,
// 			info.num_cols_per_row,
// 			info.pitch,
// 			TPC(&mat.row_sizes[0]),
// 			TPC(&mat.matrix.ell.column_indices.values[0]),
// 			TPC(&mat.matrix.coo.row_indices[0]),
// 			TPC(&mat.matrix.coo.column_indices[0]));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS>
// void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
// 						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
// 						dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat,
// 						cudaStream_t &stream)
// {
// // 	mat_info<INDEX_TYPE> info;
// // 	get_matrix_info(mat, info);

// // #if(DEBUG)
// // 	assert(info.num_rows == a.size());
// // 	assert(info.num_cols == b.size());
// // #endif

// // 	const size_t NUM_BLOCKS = BLOCKS;
// // 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void ell_add(	cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
// 				cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &B,
// 				cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &C,
// 				cudaStream_t &stream)
// {
// 	// mat_info<INDEX_TYPE> infoA, infoB, infoC;

// 	// get_matrix_info(A, infoA);
// 	// get_matrix_info(B, infoB);
// 	// get_matrix_info(C, infoC);

// 	//fix this
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS>
// void add(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &A,
// 			const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &B)
// {
// 	mat_info<INDEX_TYPE> infoA, infoB;

// 	get_matrix_info(A, infoA);
// 	get_matrix_info(B, infoB);

// 	assert(A.num_rows == B.num_rows);
// 	assert(A.num_cols == B.num_cols);

// 	const size_t BLOCK_SIZE = 256;
// 	const size_t THREADS_PER_VECTOR = 32;
// 	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

// 	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(add_matrix_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
// 	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(A.num_rows, VECTORS_PER_BLOCK));

// 	add_matrix_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
// 			infoA.num_rows,
// 			infoA.pitch,
// 			infoB.pitch,
// 			TPC(&A.column_indices[0]),
// 			TPC(&A.values[0]),
// 			TPC(&A.row_offsets[0]),
// 			TPC(&A.row_sizes[0]),
// 			TPC(&B.column_indices[0]),
// 			TPC(&B.values[0]),
// 			TPC(&B.row_offsets[0]),
// 			TPC(&B.row_sizes[0]));

// 	safeSync();
// 	//cudaPrintfDisplay(stdout, true);

// 	const size_t BLOCK_SIZE_B = BLOCK_THREAD_SIZE;
// 	const size_t THREADS_PER_VECTOR_A = 16;
// 	const size_t VECTORS_PER_BLOCK_A = BLOCK_SIZE_B / THREADS_PER_VECTOR_A;

// 	const size_t MAX_BLOCKS_A = cusp::system::cuda::detail::max_active_blocks(SetRowIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A, THREADS_PER_VECTOR_A>, BLOCK_SIZE_B, (size_t) 0);
// 	const size_t NUM_BLOCKS_A = std::min<size_t>(MAX_BLOCKS_A, ROUND_UP(A.num_rows, VECTORS_PER_BLOCK_A));

// 	INDEX_TYPE N = A.row_sizes[infoA.num_rows]; //1 larger as buffer
// 	cusp::array1d<INDEX_TYPE, cusp::device_memory> I(N + 1);
// 	cusp::array1d<INDEX_TYPE, cusp::device_memory> J(N + 1);
// 	cusp::array1d<VALUE_TYPE, cusp::device_memory> V(N + 1);
// 	cusp::array1d<INDEX_TYPE, cusp::device_memory> T(N + 1);
// 	fprintf(stderr, "N: %d\n", N);

// 	//set row indices
// 	// SetRowIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A, THREADS_PER_VECTOR_A> <<<NUM_BLOCKS_A, BLOCK_SIZE_B>>> (
// 	// 		infoA.num_rows,
// 	// 		infoA.pitch,
// 	// 		TPC(&I[0]),
// 	// 		TPC(&A.row_offsets[0]),
// 	// 		TPC(&A.row_sizes[0]));

// 	// safeSync();

// 	cudaMemcpy(TPC(&J[0]), TPC(&A.column_indices[0]), N*sizeof(INDEX_TYPE), cudaMemcpyDeviceToDevice);
// 	cudaMemcpy(TPC(&V[0]), TPC(&A.values[0]), N*sizeof(INDEX_TYPE), cudaMemcpyDeviceToDevice);
// 	//thrust::copy_n(A.column_indices.begin(), N, J.begin());
// 	//thrust::copy_n(A.values.begin(), N, V.begin());
	
// 	//sort by rows
// 	// thrust::sort_by_key(I.begin(), I.begin() + N,
// 	// 					thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())) );

// 	// thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
//  //                          thrust::make_zip_iterator(thrust::make_tuple(I.begin() + N, J.begin() + N)),
//  //                          V.begin(),
//  //                          thrust::make_zip_iterator(thrust::make_tuple(T.begin(), A.column_indices.begin())),
//  //                          A.values.begin(),
//  //                          thrust::equal_to< thrust::tuple<INDEX_TYPE,INDEX_TYPE> >(),
//  //                          thrust::plus<VALUE_TYPE>());

// 	// safeSync();
// 	fprintf(stderr, "A:(%d %d)   B:(%d %d)\n", A.num_rows, A.num_cols, B.num_rows, B.num_cols);

// 	// cusp::array1d<INDEX_TYPE, cusp::device_memory> T_offsets(A.num_rows + 1);
// 	// cusp::detail::indices_to_offsets(T, T_offsets);
	
// 	// fprintf(stderr, "A:(%d %d)   B:(%d %d)\n", A.num_rows, A.num_cols, B.num_rows, B.num_cols);

// 	// const size_t MAX_BLOCKS_B = cusp::system::cuda::detail::max_active_blocks(SetRowData<INDEX_TYPE, VALUE_TYPE, BINS>, BLOCK_SIZE_B, (size_t) 0);
// 	// const size_t NUM_BLOCKS_B = std::min<size_t>(MAX_BLOCKS_B, ROUND_UP(A.num_rows, BLOCK_SIZE_B));

// 	// SetRowData<INDEX_TYPE, VALUE_TYPE, BINS> <<<NUM_BLOCKS_B, BLOCK_SIZE_B>>> (
// 	// 		infoA.num_rows,
// 	// 		infoA.pitch,
// 	// 		TPC(&A.row_offsets[0]),
// 	// 		TPC(&A.row_sizes[0]),
// 	// 		TPC(&T_offsets[0]));
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS>
// void add(	dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &A,
// 			const dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &B)
// {
// 	mat_info<INDEX_TYPE> infoA, infoB;

// 	get_matrix_info(A, infoA);
// 	get_matrix_info(B, infoB);

// 	assert(A.num_rows == B.num_rows);
// 	assert(A.num_cols == B.num_cols);

// 	// const size_t BLOCK_SIZE = 256;
// 	// const size_t THREADS_PER_VECTOR = 16;
// 	// const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
// }

template <typename INDEX_TYPE, typename VALUE_TYPE>
void count_sorted_indices(	const cusp::array1d<INDEX_TYPE, cusp::device_memory> indices,
                   			cusp::array1d<INDEX_TYPE, cusp::device_memory> count,
                   			const int size)
{
    const INDEX_TYPE * I = thrust::raw_pointer_cast(&indices[0]);
    const INDEX_TYPE * X = thrust::raw_pointer_cast(&count[0]);

    if(size == 0)
    {
        // empty matrix
        return;
    }
    else if(size < static_cast<size_t>(WARP_SIZE))
    {
        // small matrix
        count_sorted_indices_serial_kernel<INDEX_TYPE,VALUE_TYPE> <<<1,1>>> (size, I, X);
        return;
    }

    const unsigned int BLOCK_SIZE = 256;
    const unsigned int MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(count_sorted_indices_kernel<INDEX_TYPE, VALUE_TYPE, BLOCK_SIZE>, BLOCK_SIZE, (size_t) 0);
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    const unsigned int num_units  = size / WARP_SIZE;
    const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
    const unsigned int num_blocks = cusp::system::cuda::DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
    const unsigned int num_iters  = cusp::system::cuda::DIVIDE_INTO(num_units, num_warps);
    
    const unsigned int interval_size = WARP_SIZE * num_iters;

    const INDEX_TYPE tail = num_units * WARP_SIZE; // do the last few nonzeros separately (fewer than WARP_SIZE elements)

    const unsigned int active_warps = (interval_size == 0) ? 0 : cusp::system::cuda::DIVIDE_INTO(tail, interval_size);

    cusp::array1d<INDEX_TYPE,cusp::device_memory> temp_rows(active_warps);
    cusp::array1d<VALUE_TYPE,cusp::device_memory> temp_vals(active_warps);

    count_sorted_indices_kernel<INDEX_TYPE, VALUE_TYPE, BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE>>>
        (tail, interval_size, I, X,
         thrust::raw_pointer_cast(&temp_rows[0]), thrust::raw_pointer_cast(&temp_vals[0]));

    count_sorted_indices_update_kernel<INDEX_TYPE, VALUE_TYPE, BLOCK_SIZE> <<<1, BLOCK_SIZE>>>
        (active_warps, thrust::raw_pointer_cast(&temp_rows[0]), thrust::raw_pointer_cast(&temp_vals[0]), X);
    
    count_sorted_indices_serial_kernel<INDEX_TYPE,VALUE_TYPE> <<<1,1>>>
        (size - tail, I + tail, X);
}

} //namespace device

#endif
