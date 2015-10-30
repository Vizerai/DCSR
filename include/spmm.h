#ifndef SPMM_H
#define SPMM_H

#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include "spmm.inl"

namespace device
{

// size_t GetWorkspaceCapacity(size_t workspace_capacity)
// {
// 	// TODO abstract this
// 	size_t free, total;
// 	cudaMemGetInfo(&free, &total);

// 	// divide free bytes by the size of each workspace unit
// 	size_t max_workspace_capacity = free / (4 * sizeof(IndexType) + sizeof(ValueType));

// 	// use at most one third of the remaining capacity
// 	return thrust::min<size_t>(max_workspace_capacity / 3, workspace_capacity);
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void spmm(  const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &A,
            const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &B,
            dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &C,
            const cudaStream_t *streams)
{
	mat_info<INDEX_TYPE> infoA, infoB, infoC;
	get_matrix_info(A, infoA);
	get_matrix_info(B, infoB);
	
	//check matrix dimensions
	assert(infoA.num_cols == infoB.num_rows);

	C.resize(A.num_rows, B.num_cols, infoA.bin_length, 8);
	//Initialize_Matrix(C);
	get_matrix_info(C, infoC);

	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_rows(A.num_rows, 0);
	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_sizes(A.num_rows, 0);
	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_bins(A.num_rows, 0);
	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_bin_offsets(10, 0);

	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
	const size_t THREADS_PER_VECTOR = VECTOR_SIZE;
	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(CalcSortSizes_SPMM<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(infoA.num_rows, VECTORS_PER_BLOCK));

	//calculate sizes and populate row list
	CalcSortSizes_SPMM<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoA.num_rows,
			infoA.num_cols,
			infoB.num_cols,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.row_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&B.row_sizes[0]),
         	TPC(&MM_sizes[0]));

	const size_t MAX_BLOCKS2 = cusp::system::cuda::detail::max_active_blocks(SetBins_SPMM<INDEX_TYPE, VALUE_TYPE>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS2 = std::min<size_t>(MAX_BLOCKS2, ROUND_UP(infoA.num_rows, BLOCK_SIZE));

	SetBins_SPMM<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS2, BLOCK_SIZE>>> (
			A.num_rows, 
			TPC(&MM_bins[0]),
			TPC(&MM_rows[0]),
			TPC(&MM_sizes[0]));

	thrust::sort_by_key(MM_bins.begin(), MM_bins.end(),
							thrust::make_zip_iterator(thrust::make_tuple(MM_sizes.begin(), MM_rows.begin())) );
	cusp::indices_to_offsets(MM_bins, MM_bin_offsets);

	// //DEBUG//
	//cusp::print(MM_bin_offsets);
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> rows = MM_rows;
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> bins = MM_bins;
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> sizes = MM_sizes;

	// for(int i=0; i<rows.size(); i++)
	// {
	// 	fprintf(stderr, "%d : (%d %d)\n", bins[i], rows[i], sizes[i]);
	// }
	// //DEBUG//

	const size_t BLOCK_SIZE_L = 128;
	const size_t THREADS_PER_VECTOR2 = 32;
	const size_t VECTORS_PER_BLOCK2 = BLOCK_SIZE / THREADS_PER_VECTOR2;
	const size_t VECTORS_PER_BLOCK_L = BLOCK_SIZE_L / THREADS_PER_VECTOR2;

	const size_t MAX_BLOCKS_MM1 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 32>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM1 = std::min<size_t>(MAX_BLOCKS_MM1, ROUND_UP(infoA.num_rows, VECTORS_PER_BLOCK2));
	const size_t MAX_BLOCKS_MM2 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 64>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM2 = std::min<size_t>(MAX_BLOCKS_MM2, ROUND_UP(infoA.num_rows, VECTORS_PER_BLOCK2));
	const size_t MAX_BLOCKS_MM3 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 128>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM3 = std::min<size_t>(MAX_BLOCKS_MM3, ROUND_UP(infoA.num_rows, VECTORS_PER_BLOCK2));
	const size_t MAX_BLOCKS_MM4 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 256>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM4 = std::min<size_t>(MAX_BLOCKS_MM4, ROUND_UP(infoA.num_rows, VECTORS_PER_BLOCK2));
	const size_t MAX_BLOCKS_MM5 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 512>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM5 = std::min<size_t>(MAX_BLOCKS_MM5, ROUND_UP(infoA.num_rows, VECTORS_PER_BLOCK2));
	const size_t MAX_BLOCKS_MM6 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_L, THREADS_PER_VECTOR2, 1024>, BLOCK_SIZE_L, (size_t) 0);
	const size_t NUM_BLOCKS_MM6 = std::min<size_t>(MAX_BLOCKS_MM6, ROUND_UP(infoA.num_rows, VECTORS_PER_BLOCK_L));

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 32> <<<NUM_BLOCKS_MM1, BLOCK_SIZE, 0, streams[0]>>> (
			infoA.num_rows,
			infoA.num_cols,
			infoB.num_cols,
			infoA.pitch,
			infoB.pitch,
			infoC.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.row_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&B.row_offsets[0]),
			TPC(&B.row_sizes[0]),
			TPC(&(*B.column_indices)[0]),
			TPC(&(*B.values)[0]),
			TPC(&MM_rows[0]),
			TPC(&MM_sizes[0]),
			TPC(&MM_bin_offsets[0]),
			TPC(&C.row_offsets[0]),
			TPC(&C.row_sizes[0]),
			TPC(&(*C.column_indices)[0]),
			TPC(&(*C.values)[0]));

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 64> <<<NUM_BLOCKS_MM2, BLOCK_SIZE, 0, streams[1]>>> (
			infoA.num_rows,
			infoA.num_cols,
			infoB.num_cols,
			infoA.pitch,
			infoB.pitch,
			infoC.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.row_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&B.row_offsets[0]),
			TPC(&B.row_sizes[0]),
			TPC(&(*B.column_indices)[0]),
			TPC(&(*B.values)[0]),
			TPC(&MM_rows[0]),
			TPC(&MM_sizes[0]),
			TPC(&MM_bin_offsets[0]),
			TPC(&C.row_offsets[0]),
			TPC(&C.row_sizes[0]),
			TPC(&(*C.column_indices)[0]),
			TPC(&(*C.values)[0]));

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 128> <<<1, BLOCK_SIZE, 0, streams[2]>>> (
			infoA.num_rows,
			infoA.num_cols,
			infoB.num_cols,
			infoA.pitch,
			infoB.pitch,
			infoC.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.row_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&B.row_offsets[0]),
			TPC(&B.row_sizes[0]),
			TPC(&(*B.column_indices)[0]),
			TPC(&(*B.values)[0]),
			TPC(&MM_rows[0]),
			TPC(&MM_sizes[0]),
			TPC(&MM_bin_offsets[0]),
			TPC(&C.row_offsets[0]),
			TPC(&C.row_sizes[0]),
			TPC(&(*C.column_indices)[0]),
			TPC(&(*C.values)[0]));

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 256> <<<NUM_BLOCKS_MM4, BLOCK_SIZE, 0, streams[3]>>> (
			infoA.num_rows,
			infoA.num_cols,
			infoB.num_cols,
			infoA.pitch,
			infoB.pitch,
			infoC.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.row_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&B.row_offsets[0]),
			TPC(&B.row_sizes[0]),
			TPC(&(*B.column_indices)[0]),
			TPC(&(*B.values)[0]),
			TPC(&MM_rows[0]),
			TPC(&MM_sizes[0]),
			TPC(&MM_bin_offsets[0]),
			TPC(&C.row_offsets[0]),
			TPC(&C.row_sizes[0]),
			TPC(&(*C.column_indices)[0]),
			TPC(&(*C.values)[0]));

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 512> <<<NUM_BLOCKS_MM5, BLOCK_SIZE, 0, streams[4]>>> (
			infoA.num_rows,
			infoA.num_cols,
			infoB.num_cols,
			infoA.pitch,
			infoB.pitch,
			infoC.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.row_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&(*A.values)[0]),
			TPC(&B.row_offsets[0]),
			TPC(&B.row_sizes[0]),
			TPC(&(*B.column_indices)[0]),
			TPC(&(*B.values)[0]),
			TPC(&MM_rows[0]),
			TPC(&MM_sizes[0]),
			TPC(&MM_bin_offsets[0]),
			TPC(&C.row_offsets[0]),
			TPC(&C.row_sizes[0]),
			TPC(&(*C.column_indices)[0]),
			TPC(&(*C.values)[0]));

	// spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 1024> <<<NUM_BLOCKS_MM6, BLOCK_SIZE, 0, streams[5]>>> (
	// 		infoA.num_rows,
	// 		infoA.num_cols,
	// 		infoB.num_cols,
	// 		infoA.pitch,
	// 		infoB.pitch,
	// 		infoC.pitch,
	// 		TPC(&A.row_offsets[0]),
	// 		TPC(&A.row_sizes[0]),
	// 		TPC(&(*A.column_indices)[0]),
	// 		TPC(&(*A.values)[0]),
	// 		TPC(&B.row_offsets[0]),
	// 		TPC(&B.row_sizes[0]),
	// 		TPC(&(*B.column_indices)[0]),
	// 		TPC(&(*B.values)[0]),
	// 		TPC(&MM_rows[0]),
	// 		TPC(&MM_sizes[0]),
	// 		TPC(&MM_bin_offsets[0]),
	// 		TPC(&C.row_offsets[0]),
	// 		TPC(&C.row_sizes[0]),
	// 		TPC(&(*C.column_indices)[0]),
	// 		TPC(&(*C.values)[0]));

	//cusp::print((*C.values));
}

} //namespace device

#endif