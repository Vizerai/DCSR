#ifndef SPMM_H
#define SPMM_H

//#define CUB_CDP
#define __CUDACC_RDC__

//#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_radix_sort.cuh>
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

	int avgA = (infoA.num_entries / infoA.num_rows + 1) * 2;
	int avgB = (infoB.num_entries / infoB.num_rows + 1) * 2;
	C.resize(A.num_rows, B.num_cols, max(avgA, avgB), 20);
	//Initialize_Matrix(C);
	get_matrix_info(C, infoC);

	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_rows(A.num_rows, 0);
	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_sizes(A.num_rows, 0);
	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_bins(A.num_rows, 0);
	cusp::array1d<INDEX_TYPE, cusp::device_memory> MM_bin_offsets(12, 0);

	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
	const size_t HALF_BLOCK = BLOCK_THREAD_SIZE / 2;
	const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

	const size_t M = infoA.num_rows;
	const size_t K = infoA.num_cols;
	const size_t N = infoB.num_cols;

	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(CalcSortSizes_SPMM<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(M, VECTORS_PER_BLOCK));

	//calculate sizes and populate row list
	CalcSortSizes_SPMM<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			M,
			K,
			N,
			infoA.pitch,
			TPC(&A.row_offsets[0]),
			TPC(&A.row_sizes[0]),
			TPC(&(*A.column_indices)[0]),
			TPC(&B.row_sizes[0]),
         TPC(&MM_sizes[0]));

	const size_t MAX_BLOCKS2 = cusp::system::cuda::detail::max_active_blocks(SetBins_SPMM<INDEX_TYPE, VALUE_TYPE>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS2 = std::min<size_t>(MAX_BLOCKS2, ROUND_UP(M, BLOCK_SIZE));

	SetBins_SPMM<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS2, BLOCK_SIZE>>> (
			A.num_rows, 
			TPC(&MM_bins[0]),
			TPC(&MM_rows[0]),
			TPC(&MM_sizes[0]));

	thrust::sort_by_key(MM_bins.begin(), MM_bins.end(),
							thrust::make_zip_iterator(thrust::make_tuple(MM_sizes.begin(), MM_rows.begin())) );
	cusp::indices_to_offsets(MM_bins, MM_bin_offsets);

	//DEBUG//
	// cusp::print(MM_bin_offsets);
	// cusp::print(MM_sizes);
	// cusp::print(MM_rows);
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> sizes = A.row_sizes;
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> offsets = A.row_offsets;
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> rows = MM_rows;
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> bins = MM_bins;
	// cusp::array1d<INDEX_TYPE, cusp::host_memory> sizes = MM_sizes;

	// for(int i=0; i<A.num_rows; i++)
	// {
	// 	if(offsets[i*2] > offsets[i*2+1])
	// 		fprintf(stderr, "wrong offset: %d : (%d %d)\n", i, offsets[i*2], offsets[i*2+1]);
	// }
	// //DEBUG//

	//const size_t BLOCK_SIZE_L = 128;
	const size_t THREADS_PER_VECTOR2 = WARP_SIZE;
	#if(PRECISION == 32)
	const size_t VECTORS_PER_BLOCK2 = BLOCK_SIZE / THREADS_PER_VECTOR2;
	#elif(PRECISION == 64)
	const size_t VECTORS_PER_BLOCK2 = BLOCK_SIZE / THREADS_PER_VECTOR2;
	const size_t VECTORS_PER_BLOCK_HALF = HALF_BLOCK / THREADS_PER_VECTOR2;
	#endif
	//const size_t VECTORS_PER_BLOCK_L = BLOCK_SIZE_L / THREADS_PER_VECTOR2;

	const size_t MAX_BLOCKS_MM1 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 32>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM1 = std::min<size_t>(MAX_BLOCKS_MM1, ROUND_UP(M, VECTORS_PER_BLOCK2));
	const size_t MAX_BLOCKS_MM2 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 64>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM2 = std::min<size_t>(MAX_BLOCKS_MM2, ROUND_UP(M, VECTORS_PER_BLOCK2));
	const size_t MAX_BLOCKS_MM3 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 128>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM3 = std::min<size_t>(MAX_BLOCKS_MM3, ROUND_UP(M, VECTORS_PER_BLOCK2));

	const size_t MAX_BLOCKS_MM4 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_med_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, BLOCK_SIZE, 256/BLOCK_SIZE, 256>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM4 = std::min<size_t>(MAX_BLOCKS_MM4, ROUND_UP(M, VECTORS_PER_BLOCK2));
	#if(PRECISION == 32)
	const size_t MAX_BLOCKS_MM5 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_med_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, BLOCK_SIZE, 512/BLOCK_SIZE, 512>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM5 = std::min<size_t>(MAX_BLOCKS_MM5, ROUND_UP(M, VECTORS_PER_BLOCK2));
	#elif(PRECISION == 64)
	const size_t MAX_BLOCKS_MM5 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_med_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_HALF, HALF_BLOCK, 512/HALF_BLOCK, 512>, HALF_BLOCK, (size_t) 0);
	const size_t NUM_BLOCKS_MM5 = std::min<size_t>(MAX_BLOCKS_MM5, ROUND_UP(M, VECTORS_PER_BLOCK_HALF));
	#endif

	const size_t MAX_BLOCKS_MM6 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_large_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 1024/BLOCK_SIZE, 1024>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM6 = std::min<size_t>(MAX_BLOCKS_MM6, M);
	#if(PRECISION == 32)
	const size_t MAX_BLOCKS_MM7 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_large_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 2048/BLOCK_SIZE, 2048>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM7 = std::min<size_t>(MAX_BLOCKS_MM7, M);
	#elif(PRECISION == 64)
	//had to edit double precision version because it was over the cap of shared memory
	const size_t MAX_BLOCKS_MM7 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_large_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 1280/BLOCK_SIZE, 1280>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_MM7 = std::min<size_t>(MAX_BLOCKS_MM7, M);
	#endif

	// const size_t MAX_BLOCKS_MM8 = cusp::system::cuda::detail::max_active_blocks(spmm_dcsr_mega_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 4096/BLOCK_SIZE, 4096>, BLOCK_SIZE, (size_t) 0);
	// const size_t NUM_BLOCKS_MM8 = std::min<size_t>(MAX_BLOCKS_MM8, M);

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 32> <<<NUM_BLOCKS_MM1, BLOCK_SIZE, 0, streams[0]>>> (
			M,
			N,
			K,
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
	//safeSync();

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 64> <<<NUM_BLOCKS_MM2, BLOCK_SIZE, 0, streams[1]>>> (
			M,
			N,
			K,
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
	//safeSync();

	spmm_dcsr_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, THREADS_PER_VECTOR2, 128> <<<NUM_BLOCKS_MM3, BLOCK_SIZE, 0, streams[2]>>> (
			M,
			N,
			K,
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
	//safeSync();

	spmm_dcsr_med_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, BLOCK_SIZE, 256/BLOCK_SIZE, 256> <<<NUM_BLOCKS_MM4, BLOCK_SIZE, 0, streams[3]>>> (
			M,
			N,
			K,
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
	//safeSync();

	#if(PRECISION == 32)
	spmm_dcsr_med_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK2, BLOCK_SIZE, 512/BLOCK_SIZE, 512> <<<NUM_BLOCKS_MM5, BLOCK_SIZE, 0, streams[4]>>> (
			M,
			N,
			K,
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
	//safeSync();
	#elif(PRECISION == 64)
	spmm_dcsr_med_kernel<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_HALF, HALF_BLOCK, 512/HALF_BLOCK, 512> <<<NUM_BLOCKS_MM5, HALF_BLOCK, 0, streams[4]>>> (
			M,
			N,
			K,
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
	//safeSync();
	#endif

	spmm_dcsr_large_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 1024/BLOCK_SIZE, 1024> <<<NUM_BLOCKS_MM6, BLOCK_SIZE, 0, streams[5]>>> (
			M,
			N,
			K,
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
	//safeSync();

	#if(PRECISION == 32)
	spmm_dcsr_large_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 2048/BLOCK_SIZE, 2048> <<<NUM_BLOCKS_MM7, BLOCK_SIZE, 0, streams[6]>>> (
			M,
			N,
			K,
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
	#elif(PRECISION == 64)
	// spmm_dcsr_large_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 1280/BLOCK_SIZE, 1280> <<<NUM_BLOCKS_MM7, BLOCK_SIZE, 0, streams[6]>>> (
	// 		M,
	// 		N,
	// 		K,
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
	#endif
	//safeSync();

	// spmm_dcsr_mega_kernel<INDEX_TYPE, VALUE_TYPE, BINS, BLOCK_SIZE, 4096/BLOCK_SIZE, 4096> <<<1, BLOCK_SIZE, 0, streams[7]>>> (
	// 		M,
	// 		N,
	// 		K,
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
	//safeSync();
}

} //namespace device

#endif