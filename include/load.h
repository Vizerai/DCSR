#ifndef LOAD_H
#define LOAD_H

namespace device
{

//initialize DCSR matrix
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void Initialize_Matrix(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

	InitializeMatrix_dcsr<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.bin_length,
			TPC(&mat.row_sizes[0]),
			TPC(&mat.row_offsets[0]));
}

//*******************************************************************************************//
//Fill matrices from a COO matrix
//*******************************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void LoadMatrix(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &dst,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoDst;
	get_matrix_info(dst, infoDst);

	// cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_rows(N);
	// thrust::copy(rows.begin(), rows.begin() + N, temp_rows.begin());
	thrust::copy(cols.begin(), cols.begin() + N, dst.column_indices->begin());
	thrust::copy(vals.begin(), vals.begin() + N, dst.values->begin());

	// thrust::sort_by_key(temp_rows.begin(), temp_rows.begin()+N,
			// 		thrust::make_zip_iterator(thrust::make_tuple(dst.column_indices.begin(), dst.values.begin())) );

	cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_offsets(dst.num_rows + 1);
	cusp::indices_to_offsets(rows, temp_offsets);

	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
	const size_t MAX_BLOCKS_B = cusp::system::cuda::detail::max_active_blocks(SetRowData<INDEX_TYPE, VALUE_TYPE, BINS>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_B = std::min<size_t>(MAX_BLOCKS_B, ROUND_UP(infoDst.num_rows, BLOCK_SIZE));

	//set new row offset values
	SetRowData<INDEX_TYPE, VALUE_TYPE, BINS> <<<NUM_BLOCKS_B, BLOCK_SIZE>>> (
		infoDst.num_rows,
		infoDst.pitch,
		TPC(&dst.row_offsets[0]),
		TPC(&dst.row_sizes[0]),
		TPC(&temp_offsets[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

#if(DEBUG)
	assert(src.num_rows == infoDst.num_rows);
	assert(src.num_cols == infoDst.num_cols);
#endif

   cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_offsets(mat.num_rows + 1);
   //cusp::array1d<INDEX_TYPE, cusp::device_memory> ovf_offsets(mat.num_rows + 1);
   cusp::indices_to_offsets(rows, temp_offsets);


	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
	const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(LoadMatrix_hyb_coo<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK));

	LoadMatrix_hyb_coo<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			infoMat.num_cols_per_row,
			infoMat.pitch,
			TPC(&rows[0]),
			TPC(&cols[0]),
			TPC(&vals[0]),
			TPC(&temp_offsets[0]),
			TPC(&mat.row_sizes[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.ell.values.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]),
			TPC(&mat.matrix.coo.values[0]));

	mat.num_overflow = mat.row_sizes[infoMat.num_rows];
	thrust::sort_by_key(mat.matrix.coo.row_indices.begin(), mat.matrix.coo.row_indices.begin()+mat.num_overflow,
						thrust::make_zip_iterator(thrust::make_tuple(mat.matrix.coo.column_indices.begin(), mat.matrix.coo.values.begin())) );
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	//cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_rows(N);
	//thrust::copy(rows.begin(), rows.end(), temp_rows.begin());
	thrust::copy(cols.begin(), cols.end(), mat.column_indices.begin());
	thrust::copy(vals.begin(), vals.end(), mat.values.begin());

	//remove this for now to check timings....   assume sorted input
	//thrust::sort_by_key(temp_rows.begin(), temp_rows.begin()+N,
	//					thrust::make_zip_iterator(thrust::make_tuple(mat.column_indices.begin(), mat.values.begin())) );

	cusp::indices_to_offsets(rows, mat.row_offsets);
}

//*******************************************************************************************//
//Load matrices from a CSR matrix
//*******************************************************************************************//
// template <typename INDEX_TYPE, typename VALUE_TYPE>
// void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
// 					cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
// {
// 	dst.resize(src.num_rows, src.num_cols, src.num_entries, std::max(src.num_cols/16, ulong(64)));

// 	mat_info<INDEX_TYPE> infoDst;
// 	get_matrix_info(dst, infoDst);

// #if(DEBUG)
// 	assert(src.num_rows == infoDst.num_rows);
// 	assert(src.num_cols == infoDst.num_cols);
// #endif

// 	const size_t NUM_BLOCKS = BLOCKS;
// 	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;

// 	LoadEllMatrix<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
// 			src.num_rows,
// 			src.num_entries,
// 			infoDst.num_cols_per_row,
// 			infoDst.pitch,
// 			TPC(&src.row_offsets[0]),
// 			TPC(&src.column_indices[0]),
// 			TPC(&src.values[0]),
// 			TPC(&dst.column_indices.values[0]),
// 			TPC(&dst.values.values[0]));
// }

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src)
{
	mat_info<INDEX_TYPE> infoDst;
	get_matrix_info(dst, infoDst);

	thrust::copy(src.column_indices.begin(), src.column_indices.end(), dst.column_indices.begin());
	thrust::copy(src.values.begin(), src.values.end(), dst.values.begin());
	thrust::copy(src.row_offsets.begin(), src.row_offsets.end(), dst.row_offsets.begin());
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src)
{
	mat_info<INDEX_TYPE> infoDst;
	get_matrix_info(dst, infoDst);

	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
	const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
	const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;
	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(LoadMatrix_hyb_csr<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(infoDst.num_rows, VECTORS_PER_BLOCK));

	LoadMatrix_hyb_csr<INDEX_TYPE, VALUE_TYPE, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoDst.num_rows,
			infoDst.num_cols,
			infoDst.num_cols_per_row,
			infoDst.pitch,
			TPC(&src.column_indices[0]),
			TPC(&src.values[0]),
			TPC(&src.row_offsets[0]),
			TPC(&dst.row_sizes[0]),
			TPC(&dst.matrix.ell.column_indices.values[0]),
			TPC(&dst.matrix.ell.values.values[0]),
			TPC(&dst.matrix.coo.row_indices[0]),
			TPC(&dst.matrix.coo.column_indices[0]),
			TPC(&dst.matrix.coo.values[0]));

	dst.num_overflow = dst.row_sizes[infoDst.num_rows];
	thrust::sort_by_key(dst.matrix.coo.row_indices.begin(), dst.matrix.coo.row_indices.begin()+dst.num_overflow,
						thrust::make_zip_iterator(thrust::make_tuple(dst.matrix.coo.column_indices.begin(), dst.matrix.coo.values.begin())) );
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void LoadMatrix(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src)
{
	mat_info<INDEX_TYPE> infoDst;
	get_matrix_info(dst, infoDst);

	thrust::copy(src.column_indices.begin(), src.column_indices.end(), dst.column_indices->begin());
	thrust::copy(src.values.begin(), src.values.end(), dst.values->begin());

	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(SetRowData<INDEX_TYPE, VALUE_TYPE, BINS>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(infoDst.num_rows, BLOCK_SIZE));

	SetRowData<INDEX_TYPE, VALUE_TYPE, BINS> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
		infoDst.num_rows,
		infoDst.pitch,
		TPC(&dst.row_offsets[0]),
		TPC(&dst.row_sizes[0]),
		TPC(&src.row_offsets[0]));
}

//*******************************************************************************************//
//Load matrices from a DCSR matrix
//*******************************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &src)
{
	mat_info<INDEX_TYPE> infoDst;
	get_matrix_info(dst, infoDst);

	INDEX_TYPE size = src.row_sizes[infoDst.num_rows];
	//fprintf(stderr, "size: %d, (%d %d)\n", size, dst.column_indices.size(), dst.values.size());
	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(DCSRtoCSROffsets<INDEX_TYPE, VALUE_TYPE, BINS>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(infoDst.num_rows, BLOCK_SIZE));

	thrust::copy(src.column_indices->begin(), src.column_indices->begin() + size, dst.column_indices.begin());
    thrust::copy(src.values->begin(), src.values->begin() + size, dst.values.begin());

	DCSRtoCSROffsets<INDEX_TYPE, VALUE_TYPE, BINS> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoDst.num_rows,
			infoDst.pitch,
			TPC(&src.row_offsets[0]),
			TPC(&dst.row_offsets[0]));
}

//*******************************************************************************************//
//Fill matrices from a COO format
//*******************************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void UpdateMatrix(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const bool permuted = false)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	cusp::array1d<INDEX_TYPE, cusp::device_memory> T_offsets(mat.num_rows + 1);
	cusp::indices_to_offsets(rows, T_offsets);

	if(!permuted)
	{
		const size_t BLOCK_SIZE = 128;
		const size_t THREADS_PER_VECTOR = __VECTOR_SIZE;
		const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

		const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(UpdateMatrix_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
		const size_t NUM_BLOCKS = min(int(MAX_BLOCKS), int(ROUND_UP(infoMat.num_rows, VECTORS_PER_BLOCK)));

		UpdateMatrix_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
				infoMat.num_rows,
				infoMat.pitch,
				0,					//alpha of 0
				TPC(&rows[0]),
				TPC(&cols[0]),
				TPC(&vals[0]),
				TPC(&T_offsets[0]),
				TPC(&(*mat.column_indices)[0]),
				TPC(&(*mat.values)[0]),
				TPC(&mat.row_offsets[0]),
				TPC(&mat.row_sizes[0]));
	}
	else
	{
		const size_t BLOCK_SIZE = 128;
		const size_t THREADS_PER_VECTOR = 1;
		const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / THREADS_PER_VECTOR;

		const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(UpdateMatrix_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, BLOCK_SIZE, (size_t) 0);
		const size_t NUM_BLOCKS = min(int(MAX_BLOCKS), int(ROUND_UP(infoMat.num_rows, VECTORS_PER_BLOCK)));

		UpdateMatrix_dcsr<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
				infoMat.num_rows,
				infoMat.pitch,
				4,
				TPC(&rows[0]),
				TPC(&cols[0]),
				TPC(&vals[0]),
				TPC(&T_offsets[0]),
				TPC(&(*mat.column_indices)[0]),
				TPC(&(*mat.values)[0]),
				TPC(&mat.row_offsets[0]),
				TPC(&mat.row_sizes[0]));
	}

	// safeSync();
	// cudaPrintfDisplay(stdout, true);

	// int mem_pos = mat.row_sizes[infoMat.num_rows];
	// fprintf(stderr, "mem_pos: %d\n", mem_pos);
	// if(mem_pos >= int(mat.mem_size))
	// {	
	// 	fprintf(stderr, "***ERROR*** memory allocation overflowed!\n");
	// }
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void UpdateMatrix(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
					const int N)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	cusp::array1d<INDEX_TYPE, cusp::device_memory> T_offsets(mat.num_rows + 1);
	cusp::indices_to_offsets(rows, T_offsets);

	const size_t BLOCK_SIZE = BLOCK_THREAD_SIZE;
	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(UpdateMatrix_hyb<INDEX_TYPE, VALUE_TYPE>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS = min(int(MAX_BLOCKS), int(ROUND_UP(infoMat.num_rows, BLOCK_SIZE)));

	UpdateMatrix_hyb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.num_cols,
			infoMat.num_cols_per_row,
			infoMat.pitch,
			TPC(&rows[0]),
			TPC(&cols[0]),
			TPC(&vals[0]),
			TPC(&T_offsets[0]),
			TPC(&mat.row_sizes[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.ell.values.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]),
			TPC(&mat.matrix.coo.values[0]));

	mat.num_overflow = mat.row_sizes[mat.num_rows];
	//fprintf(stderr, "num_overflow: %d  coo size: %d\n", mat.num_overflow, mat.matrix.coo.row_indices.size());
	//double stime = omp_get_wtime();
	thrust::sort_by_key(mat.matrix.coo.row_indices.begin(), mat.matrix.coo.row_indices.begin()+mat.num_overflow,
						thrust::make_zip_iterator(thrust::make_tuple(mat.matrix.coo.column_indices.begin(), mat.matrix.coo.values.begin())) );
	//double etime = omp_get_wtime();
	//fprintf(stderr, "sort time %f\n", (etime - stime));

	mat.matrix.num_entries += N;
	mat.matrix.ell.num_entries = mat.matrix.num_entries - mat.num_overflow;
	mat.matrix.coo.num_entries = mat.num_overflow;

	//fprintf(stderr, "mat.matrix.num_entries: %d\n", mat.matrix.num_entries);
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
// void UpdateMatrix(	dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat,
// 					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
// 					const cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
// 					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals,
// 					const int N)
// {
// 	mat_info<INDEX_TYPE> infoMat;
// 	get_matrix_info(mat, infoMat);

// 	const size_t BLOCK_SIZE = 128;
// 	const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(UpdateMatrix_dcsr<INDEX_TYPE, VALUE_TYPE>, BLOCK_SIZE, (size_t) 0);
// 	const size_t NUM_BLOCKS = min(int(MAX_BLOCKS), int(ROUND_UP(infoMat.num_rows, BLOCK_SIZE)));

// 	UpdateMatrix_dell<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
// 			infoMat.num_rows,
// 			infoMat.pitch,
// 			TPC(&rows[0]),
// 			TPC(&cols[0]),
// 			TPC(&vals[0]),
// 			N,
// 			TPC(&mat.ell.column_indices.values[0]),
// 			TPC(&mat.ell.values.values[0]),
// 			TPC(&mat.column_indices[0]),
// 			TPC(&mat.values[0]),
// 			TPC(&mat.row_offsets[0]),
// 			TPC(&mat.row_sizes[0]));
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void SortMatrixRow(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat,
					const cudaStream_t *streams)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	//adjust permuted row sizes to mactch actual row sizes
	thrust::gather(mat.row_ids.begin(), mat.row_ids.end(), mat.row_sizes.begin(), mat.Prow_sizes.begin());

	//perform scan on row sizes to determine new indices
	cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_offsets(infoMat.num_rows + 1);
	//cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_offsets2(infoMat.num_rows + 1);
	//thrust::exclusive_scan(mat.row_sizes.begin(), mat.row_sizes.end(), temp_offsets2.begin());
	thrust::exclusive_scan(mat.Prow_sizes.begin(), mat.Prow_sizes.end(), temp_offsets.begin());
	
	//shallow copy <- this is 10% - 20% faster than a deep copy
	INDEX_TYPE size = (*mat.column_indices).size();
	cusp::array1d<INDEX_TYPE, cusp::device_memory> *T_cols = new cusp::array1d<INDEX_TYPE, cusp::device_memory>(size);
	cusp::array1d<VALUE_TYPE, cusp::device_memory> *T_vals = new cusp::array1d<VALUE_TYPE, cusp::device_memory>(size);

	//deep copy
	//INDEX_TYPE size = temp_offsets[infoMat.num_rows];
	//cusp::array1d<INDEX_TYPE, cusp::device_memory> T_cols(size);
	//cusp::array1d<VALUE_TYPE, cusp::device_memory> T_vals(size);

	// INDEX_TYPE N = mat.row_sizes[infoMat.num_rows];
	// fprintf(stderr, "Sorting DCSR matrix:  size: %d  N: %d\n", size, N);

	const size_t BLOCK_SIZE = 128;
	// const size_t VECTORS_PER_BLOCK = BLOCK_SIZE / __VECTOR_SIZE;
	// const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, __VECTOR_SIZE>, BLOCK_SIZE, (size_t) 0);
	// const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK));

	// CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, __VECTOR_SIZE> <<<NUM_BLOCKS, BLOCK_SIZE>>> (
	// 		infoMat.num_rows,
	// 		infoMat.pitch,
	// 		TPC(&T_cols[0]),
	// 		TPC(&T_vals[0]),
	// 		TPC(&temp_offsets[0]),
	// 		TPC(&mat.column_indices[0]),
	// 		TPC(&mat.values[0]),
	// 		TPC(&mat.row_sizes[0]),
	// 		TPC(&mat.row_offsets[0]));

	const size_t VECTORS_PER_BLOCK_A1 = BLOCK_SIZE / 1;
	const size_t VECTORS_PER_BLOCK_A2 = BLOCK_SIZE / 2;
	const size_t VECTORS_PER_BLOCK_A4 = BLOCK_SIZE / 4;
	const size_t VECTORS_PER_BLOCK_A8 = BLOCK_SIZE / 8;
	const size_t VECTORS_PER_BLOCK_A16 = BLOCK_SIZE / 16;
	const size_t VECTORS_PER_BLOCK_A32 = BLOCK_SIZE / 32;

	const size_t MAX_BLOCKS_A1 = cusp::system::cuda::detail::max_active_blocks(CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A1, 1>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_A1 = std::min<size_t>(MAX_BLOCKS_A1, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK_A1));
	const size_t MAX_BLOCKS_A2 = cusp::system::cuda::detail::max_active_blocks(CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A2, 2>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_A2 = std::min<size_t>(MAX_BLOCKS_A2, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK_A2));
	const size_t MAX_BLOCKS_A4 = cusp::system::cuda::detail::max_active_blocks(CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A4, 4>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_A4 = std::min<size_t>(MAX_BLOCKS_A4, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK_A4));
	const size_t MAX_BLOCKS_A8 = cusp::system::cuda::detail::max_active_blocks(CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A8, 8>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_A8 = std::min<size_t>(MAX_BLOCKS_A8, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK_A8));
	const size_t MAX_BLOCKS_A16 = cusp::system::cuda::detail::max_active_blocks(CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A16, 16>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_A16 = std::min<size_t>(MAX_BLOCKS_A16, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK_A16));
	const size_t MAX_BLOCKS_A32 = cusp::system::cuda::detail::max_active_blocks(CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A32, 32>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_A32 = std::min<size_t>(MAX_BLOCKS_A32, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK_A32));

	//set row indices
	CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A1, 1> <<<NUM_BLOCKS_A1, BLOCK_SIZE, 0, streams[0]>>> (
			infoMat.num_rows,
			infoMat.pitch,
			TPC(&(*T_cols)[0]),
			TPC(&(*T_vals)[0]),
			TPC(&temp_offsets[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&(*mat.values)[0]),
			TPC(&mat.Prow_sizes[0]),
			TPC(&mat.row_ids[0]),
			TPC(&mat.row_offsets[0]),
			TPC(&mat.row_sizes[0]));

	CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A2, 2> <<<NUM_BLOCKS_A2, BLOCK_SIZE, 0, streams[1]>>> (
			infoMat.num_rows,
			infoMat.pitch,
			TPC(&(*T_cols)[0]),
			TPC(&(*T_vals)[0]),
			TPC(&temp_offsets[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&(*mat.values)[0]),
			TPC(&mat.Prow_sizes[0]),
			TPC(&mat.row_ids[0]),
			TPC(&mat.row_offsets[0]),
			TPC(&mat.row_sizes[0]));

	CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A4, 4> <<<NUM_BLOCKS_A4, BLOCK_SIZE, 0, streams[2]>>> (
			infoMat.num_rows,
			infoMat.pitch,
			TPC(&(*T_cols)[0]),
			TPC(&(*T_vals)[0]),
			TPC(&temp_offsets[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&(*mat.values)[0]),
			TPC(&mat.Prow_sizes[0]),
			TPC(&mat.row_ids[0]),
			TPC(&mat.row_offsets[0]),
			TPC(&mat.row_sizes[0]));

	CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A8, 8> <<<NUM_BLOCKS_A8, BLOCK_SIZE, 0, streams[3]>>> (
			infoMat.num_rows,
			infoMat.pitch,
			TPC(&(*T_cols)[0]),
			TPC(&(*T_vals)[0]),
			TPC(&temp_offsets[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&(*mat.values)[0]),
			TPC(&mat.Prow_sizes[0]),
			TPC(&mat.row_ids[0]),
			TPC(&mat.row_offsets[0]),
			TPC(&mat.row_sizes[0]));

	CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A16, 16> <<<NUM_BLOCKS_A16, BLOCK_SIZE, 0, streams[4]>>> (
			infoMat.num_rows,
			infoMat.pitch,
			TPC(&(*T_cols)[0]),
			TPC(&(*T_vals)[0]),
			TPC(&temp_offsets[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&(*mat.values)[0]),
			TPC(&mat.Prow_sizes[0]),
			TPC(&mat.row_ids[0]),
			TPC(&mat.row_offsets[0]),
			TPC(&mat.row_sizes[0]));

	CompactIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A32, 32> <<<NUM_BLOCKS_A32, BLOCK_SIZE, 0, streams[5]>>> (
			infoMat.num_rows,
			infoMat.pitch,
			TPC(&(*T_cols)[0]),
			TPC(&(*T_vals)[0]),
			TPC(&temp_offsets[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&(*mat.values)[0]),
			TPC(&mat.Prow_sizes[0]),
			TPC(&mat.row_ids[0]),
			TPC(&mat.row_offsets[0]),
			TPC(&mat.row_sizes[0]));

	safeSync();
	//cudaPrintfDisplay(stdout, true);

	//shallow copy
	SAFE_DELETE(mat.column_indices);
	SAFE_DELETE(mat.values);
	mat.column_indices = T_cols;
	mat.values = T_vals;

	//deep copy
	// thrust::copy(T_cols.begin(), T_cols.begin() + size, mat.column_indices.begin());
	// thrust::copy(T_vals.begin(), T_vals.begin() + size, mat.values.begin());

	const size_t MAX_BLOCKS_B = cusp::system::cuda::detail::max_active_blocks(SetOffsets<INDEX_TYPE, VALUE_TYPE, BINS>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_B = std::min<size_t>(MAX_BLOCKS_B, ROUND_UP(mat.num_rows, BLOCK_SIZE));

	/*NON Permuted version*/
	//set new row offset values
	// SetOffsets<INDEX_TYPE, VALUE_TYPE, BINS> <<<NUM_BLOCKS_B, BLOCK_SIZE>>> (
	// 		infoMat.num_rows,
	// 		infoMat.pitch,
	// 		TPC(&mat.row_offsets[0]),
	// 		TPC(&mat.row_sizes[0]),
	// 		TPC(&temp_offsets[0]));

	/*Permuted version*/
	SetOffsets<INDEX_TYPE, VALUE_TYPE, BINS> <<<NUM_BLOCKS_B, BLOCK_SIZE>>> (
			infoMat.num_rows,
			infoMat.pitch,
			TPC(&mat.row_offsets[0]),
			TPC(&mat.row_sizes[0]),
			TPC(&mat.row_ids[0]),
			TPC(&temp_offsets[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void BinRows(dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat)
{
	mat_info<INDEX_TYPE> infoMat;
	get_matrix_info(mat, infoMat);

	cusp::array1d<INDEX_TYPE, cusp::device_memory> bins(infoMat.num_rows);

	const size_t BLOCK_SIZE = 128;
	const size_t MAX_BLOCKS_C = cusp::system::cuda::detail::max_active_blocks(SetBins<INDEX_TYPE, VALUE_TYPE>, BLOCK_SIZE, (size_t) 0);
	const size_t NUM_BLOCKS_C = std::min<size_t>(MAX_BLOCKS_C, ROUND_UP(mat.num_rows, BLOCK_SIZE));

	SetBins<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS_C, BLOCK_SIZE>>> (
			infoMat.num_rows,
			TPC(&mat.bins[0]),
			TPC(&mat.row_ids[0]),
			TPC(&mat.row_sizes[0]));

	thrust::copy(mat.row_sizes.begin(), mat.row_sizes.begin() + infoMat.num_rows, mat.Prow_sizes.begin());
	thrust::sort_by_key(mat.bins.begin(), mat.bins.end(), 
						thrust::make_zip_iterator(thrust::make_tuple(mat.row_ids.begin(), mat.Prow_sizes.begin())) );
	cusp::indices_to_offsets(mat.bins, mat.bin_offsets);

	cudaMemcpyToSymbol(c_bin_offsets, TPC(&mat.bin_offsets[0]), 8*sizeof(int), 0, cudaMemcpyDeviceToDevice);
	
	//mat.bin_offsets_H = mat.bin_offsets;		//not needed atm
	//cusp::print(mat.bin_offsets);
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
// void SortMatrixRowCol(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat)
// {
// 	mat_info<INDEX_TYPE> infoMat;
// 	get_matrix_info(mat, infoMat);

// 	const size_t BLOCK_SIZE = 128;
// 	const size_t THREADS_PER_VECTOR_A = __VECTOR_SIZE;
// 	const size_t VECTORS_PER_BLOCK_A = BLOCK_SIZE / THREADS_PER_VECTOR_A;

// 	const size_t MAX_BLOCKS_A = cusp::system::cuda::detail::max_active_blocks(SetRowIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A, THREADS_PER_VECTOR_A>, BLOCK_SIZE, (size_t) 0);
// 	const size_t NUM_BLOCKS_A = std::min<size_t>(MAX_BLOCKS_A, ROUND_UP(mat.num_rows, VECTORS_PER_BLOCK_A));

// 	INDEX_TYPE N = mat.row_sizes[infoMat.num_rows];
// 	cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_rows(N);
// 	//set row indices
// 	SetRowIndices<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK_A, THREADS_PER_VECTOR_A> <<<NUM_BLOCKS_A, BLOCK_SIZE>>> (
// 			infoMat.num_rows,
// 			TPC(&temp_rows[0]),
// 			TPC(&mat.row_offsets[0]),
// 			TPC(&mat.row_sizes[0]));

// 	//sort by rows
// 	cusp::detail::sort_by_row_and_column(temp_rows, mat.column_indices, mat.values());

// 	//perform scan on row sizes to determine new indices
// 	cusp::array1d<INDEX_TYPE, cusp::device_memory> temp_offsets(mat.num_rows + 1);
// 	thrust::exclusive_scan(mat.row_sizes.begin(), mat.row_sizes.end(), temp_offsets.begin());

// 	const size_t MAX_BLOCKS_B = cusp::system::cuda::detail::max_active_blocks(SetOffsets<INDEX_TYPE, VALUE_TYPE, BINS>, BLOCK_SIZE, (size_t) 0);
// 	const size_t NUM_BLOCKS_B = std::min<size_t>(MAX_BLOCKS_B, ROUND_UP(mat.num_rows, BLOCK_SIZE));

// 	//set new row offset values
// 	SetOffsets<INDEX_TYPE, VALUE_TYPE, BINS> <<<NUM_BLOCKS_B, BLOCK_SIZE>>> (
// 			infoMat.num_rows,
// 			TPC(&mat.row_offsets[0]),
// 			TPC(&temp_offsets[0]));
// }

} //namespace device

#endif