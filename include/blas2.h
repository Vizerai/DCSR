template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProduct(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
					const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
					cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> info;
	get_matrix_info<VALUE_TYPE> (mat, info);

#if(DEBUG)
	assert(info.num_rows == a.size());
	assert(info.num_cols == b.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProduct<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			TPC(&a[0]),
			TPC(&b[0]),
			info.num_rows,
			info.num_cols,
			info.num_cols_per_row,
			info.pitch,
			TPC(&mat.column_indices.values[0]),
			TPC(&mat.values.values[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
						cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
						cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> info;
	get_matrix_info<VALUE_TYPE> (mat, info);

#if(DEBUG)
	assert(info.num_rows == a.size());
	assert(info.num_cols == b.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProductAdd_ELL<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&index_count[0]),
			info.num_rows,
			info.num_cols,
			info.num_cols_per_row,
			info.pitch,
			TPC(&mat.column_indices.values[0]),
			TPC(&mat.values.values[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
						hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
						cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> info;
	get_matrix_info<VALUE_TYPE> (mat, info);

#if(DEBUG)
	assert(info.num_rows == a.size());
	assert(info.num_cols == b.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProductAdd_HYB<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&index_count[0]),
			info.num_rows,
			info.num_cols,
			info.num_cols_per_row,
			info.pitch,
			TPC(&mat.row_sizes[0]),
			TPC(&mat.matrix.ell.column_indices.values[0]),
			TPC(&mat.matrix.coo.row_indices[0]),
			TPC(&mat.matrix.coo.column_indices[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
						dell_matrix_B<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
						cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> info;
	get_matrix_info<VALUE_TYPE> (mat, info);

#if(DEBUG)
	assert(info.num_rows == a.size());
	assert(info.num_cols == b.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProductAdd_DELL_B<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&index_count[0]),
			TPC(&(*mat.row_offsets)[0]),
			TPC(&(*mat.column_indices)[0]),
			TPC(&mat.row_sizes[0]),
			TPC(&mat.coo.row_indices[0]),
			TPC(&mat.coo.column_indices[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProductAdd(	const cusp::array1d<VALUE_TYPE, cusp::device_memory> &a,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &b,
						const cusp::array1d<VALUE_TYPE, cusp::device_memory> &index_count,
						cusp::array1d<INDEX_TYPE, cusp::device_memory> &queue,
						cudaStream_t &stream)
{
	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	OuterProductAdd_Queue<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			TPC(&a[0]),
			TPC(&b[0]),
			TPC(&index_count[0]),
			TPC(&queue[0]));

	safeSync();
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info<VALUE_TYPE> (A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

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
	get_matrix_info<VALUE_TYPE> (A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_hybb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
    	    infoA.num_cols_per_row,
   			infoA.pitch,
        	TPC(&A.matrix.ell.column_indices.values[0]),
        	TPC(&A.matrix.coo.row_indices[0]),
        	TPC(&A.matrix.coo.column_indices[0]),
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
	get_matrix_info<VALUE_TYPE> (A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_csrb<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
			TPC(&A.row_offsets[0]),
            TPC(&A.column_indices[0]),
            TPC(&x[0]), 
        	TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void spmv(	const dell_matrix_B<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
     		const cusp::array1d<VALUE_TYPE, cusp::device_memory> &x,
			cusp::array1d<VALUE_TYPE, cusp::device_memory> &y,
			cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA;
	get_matrix_info<VALUE_TYPE> (A, infoA);

#if(DEBUG)
	assert(infoA.num_cols == x.size());
	assert(infoA.num_rows == y.size());
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;

	spmv_dell_b<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (
			infoA.num_rows,
			TPC(&(*A.row_offsets)[0]),
            TPC(&(*A.column_indices)[0]),
            TPC(&A.coo.row_indices[0]),
            TPC(&A.coo.column_indices[0]),
            TPC(&x[0]), 
        	TPC(&y[0]));
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void ell_add(	cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &A,
				cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &B,
				cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &C,
				cudaStream_t &stream)
{
	mat_info<INDEX_TYPE> infoA, infoB, infoC;

	get_matrix_info<VALUE_TYPE> (A, infoA);
	get_matrix_info<VALUE_TYPE> (B, infoB);
	get_matrix_info<VALUE_TYPE> (C, infoC);

#if(DEBUG)
	assert(infoA.num_rows == infoB.num_rows && infoA.num_rows == infoC.num_rows);
	assert(infoA.num_cols == infoB.num_cols && infoA.num_cols == infoC.num_cols);
#endif

	const size_t NUM_BLOCKS = BLOCKS;
	const size_t BLOCK_SIZE = BLOCK_THREADS;
	
	ell_add<INDEX_TYPE, VALUE_TYPE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, stream>>> (	
			infoA.num_rows,
			infoA.num_cols,
			infoA.num_cols_per_row,
			infoB.num_cols_per_row,
			infoC.num_cols_per_row,
			infoA.pitch,
			infoB.pitch,
			infoC.pitch,
			TPC(&A.column_indices.values[0]),
			TPC(&B.column_indices.values[0]),
			TPC(&C.column_indices.values[0]),
			TPC(&C.values.values[0]));
}