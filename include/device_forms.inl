#if(BUILD_TYPE == GPU)

//*********************************************************************//
//Device forms

//F_call
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_call_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int j, const int sID, cudaStream_t stream)
{
	FILL<VALUE_TYPE> (accum_vf_vec[sID], 0, stream);

	get_indices<VALUE_TYPE> (s, s_indices[sID], stream);
	for(int index=0; index < entry_count_host[sID]; ++index)
	{
		column_select(Fun, s_indices[sID], index, Fun_vec[sID], stream);
		spmv(sigma, Fun_vec[sID], vf[sID], stream);
		AccumVec<VALUE_TYPE> (accum_vf_vec[sID], vf[sID], stream);

		for(int i=0; i<j; ++i)
		{
			column_select(Arg[i], s_indices[sID], index, Arg_vec[sID], stream);
			spmv(sigma, Arg_vec[sID], v[i], stream);
			spmv(Var[i], vf[sID], a[i], stream);
		}
		
		for(int i=0; i<j; ++i)
		{
			gather_reduce(v[i], temp_row_indices[sID], index_count[sID], 0, stream);
			gather_reduce(a[i], temp_col_indices[sID], index_count[sID], 1, stream);
			OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], update_queue[sID], stream);
			OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], sigma, stream);
		}
	}

	//r_prime
	spmv(Body, accum_vf_vec[sID], Body_vec[sID], stream);
	AccumVec<VALUE_TYPE> (r_prime, Body_vec[sID], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );
}

//f_call
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_call()
{
	cudaStream_t stream = stream_Call;
	for(int j=1; j<=m_maxCall; ++j)
	{
		if(valid_Call[j])
		{
			AND_OP<VALUE_TYPE> (r_prime, Call[j], s[STREAM_CALL], stream);
			count(s[STREAM_CALL], 1, &entry_count_host[STREAM_CALL], &entry_count_device[STREAM_CALL], stream);
			checkCudaErrors( cudaStreamSynchronize(stream) );

			fprintf(stderr, "f_call_%d: %d\n", j, entry_count_host[STREAM_CALL]);
			if(entry_count_host[STREAM_CALL] > 0)
				f_call_device(s[STREAM_CALL], j, STREAM_CALL, stream);
		}
	}
}

//F_list
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_list_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int j, const int sID, cudaStream_t stream)
{
	FILL<VALUE_TYPE> (accum_vf_vec[sID], 0, stream);
	
	get_indices<VALUE_TYPE> (s, s_indices[sID], stream);
	for(int i=0; i < entry_count_host[sID]; ++i)
	{
		FILL<VALUE_TYPE> (accum_var_vec[sID], 0, stream);
		FILL<VALUE_TYPE> (v_list, 0, stream);

		column_select(Fun, s_indices[sID], i, Fun_vec[sID], stream);
		spmv(sigma, Fun_vec[sID], vf[sID], stream);
		spmv(Var[0], vf[sID], a_var[sID], stream);

		AccumVec<VALUE_TYPE> (accum_vf_vec[sID], vf[sID], stream);
		AccumVec<VALUE_TYPE> (accum_var_vec[sID], a_var[sID], stream);

		for(int k=0; k<j; ++k)
		{
			column_select(Arg[k], s_indices[sID], i, Arg_vec[sID], stream);
			spmv(sigma, Arg_vec[sID], v[k], stream);
			AccumVec<VALUE_TYPE> (v_list, v[k], stream);
		}
		AccumVec<VALUE_TYPE> (v_list, LIST_vec, stream);

		gather_reduce(v_list, temp_row_indices[sID], index_count[sID], 0, stream);
		gather_reduce(accum_var_vec[sID], temp_col_indices[sID], index_count[sID], 1, stream);
		OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], update_queue[sID], stream);
		OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], sigma, stream);
	}

	//r_prime
	spmv(Body, accum_vf_vec[sID], Body_vec[sID], stream);
	AccumVec<VALUE_TYPE> (r_prime, Body_vec[sID], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );
}

//entry point
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_list()
{
	cudaStream_t stream = stream_List;
	for(int j=0; j<=m_maxList; ++j)
	{
		if(valid_List[j])
		{
			AND_OP<VALUE_TYPE> (r_prime, PrimList[j], s[STREAM_LIST], stream);
			count(s[STREAM_LIST], 1, &entry_count_host[STREAM_LIST], &entry_count_device[STREAM_LIST], stream);
			checkCudaErrors( cudaStreamSynchronize(stream) );

			fprintf(stderr, "f_list_%d: %d\n", j, entry_count_host[STREAM_LIST]);
			if(entry_count_host[STREAM_LIST] > 0)	
				f_list_device(s[STREAM_LIST], j, STREAM_LIST, stream);
		}
	}
}

// //F_set
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_set_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int sID, cudaStream_t stream)
{
	FILL<VALUE_TYPE> (accum_vf_vec[sID], 0, stream);
	FILL<VALUE_TYPE> (accum_var_vec[sID], 0, stream);

	get_indices<VALUE_TYPE> (s, s_indices[sID], stream);
	for(int i=0; i < entry_count_host[sID]; ++i)
	{
		column_select(Fun, s_indices[sID], i, Fun_vec[sID], stream);
		spmv(sigma, Fun_vec[sID], vf[sID], stream);
		spmv(Var[0], vf[sID], a_var[sID], stream);
		column_select(Arg[0], s_indices[sID], i, a_set, stream);
		column_select(Arg[1], s_indices[sID], i, Arg_vec[sID], stream);
		spmv(sigma, Arg_vec[sID], v_set, stream);

		AccumVec<VALUE_TYPE> (accum_vf_vec[sID], vf[sID], stream);
		AccumVec<VALUE_TYPE> (accum_var_vec[sID], a_var[sID], stream);
		
		gather_reduce(v_set, temp_row_indices[sID], index_count[sID], 0, stream);
		gather_reduce(a_set, temp_col_indices[sID], index_count[sID], 1, stream);
		OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], update_queue[sID], stream);		
		OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], sigma, stream);		
	}

	//sigma + (a_var (X) void) + (a_set (X) v_set)
	gather_reduce(VOID_vec, temp_row_indices[sID], index_count[sID], 0, stream);
	gather_reduce(accum_var_vec[sID], temp_col_indices[sID], index_count[sID], 1, stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], update_queue[sID], stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], sigma, stream);

	//r_prime
	spmv(Body, accum_vf_vec[sID], Body_vec[sID], stream);
	AccumVec<VALUE_TYPE> (r_prime, Body_vec[sID], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );
}

//entry point
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_set()
{
	cudaStream_t stream = stream_Set;

	AND_OP<VALUE_TYPE> (r_prime, SET, s[STREAM_SET], stream);
	count(s[STREAM_SET], 1, &entry_count_host[STREAM_SET], &entry_count_device[STREAM_SET], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );

	fprintf(stderr, "f_set: %d\n", entry_count_host[STREAM_SET]);	
	if(entry_count_host[STREAM_SET] > 0)
		f_set_device(s[STREAM_SET], STREAM_SET, stream);
}

//F_if
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_if_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int sID, cudaStream_t stream)
{
	get_indices<VALUE_TYPE> (s, s_indices[sID], stream);
	for(int i=0; i < entry_count_host[sID]; ++i)
	{
		column_select(Arg[0], s_indices[sID], i, Arg_vec[sID], stream);
		spmv(sigma, Arg_vec[sID], v_cond, stream);
	
		InnerProductStore<VALUE_TYPE> (NOT_FALSE_vec, v_cond, AND_vec1, i, stream);
		InnerProductStore<VALUE_TYPE> (FALSE_vec, v_cond, AND_vec2, i, stream);

		column_select_if<INDEX_TYPE, VALUE_TYPE> (CondTrue, s_indices[sID], AND_vec1, i, Cond_vec, stream);
		AccumVec<VALUE_TYPE> (r_prime, Cond_vec, stream);
		column_select_if<INDEX_TYPE, VALUE_TYPE> (CondFalse, s_indices[sID], AND_vec2, i, Cond_vec, stream);
		AccumVec<VALUE_TYPE> (r_prime, Cond_vec, stream);
	}
	checkCudaErrors( cudaStreamSynchronize(stream) );
}

//entry point
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_if()
{
	cudaStream_t stream = stream_If;

	AND_OP<VALUE_TYPE> (r_prime, IF, s[STREAM_IF], stream);
	count(s[STREAM_IF], 1, &entry_count_host[STREAM_IF], &entry_count_device[STREAM_IF], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );

	fprintf(stderr, "f_if: %d\n", entry_count_host[STREAM_IF]);
	if(entry_count_host[STREAM_IF] > 0)
		f_if_device(s[STREAM_IF], STREAM_IF, stream);
}

//F_primNum
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primNum_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int sID, cudaStream_t stream)
{
	FILL<VALUE_TYPE> (accum_vf_vec[sID], 0, stream);
	FILL<VALUE_TYPE> (accum_var_vec[sID], 0, stream);

	get_indices<VALUE_TYPE> (s, s_indices[sID], stream);
	for(int i=0; i < entry_count_host[sID]; ++i)
	{
		column_select(Fun, s_indices[sID], i, Fun_vec[sID], stream);
		spmv(sigma, Fun_vec[sID], vf[sID], stream);
		spmv(Var[0], vf[sID], a_var[sID], stream);
		AccumVec<VALUE_TYPE> (accum_var_vec[sID], a_var[sID], stream);
		AccumVec<VALUE_TYPE> (accum_vf_vec[sID], vf[sID], stream);
	}

	//sigma + (a_var (X) NUM)
	gather_reduce(NUM_vec, temp_row_indices[sID], index_count[sID], 0, stream);
	gather_reduce(accum_var_vec[sID], temp_col_indices[sID], index_count[sID], 1, stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], update_queue[sID], stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], sigma, stream);

	//r_prime
	spmv(Body, accum_vf_vec[sID], Body_vec[sID], stream);
	AccumVec<VALUE_TYPE> (r_prime, Body_vec[sID], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );
}

//entry point
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primNum()
{
	cudaStream_t stream = stream_Num;

	AND_OP<VALUE_TYPE> (r_prime, PrimNum, s[STREAM_NUM], stream);
	count(s[STREAM_NUM], 1, &entry_count_host[STREAM_NUM], &entry_count_device[STREAM_NUM], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );

	fprintf(stderr, "f_primNum: %d\n", entry_count_host[STREAM_NUM]);
	if(entry_count_host[STREAM_NUM] > 0)
		f_primNum_device(s[STREAM_NUM], STREAM_NUM, stream);
}

//F_primBool
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primBool_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int sID, cudaStream_t stream)
{
	FILL<VALUE_TYPE> (accum_vf_vec[sID], 0, stream);
	FILL<VALUE_TYPE> (accum_var_vec[sID], 0, stream);

	get_indices<VALUE_TYPE> (s, s_indices[sID], stream);
	for(int i=0; i < entry_count_host[sID]; ++i)
	{
		column_select(Fun, s_indices[sID], i, Fun_vec[sID], stream);
		spmv(sigma, Fun_vec[sID], vf[sID], stream);
		spmv(Var[0], vf[sID], a_var[sID], stream);
		AccumVec<VALUE_TYPE> (accum_var_vec[sID], a_var[sID], stream);
		AccumVec<VALUE_TYPE> (accum_vf_vec[sID], vf[sID], stream);
	}

	//sigma + (a_var (X) #T#F)
	gather_reduce(BOOL_vec, temp_row_indices[sID], index_count[sID], 0, stream);
	gather_reduce(accum_var_vec[sID], temp_col_indices[sID], index_count[sID], 1, stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], update_queue[sID], stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], sigma, stream);

	//r_prime
	spmv(Body, accum_vf_vec[sID], Body_vec[sID], stream);
	AccumVec<VALUE_TYPE> (r_prime, Body_vec[sID], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );
}

//entry point
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primBool()
{
	cudaStream_t stream = stream_Bool;

	AND_OP<VALUE_TYPE> (r_prime, PrimBool, s[STREAM_BOOL], stream);
	count(s[STREAM_BOOL], 1, &entry_count_host[STREAM_BOOL], &entry_count_device[STREAM_BOOL], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );

	fprintf(stderr, "f_primBool: %d\n", entry_count_host[STREAM_BOOL]);
	if(entry_count_host[STREAM_BOOL] > 0)
		f_primBool_device(s[STREAM_BOOL], STREAM_BOOL, stream);
}

//F_primVoid
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primVoid_device(const cusp::array1d<VALUE_TYPE, cusp::device_memory> &s, const int sID, cudaStream_t stream)
{
	FILL<VALUE_TYPE> (accum_vf_vec[sID], 0, stream);
	FILL<VALUE_TYPE> (accum_var_vec[sID], 0, stream);

	get_indices<VALUE_TYPE> (s, s_indices[sID], stream);
	for(int i=0; i < entry_count_host[sID]; ++i)
	{
		column_select(Fun, s_indices[sID], i, Fun_vec[sID], stream);
		spmv(sigma, Fun_vec[sID], vf[sID], stream);
		spmv(Var[0], vf[sID], a_var[sID], stream);
		AccumVec<VALUE_TYPE> (accum_var_vec[sID], a_var[sID], stream);
		AccumVec<VALUE_TYPE> (accum_vf_vec[sID], vf[sID], stream);
	}

	//sigma + (a_var (X) VOID)
	gather_reduce(VOID_vec, temp_row_indices[sID], index_count[sID], 0, stream);
	gather_reduce(accum_var_vec[sID], temp_col_indices[sID], index_count[sID], 1, stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], update_queue[sID], stream);
	OuterProductAdd(temp_row_indices[sID], temp_col_indices[sID], index_count[sID], sigma, stream);

	//r_prime
	spmv(Body, accum_vf_vec[sID], Body_vec[sID], stream);
	//DEBUG_PRINT("Body_vec:  ", Body_vec[sID]);
	AccumVec<VALUE_TYPE>(r_prime, Body_vec[sID], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );
}

//entry point
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primVoid()
{
	cudaStream_t stream = stream_Void;

	AND_OP<VALUE_TYPE> (r_prime, PrimVoid, s[STREAM_VOID], stream);
	count(s[STREAM_VOID], 1, &entry_count_host[STREAM_VOID], &entry_count_device[STREAM_VOID], stream);
	checkCudaErrors( cudaStreamSynchronize(stream) );

	fprintf(stderr, "f_PrimVoid: %d\n", entry_count_host[STREAM_VOID]);
	if(entry_count_host[STREAM_VOID] > 0)
		f_primVoid_device(s[STREAM_VOID], STREAM_VOID, stream);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::LoadMatrix(
			cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
			cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
{
	LoadMatrix(src, dst);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::LoadMatrix(
			cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
			hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
{
	LoadMatrix(src, dst);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::LoadMatrix(
			cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
			dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
{
	LoadMatrix(src, dst);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::RebuildMatrix(
			dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat)
{
	RebuildDellMatrix_device(mat);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_UpdateStore_device(const int sID, cudaStream_t stream)
{
	//check size
	for(int i=0; i<NUM_TRANSFER_FUNCS; i++)
	{
		fprintf(stderr, "update matrix: %d\n", i);
		UpdateMatrix(sigma, update_queue[i]);
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_UpdateStore()
{
	cudaStream_t stream = stream_Update;

	f_UpdateStore_device(STREAM_UPDATE, stream);
}

// hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
// {
// 	LoadHybMatrix_device(src, dst);
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::LoadMatrix(
			cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src,
			dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst)
{
	LoadDellMatrix_device(src, dst);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::RebuildMatrix(
			dell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat)
{
	RebuildDellMatrix_device(mat);
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
// void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::UpdateStore(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &sigma_CSR,
// 															cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &sigma_ELL,
// 															cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &new_sigma)
// {
// 	//check size
// 	INDEX_TYPE matrix_size, threshold;
// 	CalcSize(sigma_CSR, sigma_ELL, &matrix_size, stream_Update);
// 	if(new_sigma.column_indices.size() > matrix_size || threshold > 0)
// 	{
// 		RebuildMatrix(sigma_CSR, sigma_ELL, new_sigma);
// 		checkCudaErrors( cudaStreamSynchronize(stream) );
// 	}
// }

// template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
// void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::UpdateStore()
// {
// 	if(sigma == &sigmaA)
// 		UpdateStore(sigmaA, sigma_ELL, sigmaB);
// 	else
// 		UpdateStore(sigmaB, sigma_ELL, sigmaA);
// }

#endif
