#ifndef LOAD_MATRIX_H
#define LOAD_MATRIX_H

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
void LoadMatrix( 	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &rows,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &cols,
					const cusp::array1d<VALUE_TYPE, MEM_TYPE> &vals,
					const unsigned int NNZ);

template<typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void LoadMatrix( 	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src);

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void LoadMatrix( 	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &rows,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &cols,
					const cusp::array1d<VALUE_TYPE, MEM_TYPE> &vals,
					const unsigned int NNZ);

template<typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix( 	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src);

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void LoadMatrix( 	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &rows,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &cols,
					const cusp::array1d<VALUE_TYPE, MEM_TYPE> &vals,
					const unsigned int NNZ);

template<typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix( 	cusp::coo_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals);

template<typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src);

template<typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void LoadMatrix( 	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &src);

template <typename INDEX_TYPE, typename VALUE_TYPE>
int ReadMatrixFile(	const std::string &filename, 
					std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &mat,
					CuspVectorInt_h &row_vec, 
					CuspVectorInt_h &col_vec,
					cusp::array1d<VALUE_TYPE, cusp::host_memory> &val_vec,
					int &mat_rows,
					int &mat_cols,
					bool zeroIndex = true)
{
	std::ifstream mat_file(filename.c_str());
	int nnz = 0;

	if(mat_file.is_open())
	{
		int rows, cols;
		char buf[256];

		do
		{
			mat_file.getline(buf, 256);
		} while(buf[0] == '%');

		std::istringstream iss(buf);
		iss >> rows;
		iss >> cols;
		iss >> nnz;

		fprintf(stderr, "Matrix Size: %d x %d\tNNZ: %d\n", rows, cols, nnz);
		mat.resize(rows);
		row_vec.resize(nnz);
		col_vec.resize(nnz);
		val_vec.resize(nnz);

		for(int i=0; i<nnz; ++i)
		{
			INDEX_TYPE row, col;
			VALUE_TYPE val = 0.0;
			memset(buf, 0, sizeof(buf));
			mat_file.getline(buf, 256);
			std::istringstream iss(buf);

			iss >> row;
			iss >> col;
			iss >> val;
			if(iss.eof() || iss.fail() || val == 0)
				val = 1;
			if(!zeroIndex) {
				row--;			//matrix file uses 1 based indexing
				col--;			//matrix file uses 1 based indexing
			}
			row_vec[i] = row;
			col_vec[i] = col;
			val_vec[i] = val;
			//printf("(%d %d) : %f\n", row, col, val);
			mat[row].push_back(std::pair<INDEX_TYPE,VALUE_TYPE>(col, val));
		}

		for(int i=0; i<rows; ++i)
			sort(mat[i].begin(), mat[i].end());

		mat_rows = rows;
		mat_cols = cols;
		fprintf(stderr, "Finished reading matrix: %s\n", filename.c_str());
	}
	else
	{
		fprintf(stderr, "Error opening matrix file: %s\n", filename.c_str());
		return 0;
	}

	return nnz;
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
int getPoissonMatrix(	std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &mat,
								cusp::array1d<INDEX_TYPE, cusp::host_memory> &row_vec, 
								cusp::array1d<INDEX_TYPE, cusp::host_memory> &col_vec,
								cusp::array1d<VALUE_TYPE, cusp::host_memory> &val_vec,
								int numPts,
								int M,
								int N,
								int K) 
{
	cusp::coo_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> A;
	mat.resize(M);

	// create Poisson matrix
	switch(numPts) {
		case 5:	//2D
			cusp::gallery::poisson5pt(A, M, N);
			break;
		case 7:	//3D
			cusp::gallery::poisson7pt(A, M, N, K);
			break;
		case 9:	//2D
			cusp::gallery::poisson9pt(A, M, N);
			break;
		case 27:	//3D
			cusp::gallery::poisson27pt(A, M, N, K);
			break;
		default:
			break;	
	}

	A.row_indices = row_vec;
	A.column_indices = col_vec;
	A.values = val_vec;

	int size = row_vec.size();
	for(int i=0; i<size; ++i) {
		int row = row_vec[i];
		std::pair<INDEX_TYPE,VALUE_TYPE> p = std::make_pair(col_vec[i], val_vec[i]);
		mat[row].push_back(p);
	}

	return size;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void CheckMatrices(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &DCSR_mat,
					hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &HYB_mat,
					const std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &GS_mat)
{
	mat_info<int> infoDCSRMat, infoHybMat;
	get_matrix_info<int> (DCSR_mat, infoDCSRMat);
	get_matrix_info<int> (HYB_mat, infoHybMat);

	CuspVectorInt_h cols_A = DCSR_mat.column_indices;
	CuspVectorInt_h cols_B = HYB_mat.matrix.ell.column_indices.values;
	CuspVectorS_h vals_A = DCSR_mat.values;
	CuspVectorS_h vals_B = HYB_mat.matrix.ell.values.values;
	CuspVectorInt_h rsA = DCSR_mat.row_sizes;
	CuspVectorInt_h rsB = HYB_mat.row_sizes;
	CuspVectorInt_h roA = DCSR_mat.row_offsets;
	CuspVectorInt_h overflow_rowB = HYB_mat.matrix.coo.row_indices;
	CuspVectorInt_h overflow_colB = HYB_mat.matrix.coo.column_indices;
	CuspVectorS_h overflow_valsB = HYB_mat.matrix.coo.values;

	int num_rows = infoDCSRMat.num_rows;
	int pitchA = infoDCSRMat.pitch;
	int pitchB = infoHybMat.pitch;
	int ell_widthB = infoHybMat.num_cols_per_row;
	int overflow_size = rsB[num_rows];

	std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > vec_mat1(GS_mat.size()), vec_mat2(GS_mat.size());

	int nnz = 0;
	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		int r_idxA = 0, r_idxB = 0, rlA = rsA[row], rlB = rsB[row];
		//load DCSR mat entries
		for(int offset=0; offset<BINS; offset++)
		{
			int start = roA[offset*pitchA + row*2];
			int end = roA[offset*pitchA + row*2 + 1];

			for(int jj=start; jj<end && r_idxA < rlA; jj++, r_idxA++)
				vec_mat1[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_A[jj], vals_A[jj]) );
		}

		//load HYB mat entries
		int offsetB = row;
		for(r_idxB=0; r_idxB < rlB && r_idxB < ell_widthB; r_idxB++)
		{
			vec_mat2[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_B[offsetB + r_idxB*pitchB], vals_B[offsetB + r_idxB*pitchB]) );
		}

		for(int i=0; i < overflow_size; i++)
		{
			if(overflow_rowB[i] == row)
				vec_mat2[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(overflow_colB[i], overflow_valsB[i]) );
		}

		//sort vectors
		sort(vec_mat1[row].begin(), vec_mat1[row].end());
		sort(vec_mat2[row].begin(), vec_mat2[row].end());

		if(vec_mat1[row].size() != GS_mat[row].size())
			fprintf(stderr, "*** Row Size A: %d   Row Size GS: %d\n", vec_mat1[row].size(), GS_mat[row].size());

		if(vec_mat2[row].size() != GS_mat[row].size())
			fprintf(stderr, "*** Row Size B: %d   Row Size GS: %d\n", vec_mat2[row].size(), GS_mat[row].size());

		if(vec_mat1[row].size() != vec_mat2[row].size())
			fprintf(stderr, "*** Row %d \t A Size: %d \t B Size: %d\n", row, vec_mat1[row].size(), vec_mat2[row].size());

		for(int i=0; i<GS_mat[row].size(); ++i)
		{
			if(	vec_mat1[row][i].first != GS_mat[row][i].first || vec_mat2[row][i].first != GS_mat[row][i].first ||
				vec_mat1[row][i].second != GS_mat[row][i].second || vec_mat2[row][i].second != GS_mat[row][i].second)
			{
				fprintf(stderr, "GS(%d, %d):  %f\t", row, GS_mat[row][i].first, GS_mat[row][i].second);
				fprintf(stderr, "DCSR(%d, %d):  %f \t HYB(%d, %d):  %f", row, vec_mat1[row][i].first, vec_mat1[row][i].second, row, vec_mat2[row][i].first, vec_mat2[row][i].second);
				fprintf(stderr, "\n");
				num_diff++;
			}
		}

		nnz += GS_mat[row].size();
	}

	//overflow sections
	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
	else
		fprintf(stderr, "Matrices have %d differences...\n", num_diff);

	fprintf(stderr, "Number of Nonzeros in final matrix: %d\n", nnz);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void CheckMatrices(	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &DCSR_mat,
					const std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &GS_mat)
{
	mat_info<int> infoDCSRMat;
	get_matrix_info<int> (DCSR_mat, infoDCSRMat);

	CuspVectorInt_h cols_A = *DCSR_mat.column_indices;
	CuspVectorS_h vals_A = *DCSR_mat.values;
	CuspVectorInt_h rsA = DCSR_mat.row_sizes;
	CuspVectorInt_h roA = DCSR_mat.row_offsets;

	int num_rows = infoDCSRMat.num_rows;
	int pitchA = infoDCSRMat.pitch;

	std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > vec_mat1(GS_mat.size());

	int nnz = 0;
	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		int r_idxA = 0, rlA = rsA[row];
		//load DCSR mat entries
		for(int offset=0; offset<BINS; offset++)
		{
			int start = roA[offset*pitchA + row*2];
			int end = roA[offset*pitchA + row*2 + 1];
				
			for(int jj=start; jj<end && r_idxA < rlA; jj++, r_idxA++)
			{
				vec_mat1[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_A[jj], vals_A[jj]) );
			}
		}

		//sort vectors
		sort(vec_mat1[row].begin(), vec_mat1[row].end());

		if(vec_mat1[row].size() != GS_mat[row].size())
			fprintf(stderr, "*** Row Size A: %d   Row Size GS: %d\n", vec_mat1[row].size(), GS_mat[row].size());

		for(int i=0; i<GS_mat[row].size(); ++i)
		{
			if(	vec_mat1[row][i].first != GS_mat[row][i].first || vec_mat1[row][i].second != GS_mat[row][i].second)
			{
				// fprintf(stderr, "GS(%d, %d):  %f\t", row, GS_mat[row][i].first, GS_mat[row][i].second);
				// fprintf(stderr, "DCSR(%d, %d):  %f", row, vec_mat1[row][i].first, vec_mat1[row][i].second);
				//fprintf(stderr, "\n");
				num_diff++;
			}
		}

		nnz += GS_mat[row].size();
	}

	//overflow sections
	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
	else
		fprintf(stderr, "Matrices have %d differences...\n", num_diff);

	fprintf(stderr, "Number of Nonzeros in final matrix: %d\n", nnz);
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void CheckMatrices(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &CSR_mat,
					const std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > &GS_mat)
{
	mat_info<int> infoCSRMat;
	get_matrix_info<int> (CSR_mat, infoCSRMat);

	CuspVectorInt_h row_offsets = CSR_mat.row_offsets;
	CuspVectorInt_h cols = CSR_mat.column_indices;
	CuspVectorS_h vals = CSR_mat.values;

	int num_rows = infoCSRMat.num_rows;
	std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > vec_mat1(GS_mat.size());

	int nnz = 0;
	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		for(int idx=row_offsets[row]; idx < row_offsets[row+1]; ++idx)
		{
			vec_mat1[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols[idx], vals[idx]) );
		}

		//sort vectors
		sort(vec_mat1[row].begin(), vec_mat1[row].end());

		if(vec_mat1[row].size() != GS_mat[row].size())
				fprintf(stderr, "*** Row %d   Size A: %d   Row Size GS: %d\n", row, vec_mat1[row].size(), GS_mat[row].size());

		for(int i=0; i<GS_mat[row].size(); ++i)
		{
			if(	vec_mat1[row][i].first != GS_mat[row][i].first || vec_mat1[row][i].second != GS_mat[row][i].second )
			{
				fprintf(stderr, "GS(%d, %d):  %f\t", row, GS_mat[row][i].first, GS_mat[row][i].second);
				fprintf(stderr, "CSR(%d, %d):  %f", row, vec_mat1[row][i].first, vec_mat1[row][i].second);
				fprintf(stderr, "\n");
				num_diff++;
			}
		}

		nnz += GS_mat[row].size();
	}

	//overflow sections
	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
	else
		fprintf(stderr, "Matrices have %d differences...\n", num_diff);

	fprintf(stderr, "Number of Nonzeros in final matrix: %d\n", nnz);
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void CheckMatrices(	const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &DCSR_mat,
							const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &CSR_mat)
{
	mat_info<int> infoDCSRMat, infoCSRMat;
	get_matrix_info<int> (DCSR_mat, infoDCSRMat);
	get_matrix_info<int> (CSR_mat, infoCSRMat);

	CuspVectorInt_h cols_A = *DCSR_mat.column_indices;
	CuspVectorS_h vals_A = *DCSR_mat.values;
	CuspVectorInt_h rsA = DCSR_mat.row_sizes;
	CuspVectorInt_h roA = DCSR_mat.row_offsets;

	CuspVectorInt_h roB = CSR_mat.row_offsets;
	CuspVectorInt_h cols_B = CSR_mat.column_indices;
	CuspVectorS_h vals_B = CSR_mat.values;
	
	int num_rows = infoDCSRMat.num_rows;
	int pitchA = infoDCSRMat.pitch;

	std::vector< std::vector< std::pair<INDEX_TYPE,VALUE_TYPE> > > vec_mat1(DCSR_mat.num_rows), vec_mat2(DCSR_mat.num_rows);

	int nnz = 0;
	int num_diff = 0;
	for(int row=0; row<num_rows; ++row)
	{
		int r_idxA = 0, rlA = rsA[row];
		//load DCSR mat entries
		for(int offset=0; offset<BINS; offset++)
		{
			int start = roA[offset*pitchA + row*2];
			int end = roA[offset*pitchA + row*2 + 1];
			
			for(int jj=start; jj<end && r_idxA < rlA; jj++, r_idxA++)
			{
				vec_mat1[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_A[jj], vals_A[jj]) );
			}
		}

		int start = roB[row];
		int end = roB[row+1];
		for(int idx=start; idx < end; ++idx)
		{
			vec_mat2[row].push_back( std::pair<INDEX_TYPE,VALUE_TYPE>(cols_B[idx], vals_B[idx]) );
		}

		//sort vectors
		sort(vec_mat1[row].begin(), vec_mat1[row].end());
		sort(vec_mat2[row].begin(), vec_mat2[row].end());
		nnz += vec_mat2[row].size();
	}

	//overflow sections
	if(num_diff == 0)
		fprintf(stderr, "Matrices are identical...\n");
	else
		fprintf(stderr, "Matrices have %d differences...\n", num_diff);

	fprintf(stderr, "Number of Nonzeros in final matrix: %d\n", nnz);
}

//DCSR matrix
template<typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
void LoadMatrix( 	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &mat,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &rows,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &cols,
					const cusp::array1d<VALUE_TYPE, MEM_TYPE> &vals,
					const unsigned int NNZ)
{
	double startTime = omp_get_wtime();
	//device::UpdateMatrix(mat, rows, cols, vals);
	device::LoadMatrix(mat, rows, cols, vals, NNZ);
	safeSync();
	double endTime = omp_get_wtime();

	//cudaPrintfDisplay(stdout, true);
	//fprintf(stderr, "DCSR matrix load time:  %f\n", (endTime - startTime));

	device::BinRows(mat);
}

template<typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void LoadMatrix( 	dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src)
{
	double startTime = omp_get_wtime();
	device::LoadMatrix(dst, src);
	safeSync();
	double endTime = omp_get_wtime();

	fprintf(stderr, "CSR->DCSR matrix load time:  %f\n", (endTime - startTime));
}

//HYB matrix
template<typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void LoadMatrix( 	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &rows,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &cols,
					const cusp::array1d<VALUE_TYPE, MEM_TYPE> &vals,
					const unsigned int NNZ)
{
	double startTime = omp_get_wtime();
	//device::UpdateMatrix(mat, rows, cols, vals, NNZ);
	device::LoadMatrix(mat, rows, cols, vals, NNZ);
	safeSync();
	double endTime = omp_get_wtime();
	//cudaPrintfDisplay(stdout, true);
	fprintf(stderr, "Hybrid matrix update time:  %f\n", (endTime - startTime));
	fprintf(stderr, "Overflow entries: %d\n", mat.num_overflow);

	startTime = omp_get_wtime();
	thrust::sort_by_key(mat.matrix.coo.row_indices.begin(), mat.matrix.coo.row_indices.begin() + mat.num_overflow,
						thrust::make_zip_iterator(thrust::make_tuple(mat.matrix.coo.column_indices.begin(), mat.matrix.coo.values.begin())) );
	safeSync();
	endTime = omp_get_wtime();
	fprintf(stderr, "Hybrid matrix sort time:  %f\n", (endTime - startTime));
}

//HYB matrix from CSR
template<typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix( 	hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src)
{
	double startTime = omp_get_wtime();
	dst.matrix = src;
	//device::LoadMatrix(dst, src);
	safeSync();
	double endTime = omp_get_wtime();
	fprintf(stderr, "CSR->HYB matrix load time:  %f\n", (endTime - startTime));
	fprintf(stderr, "Overflow entries: %d\n", dst.num_overflow);
}

//CSR matrix from COO
template<typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &rows,
					const cusp::array1d<INDEX_TYPE, MEM_TYPE> &cols,
					const cusp::array1d<VALUE_TYPE, MEM_TYPE> &vals,
					const unsigned int NNZ)
{
	double startTime = omp_get_wtime();
	device::LoadMatrix(mat, rows, cols, vals, NNZ);
	safeSync();
	double endTime = omp_get_wtime();
	//cudaPrintfDisplay(stdout, true);
	//fprintf(stderr, "CSR matrix load time:  %f\n", (endTime - startTime));
}

//CSR matrix from CSR
template<typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &src)
{
	double startTime = omp_get_wtime();
	device::LoadMatrix(dst, src);
	safeSync();
	double endTime = omp_get_wtime();
	fprintf(stderr, "CSR->CSR matrix load time:  %f\n", (endTime - startTime));
}

template<typename INDEX_TYPE, typename VALUE_TYPE>
void LoadMatrix( 	cusp::coo_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &rows,
					cusp::array1d<INDEX_TYPE, cusp::device_memory> &cols,
					cusp::array1d<VALUE_TYPE, cusp::device_memory> &vals)
{
	double startTime = omp_get_wtime();

	dst.row_indices = rows;
	dst.column_indices = cols;
	dst.values = vals;

	safeSync();
	double endTime = omp_get_wtime();
	fprintf(stderr, "COO matrix update time:  %f\n", (endTime - startTime));
}

template<typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
void LoadMatrix( 	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &dst,
					const dcsr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory, BINS> &src)
{
	double startTime = omp_get_wtime();
	device::LoadMatrix(dst, src);
	safeSync();
	double endTime = omp_get_wtime();
	fprintf(stderr, "DCSR->CSR matrix load time:  %f\n", (endTime - startTime));
}

#endif