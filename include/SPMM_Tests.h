#ifndef SPMM_TESTS_H
#define SPMM_TESTS_H

void SPMMTests(const std::string &filenameA)
{
	//set cuda device
	cudaSetDevice(0);

	#if(PRECISION == 32)
		#define PTYPE	float
	#elif(PRECISION == 64)
		#define PTYPE	double
	#endif

	int mat_rows, mat_cols;
	CuspVectorInt_h rows_h, cols_h;
	CuspVectorInt_d rows_d, cols_d;
#if(PRECISION == 32)
	CuspVectorS_h vals_h;
	CuspVectorS_d vals_d;
	fprintf(stderr, "PRECISION: SINGLE\n");
#elif(PRECISION == 64)
	CuspVectorD_h vals_h;
	CuspVectorD_d vals_d;
	fprintf(stderr, "PRECISION: DOUBLE\n");
#endif

	std::vector< std::vector< std::pair<int,PTYPE> > > GS_mat;
	int NNZ = ReadMatrixFile(filenameA, GS_mat, rows_h, cols_h, vals_h, mat_rows, mat_cols);
	if(NNZ == 0)
		exit(1);

	//calcuate average 
	int max_rows = 0, nnz_count = 0, avg_row;
	float std_dev = 0;
	for(int i=0; i<GS_mat.size(); ++i)
	{
		if(GS_mat[i].size() > max_rows)
			max_rows = GS_mat[i].size();
		nnz_count += GS_mat[i].size();
	}
	avg_row = nnz_count / mat_rows;

	//caculate standard deviation
	for(int i=0; i<GS_mat.size(); ++i)
	{
		std_dev += (GS_mat[i].size() - avg_row) * (GS_mat[i].size() - avg_row);
	}
	std_dev = sqrt(std_dev / mat_rows);
	// fprintf(stderr, "average entries per row: %d\n", avg_row);
	// fprintf(stderr, "max row size: %d\n", max_rows);
	// fprintf(stderr, "standard deviation: %f\n", std_dev);
	// fprintf(stderr, "nnz count: %d\tNNZ returned: %d\n", nnz_count, NNZ);

	const size_t BINS = MAX_OFFSET;
	const size_t width = 4;
	int ell_width = avg_row; //(avg_row / width + (((avg_row % width) == 0) ? 0 : 1)) * width;
	//ell_width = 6;
	int bin_size = ell_width;
	int overflow_size = NNZ;//mat_rows*ell_width*(BINS/2);
	
	//fprintf(stderr, "alpha: %f\n", DCSR_matA.alpha);
	fprintf(stderr, "rows: %d  cols: %d\n", mat_rows, mat_cols);
	fprintf(stderr, "ell width: %d\n", ell_width);
	fprintf(stderr, "bin_size: %d\n", ell_width);
	fprintf(stderr, "overflow memsize: %d\n", overflow_size);
	fprintf(stderr, "VECTOR_SIZE: %d\n", VECTOR_SIZE);
	if(mat_rows != mat_cols)
	{
		fprintf(stderr, "**ERROR**  rows != cols\n");
		exit(1);
	}

	//Setup and initialize matrices
#if(RUN_CSR == 1)
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matA;
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matB;
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matC;
	CSR_matA.resize(mat_rows, mat_cols, NNZ);
	CSR_matB.resize(mat_rows, mat_cols, NNZ);
#endif
#if(RUN_HYB == 1)
	hyb_matrix<int, PTYPE, cusp::device_memory> HYB_mat;
	HYB_mat.resize(mat_rows, mat_cols, min(overflow_size, (int)ALIGN_UP(NNZ,32)), ell_width);
#endif
#if(RUN_DCSR == 1)
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matA;
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matB;
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matC;
	DCSR_matA.resize(mat_rows, mat_cols, bin_size, 2.5);
	DCSR_matB.resize(mat_rows, mat_cols, bin_size, 2.5);
	device::Initialize_Matrix(DCSR_matA);
	device::Initialize_Matrix(DCSR_matB);
#endif

	//initialize streams
	createStreams();
	//cudaPrintfInit();

	//sort vectors by row
	cusp::sort_by_row(rows_h, cols_h, vals_h);

	//half size vectors
	rows_d = rows_h;
	cols_d = cols_h;
	vals_d = vals_h;

	#if(RUN_DCSR == 1)
	LoadMatrix(DCSR_matA, rows_d, cols_d, vals_d, NNZ);
	LoadMatrix(DCSR_matB, rows_d, cols_d, vals_d, NNZ);
	//LoadMatrix(DCSR_matA, CSR_mat_d);
	//LoadMatrix(CSR_mat_d, DCSR_matA);
	#endif
	#if(RUN_HYB == 1)
	LoadMatrix(HYB_mat, rows_d, cols_d, vals_d, NNZ);
	//LoadMatrix(HYB_mat, CSR_mat_d);
	#endif
	#if(RUN_CSR == 1)
	LoadMatrix(CSR_matA, rows_d, cols_d, vals_d, NNZ);
	LoadMatrix(CSR_matB, rows_d, cols_d, vals_d, NNZ);
	//LoadMatrix(CSR_mat, CSR_mat_d);
	#endif

	// size_t total_mem, free_mem;
	// cudaMemGetInfo(&free_mem, &total_mem);
	// fprintf(stderr, "total_mem: %lld   free_mem: %lld  used_mem: %lld\n", total_mem, free_mem, total_mem - free_mem);

	/****************************************************************************/
	//Free memory
	rows_d.clear();
	cols_d.clear();
	vals_d.clear();
	rows_d.shrink_to_fit();
	cols_d.shrink_to_fit();
	vals_d.shrink_to_fit();
	/****************************************************************************/
	
	// cudaMemGetInfo(&free_mem, &total_mem);
	// fprintf(stderr, "total_mem: %lld   free_mem: %lld  used_mem: %lld\n", total_mem, free_mem, total_mem - free_mem);
	//Check to ensure that matrices are equivalent to CPU generated matrix
	//CheckMatrices(DCSR_matA, HYB_mat, GS_mat);
	//CheckMatrices(DCSR_matA, GS_mat);
	//CheckMatrices(CSR_mat, GS_mat);
	//CheckMatrices(DCSR_matA, CSR_matA);
	//CheckMatrices(DCSR_matB, CSR_matB);

	//SPMV tests
	#define TEST_COUNT	1
	double startTime, endTime;

#if(SPMM_TEST == 1)

#if(RUN_DCSR == 1)
	startTime = omp_get_wtime();

	for(int i=0; i<TEST_COUNT; i++)
	{
		device::spmm(DCSR_matA, DCSR_matB, DCSR_matC, __streams);
	}

	endTime = omp_get_wtime();
	fprintf(stderr, "DCSR SpMM time:  %f\n", (endTime - startTime));
#endif

#if(RUN_HYB == 1)
	startTime = omp_get_wtime();

	endTime = omp_get_wtime();
#endif

#if(RUN_CSR == 1)
	startTime = omp_get_wtime();
	
	for(int i=0; i<TEST_COUNT; i++)
	{
		cusp::multiply(CSR_matA, CSR_matB, CSR_matC);
	}

	endTime = omp_get_wtime();
	fprintf(stderr, "CSR SpMM time:  %f\n", (endTime - startTime));
#endif

#endif

	//check final results
	CheckMatrices(DCSR_matC, CSR_matC);

	destroyStreams();
}

#endif