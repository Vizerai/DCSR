#ifndef SPMM_TESTS_H
#define SPMM_TESTS_H

void SPMMTests(const std::string fileA, const std::string fileB, const std::string fileC)
{
	//set cuda device
	cudaSetDevice(0);

	#if(PRECISION == 32)
		#define PTYPE	float
	#elif(PRECISION == 64)
		#define PTYPE	double
	#endif

	int mat_rowsR, mat_rowsA, mat_rowsP, mat_colsR, mat_colsA, mat_colsP;
	CuspVectorInt_h rows_h1, rows_h2, rows_h3, cols_h1, cols_h2, cols_h3;
	CuspVectorInt_d rows_d1, rows_d2, rows_d3, cols_d1, cols_d2, cols_d3;
#if(PRECISION == 32)
	CuspVectorS_h vals_h1, vals_h2, vals_h3;
	CuspVectorS_d vals_d1, vals_d2, vals_d3;
	fprintf(stderr, "PRECISION: SINGLE\n");
#elif(PRECISION == 64)
	CuspVectorD_h vals_h1, vals_h2, vals_h3;
	CuspVectorD_d vals_d1, vals_d2, vals_d3;
	fprintf(stderr, "PRECISION: DOUBLE\n");
#endif

	fprintf(stderr, "%s\n", fileA.c_str());
	fprintf(stderr, "%s\n", fileB.c_str());
	fprintf(stderr, "%s\n", fileC.c_str());

	std::vector< std::vector< std::pair<int,PTYPE> > > GS_mat;
	int NNZ_R = ReadMatrixFile(fileA, GS_mat, rows_h1, cols_h1, vals_h1, mat_rowsR, mat_colsR);
	int NNZ_A = ReadMatrixFile(fileB, GS_mat, rows_h2, cols_h2, vals_h2, mat_rowsA, mat_colsA);
	int NNZ_P = ReadMatrixFile(fileC, GS_mat, rows_h3, cols_h3, vals_h3, mat_rowsP, mat_colsP);
	// const int length = 1024;
	// const int numPts = 5;	//5 and 9 are 2D, 7 and 27 are 3D
	//int NNZ = getPoissonMatrix(GS_mat, rows_h, cols_h, vals_h, numPts, length, length, length);

	const size_t BINS = MAX_OFFSET;
	
	fprintf(stderr, "rowsR: %d colsR: %d  rowsA: %d colsA: %d  rowsP: %d colsP: %d\n", mat_rowsR, mat_colsR, mat_rowsA, mat_colsA, mat_rowsP, mat_colsP);
	fprintf(stderr, "NNZ_R: %d  NNZ_A: %d  NNZ_P: %d\n", NNZ_R, NNZ_A, NNZ_P);
	fprintf(stderr, "VECTOR_SIZE: %d\n", __VECTOR_SIZE);

	//Setup and initialize matrices
#if(RUN_CSR == 1)
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matR;
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matA;
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matP;
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matAP;
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matRAP;
	CSR_matR.resize(mat_rowsR, mat_colsR, NNZ_R);
	CSR_matA.resize(mat_rowsA, mat_colsA, NNZ_A);
	CSR_matP.resize(mat_rowsP, mat_colsP, NNZ_P);
#endif
#if(RUN_DCSR == 1)
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matR;
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matA;
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matP;
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matAP;
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matRAP;
	DCSR_matR.resize(mat_rowsR, mat_colsR, NNZ_R);
	DCSR_matA.resize(mat_rowsA, mat_colsA, NNZ_A);
	DCSR_matP.resize(mat_rowsP, mat_colsP, NNZ_P);
	device::Initialize_Matrix(DCSR_matR);
	device::Initialize_Matrix(DCSR_matA);
	device::Initialize_Matrix(DCSR_matP);
#endif

	//initialize streams
	createStreams();

	//sort vectors by row
	cusp::sort_by_row(rows_h1, cols_h1, vals_h1);
	cusp::sort_by_row(rows_h2, cols_h2, vals_h2);
	cusp::sort_by_row(rows_h3, cols_h3, vals_h3);

	//half size vectors
	rows_d1 = rows_h1, rows_d2 = rows_h2, rows_d3 = rows_h3;
	cols_d1 = cols_h1, cols_d2 = cols_h2, cols_d3 = cols_h3;
	vals_d1 = vals_h1, vals_d2 = vals_h2, vals_d3 = vals_h3;

	fprintf(stderr, "loading matrices...\n");
	#if(RUN_DCSR == 1)
	LoadMatrix(DCSR_matR, rows_d1, cols_d1, vals_d1, NNZ_R);
	LoadMatrix(DCSR_matA, rows_d2, cols_d2, vals_d2, NNZ_A);
	LoadMatrix(DCSR_matP, rows_d3, cols_d3, vals_d3, NNZ_P);
	#endif
	#if(RUN_CSR == 1)
	LoadMatrix(CSR_matR, rows_d1, cols_d1, vals_d1, NNZ_R);
	LoadMatrix(CSR_matA, rows_d2, cols_d2, vals_d2, NNZ_A);
	LoadMatrix(CSR_matP, rows_d3, cols_d3, vals_d3, NNZ_P);
	#endif

	// size_t total_mem, free_mem;
	// cudaMemGetInfo(&free_mem, &total_mem);
	// fprintf(stderr, "total_mem: %lld   free_mem: %lld  used_mem: %lld\n", total_mem, free_mem, total_mem - free_mem);

	/****************************************************************************/
	//Free memory
	fprintf(stderr, "free memory...\n");
	rows_d1.clear(), rows_d2.clear(), rows_d3.clear();
	cols_d1.clear(), cols_d2.clear(), cols_d3.clear();
	vals_d1.clear(), vals_d2.clear(), vals_d3.clear();
	rows_d1.shrink_to_fit(), rows_d2.shrink_to_fit(), rows_d3.shrink_to_fit();
	cols_d1.shrink_to_fit(), cols_d2.shrink_to_fit(), cols_d3.shrink_to_fit();
	vals_d1.shrink_to_fit(), vals_d2.shrink_to_fit(), vals_d3.shrink_to_fit();
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

	fprintf(stderr, "starting tests...\n");
#if(RUN_DCSR == 1)
	startTime = omp_get_wtime();

	device::spmm(DCSR_matA, DCSR_matP, DCSR_matAP, __streams);
	safeSync();
	// device::spmm(DCSR_matR, DCSR_matAP, DCSR_matRAP, __streams);
	// safeSync();
	
	endTime = omp_get_wtime();
	fprintf(stderr, "DCSR SpMM time:  %f\n", (endTime - startTime));
#endif

#if(RUN_CSR == 1)
	startTime = omp_get_wtime();
	
	cusp::multiply(CSR_matA, CSR_matP, CSR_matAP);
	safeSync();
	// cusp::multiply(CSR_matR, CSR_matAP, CSR_matRAP);
	// safeSync();

	endTime = omp_get_wtime();
	fprintf(stderr, "CSR SpMM time:  %f\n", (endTime - startTime));
#endif

#endif

	//check final results
#if(RUN_CSR == 1 && RUN_DCSR == 1)
	//CheckMatrices(DCSR_matRAP, CSR_matRAP);
#endif
	
	destroyStreams();
}

#endif