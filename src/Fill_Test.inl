void FillTests(const std::string &filename)
{
	//set cuda device
	cudaSetDevice(1);

	#if(PRECISION == 32)
		#define PTYPE	float
	#elif(PRECISION == 64)
		#define PTYPE	double
	#endif

	int mat_rows, mat_cols;
	CuspVectorInt_h rows_h, cols_h;
	CuspVectorInt_d rows_d, cols_d;
#if(PRECISION == 32)
	CuspVectorS_h vals_h, x_vec_h, y1_vec_h, y2_vec_h, y3_vec_h;
	CuspVectorS_d vals_d, x_vec_d, y_vec_d;
	fprintf(stderr, "PRECISION: SINGLE\n");
#elif(PRECISION == 64)
	CuspVectorD_h vals_h, x_vec_h, y1_vec_h, y2_vec_h, y3_vec_h;
	CuspVectorD_d vals_d, x_vec_d, y_vec_d;
	fprintf(stderr, "PRECISION: DOUBLE\n");
#endif

	std::vector< std::vector< std::pair<int,PTYPE> > > GS_mat;
	int NNZ = ReadMatrixFile(filename, GS_mat, rows_h, cols_h, vals_h, mat_rows, mat_cols);
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

	const int BINS = MAX_OFFSET;
	int ell_width = avg_row;
	int bin_size = ell_width;
	int overflow_size = NNZ;//mat_rows*ell_width*(BINS/2);
	
	//fprintf(stderr, "alpha: %f\n", DCSR_matA.alpha);
	fprintf(stderr, "rows: %d  cols: %d\n", mat_rows, mat_cols);
	fprintf(stderr, "ell width: %d\n", ell_width);
	fprintf(stderr, "bin_size: %d\n", ell_width);
	fprintf(stderr, "overflow memsize: %d\n", overflow_size);
	fprintf(stderr, "VECTOR_SIZE: %d\n", __VECTOR_SIZE);

	//Setup and initialize matrices
#if(RUN_CSR == 1)
	cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_mat;
	CSR_mat.resize(mat_rows, mat_cols, NNZ);
#endif
#if(RUN_HYB == 1)
	hyb_matrix<int, PTYPE, cusp::device_memory> HYB_mat;
	HYB_mat.resize(mat_rows, mat_cols, min(overflow_size, (int)ALIGN_UP(NNZ,32)), ell_width);
#endif
#if(RUN_DCSR == 1)
	dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_matA;
	DCSR_matA.resize(mat_rows, mat_cols, bin_size);
	device::Initialize_Matrix(DCSR_matA);
#endif
	//initialize streams
	createStreams();
	//cudaPrintfInit();

	//generate random vector
	x_vec_h.resize(mat_cols);
	for(int i=0; i<mat_cols; ++i)
		x_vec_h[i] = double(rand()) / RAND_MAX;

	x_vec_d = x_vec_h;
	//sort vectors by row
	cusp::sort_by_row(rows_h, cols_h, vals_h);

	//half size vectors
	rows_d = rows_h;
	cols_d = cols_h;
	vals_d = vals_h;

	#if(RUN_DCSR == 1)
	LoadMatrix(DCSR_matA, rows_d, cols_d, vals_d, NNZ);
	//LoadMatrix(DCSR_matA, CSR_mat_d);
	//LoadMatrix(CSR_mat_d, DCSR_matA);
	#endif
	#if(RUN_HYB == 1)
	//LoadMatrix(HYB_mat, rows_d, cols_d, vals_d, NNZ);
	//LoadMatrix(HYB_mat, CSR_mat_d);
	#endif
	#if(RUN_CSR == 1)
	LoadMatrix(CSR_mat, rows_d, cols_d, vals_d, NNZ);
	//LoadMatrix(CSR_mat, CSR_mat_d);
	#endif

	/****************************************************************************/
	//Free memory
	rows_d.clear();
	cols_d.clear();
	vals_d.clear();
	rows_d.shrink_to_fit();
	cols_d.shrink_to_fit();
	vals_d.shrink_to_fit();
	/****************************************************************************/
	
	//SPMV tests
	#define TEST_COUNT	100
	double startTime, endTime;

#if(SPMV_TEST == 1)

#if(RUN_DCSR == 1)
	// y_vec_d.resize(mat_rows, 0);
	// startTime = omp_get_wtime();
	// for(int i=0; i<TEST_COUNT; i++)
	// {
	// 	device::spmv(DCSR_matA, x_vec_d, y_vec_d, __streams);
	// }
	// safeSync();
	// endTime = omp_get_wtime();
	// //cudaPrintfDisplay(stdout, true);
	// fprintf(stderr, "DCSR matrix SpMV time:  %f\n", (endTime - startTime));
	// y1_vec_h = y_vec_d;

	device::BinRows(DCSR_matA);
	startTime = omp_get_wtime();
	device::SortMatrixRow(DCSR_matA, __streams);
	safeSync();
	endTime = omp_get_wtime();
	fprintf(stderr, "DCSR sort time:  %f\n", (endTime - startTime));

	int mem_pos = DCSR_matA.row_sizes[mat_rows];
	fprintf(stderr, "DCSR_matA mem_pos: %d\n", mem_pos);

	y_vec_d.resize(mat_rows, 0);
	startTime = omp_get_wtime();
	for(int i=0; i<TEST_COUNT; i++)
	{
		device::spmv(DCSR_matA, x_vec_d, y_vec_d, __streams);
	}
	safeSync();
	endTime = omp_get_wtime();
	//cudaPrintfDisplay(stdout, true);
	fprintf(stderr, "DCSR sorted matrix SpMV time:  %f\n", (endTime - startTime));
	y1_vec_h = y_vec_d;
#endif

#if(RUN_HYB == 1)
	y_vec_d.resize(mat_rows, 0);
	startTime = omp_get_wtime();
	for(int i=0; i<TEST_COUNT; i++)
	{
		cusp::multiply(HYB_mat.matrix, x_vec_d, y_vec_d);
	}
	safeSync();
	endTime = omp_get_wtime();
	fprintf(stderr, "Hybrid matrix SpMV time:  %f\n", (endTime - startTime));
	y2_vec_h = y_vec_d;
#endif

#if(RUN_CSR == 1)
	y_vec_d.resize(mat_rows, 0);
	startTime = omp_get_wtime();
	for(int i=0; i<TEST_COUNT; i++)
	{
		//cusp::multiply(CSR_mat, x_vec_d, y_vec_d);
		device::spmv(CSR_mat, x_vec_d, y_vec_d);
	}
	safeSync();
	endTime = omp_get_wtime();
	fprintf(stderr, "CSR matrix SpMV time:  %f\n", (endTime - startTime));
	y3_vec_h = y_vec_d;
#endif

#endif

	//ADD tests
#if(RUN_ADD == 1)
	//setup additional entries (10% of nonzeros from original matrix)
	int num_iters = 5;
	int num_slices = 50;
	int slice = NNZ / 500;
	double total_time = 0;
	double add_time = 0;
	double spmv_time = 0;

	rows_h.resize(slice);
	cols_h.resize(slice);
	vals_h.resize(slice);
	fprintf(stderr, "slice size: %d\n", slice);

	CuspVectorInt_d Trows_d[num_slices], Tcols_d[num_slices];
#if(PRECISION == 32)
	CuspVectorS_d Tvals_d[num_slices];
#elif(PRECISION == 64)
	CuspVectorD_d Tvals_d[num_slices];
#endif

	for(int n=0; n<num_slices; n++)
	{
		for(int i=0; i<slice; ++i)
		{
			rows_h[i] = rand() % mat_rows;
			cols_h[i] = rand() % mat_cols;
			vals_h[i] = double(rand()) / RAND_MAX;
		}
		
		Trows_d[n] = rows_h;
		Tcols_d[n] = cols_h;
		Tvals_d[n] = vals_h;

		cusp::detail::sort_by_row(Trows_d[n], Tcols_d[n], Tvals_d[n]);
	}

	y_vec_d.resize(mat_rows, 0);

	//add new entries in
	#if(RUN_CSR == 1)
		cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_matB, CSR_matC;
		CSR_matB.resize(mat_rows, mat_cols, slice);

		total_time = 0;
		//startTime = omp_get_wtime();
		for(int i=0; i<num_slices; i++)
		{
			startTime = omp_get_wtime();
			device::LoadMatrix(CSR_matB, Trows_d[i], Tcols_d[i], Tvals_d[i], slice);
			cusp::add(CSR_mat, CSR_matB, CSR_matC);
			CSR_mat = CSR_matC;
			safeSync();
			endTime = omp_get_wtime();
			//fprintf(stderr, "update time:  %f\n", (endTime - startTime));
			add_time += (endTime - startTime);

			startTime = omp_get_wtime();
			for(int j=0; j<num_iters; j++)
				device::spmv(CSR_mat, x_vec_d, y_vec_d);
			safeSync();
			endTime = omp_get_wtime();
			// fprintf(stderr, "spmv time:  %f\n", (endTime - startTime));
			spmv_time += (endTime - startTime);
		}
		safeSync();
		//endTime = omp_get_wtime();
		fprintf(stderr, "Matrix ADD time:  %f\n", add_time);
		fprintf(stderr, "Matrix spmv time:  %f\n", spmv_time);
		fprintf(stderr, "Matrix iter total time:  %f\n", spmv_time + add_time);
	#endif

	#if(RUN_HYB == 1)
		total_time = 0;
		//startTime = omp_get_wtime();
		for(int i=0; i<num_slices; i++)
		{
			startTime = omp_get_wtime();
			device::UpdateMatrix(HYB_mat, Trows_d[i], Tcols_d[i], Tvals_d[i], slice);
			safeSync();
			endTime = omp_get_wtime();
			//fprintf(stderr, "update time:  %f\n", (endTime - startTime));
			add_time += (endTime - startTime);

			startTime = omp_get_wtime();
			for(int j=0; j<num_iters; j++)
				cusp::multiply(HYB_mat.matrix, x_vec_d, y_vec_d);
			safeSync();
			endTime = omp_get_wtime();
			//fprintf(stderr, "spmv time:  %f\n", (endTime - startTime));
			spmv_time += (endTime - startTime);
		}
		safeSync();
		//endTime = omp_get_wtime();
		fprintf(stderr, "Matrix ADD time:  %f\n", add_time);
		fprintf(stderr, "Matrix spmv time:  %f\n", spmv_time);
		fprintf(stderr, "Matrix iter total time:  %f\n", spmv_time + add_time);
	#endif

	#if(RUN_DCSR == 1)
		total_time = 0;
		//startTime = omp_get_wtime();
		for(int i=0; i<num_slices; i++)
		{
			fprintf(stderr, "iter: %d\n", i);

			startTime = omp_get_wtime();
			device::UpdateMatrix(DCSR_matA, Trows_d[i], Tcols_d[i], Tvals_d[i], true);
			safeSync();
			endTime = omp_get_wtime();
			//fprintf(stderr, "update time:  %f\n", (endTime - startTime));
			add_time += (endTime - startTime);

			if(i > 0 && i % 5 == 0)
			{
				startTime = omp_get_wtime();
				//device::BinRows(DCSR_matA);
				device::SortMatrixRow(DCSR_matA, __streams);
				safeSync();
				endTime = omp_get_wtime();
				//fprintf(stderr, "sort time:  %f\n", (endTime - startTime));
				add_time += (endTime - startTime);
			}

			startTime = omp_get_wtime();
			for(int j=0; j<num_iters; j++)
				device::spmv(DCSR_matA, x_vec_d, y_vec_d, __streams);
			safeSync();
			endTime = omp_get_wtime();
			//fprintf(stderr, "spmv time:  %f\n", (endTime - startTime));
			spmv_time += (endTime - startTime);
		}
		safeSync();
		//endTime = omp_get_wtime();
		fprintf(stderr, "Matrix ADD time:  %f\n", add_time);
		fprintf(stderr, "Matrix spmv time:  %f\n", spmv_time);
		fprintf(stderr, "Matrix iter total time:  %f\n", spmv_time + add_time);
	#endif

#endif

	destroyStreams();
}