#ifndef FILL_TEST_H
#define FILL_TEST_H

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
	const int width = 4;
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
	cusp::detail::sort_by_row(rows_h, cols_h, vals_h);

	//CSR load TESTS ***
	// cusp::csr_matrix<int, PTYPE, cusp::device_memory> CSR_mat_d;
	// CSR_mat_d.resize(mat_rows, mat_cols, NNZ);

	// CSR_mat_d.column_indices = cols_h;
	// CSR_mat_d.values = vals_h;
	// rows_d = rows_h;
	// cusp::detail::indices_to_offsets(rows_d, CSR_mat_d.row_offsets);
	// rows_d.resize(0);
	// rows_d.shrink_to_fit();
	//CSR load TESTS ***

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

	// double dprod = 0.0;
	// for(int row=0; row<mat_rows; ++row)
	// {
	// 	float sum = 0;
	// 	for(int j=0; j<GS_mat[row].size(); j++)
	// 		sum += GS_mat[row][j].second * x_vec_h[GS_mat[row][j].first];

	// 	//fabs(y2_vec_h[row] - sum) > 1e-3 ||
	// 	if( fabs(y1_vec_h[row] - sum) > 1e-3)// ||  fabs(y3_vec_h[row] - sum) > 1e-3 )
	// 	{
	// 		//fprintf(stderr, "ERROR   %d:  %15f\t\t%15f\t\t%15f\t\t%15f\n", row, y1_vec_h[row], y2_vec_h[row], y3_vec_h[row], sum);
	// 		//fprintf(stderr, "ERROR   %d:  %15f\t\t%15f\t\t%15f\n", row, y1_vec_h[row], y3_vec_h[row], sum);
	// 		fprintf(stderr, "ERROR   %d:  %15f\t\t%15f\n", row, y1_vec_h[row], sum);
	// 	}
	// 	dprod += sum*sum;
	// }
	//fprintf(stderr, "dot product of result: %f\n", dprod);


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

/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/

void FillTests_Multi(const std::string &filename)
{
	int mat_rows, mat_cols;
	CuspVectorInt_h rows_multi_h, cols_multi_h;
	#if(PRECISION == 32)
		CuspVectorS_h x_vec_h, y1_vec_h, y2_vec_h, y3_vec_h;
		CuspVectorS_h vals_multi_h;
	#elif(PRECISION == 64)
		CuspVectorD_h x_vec_h, y1_vec_h, y2_vec_h, y3_vec_h;
		CuspVectorD_h vals_multi_h;
	#endif

	std::vector< std::vector< std::pair<int,PTYPE> > > GS_mat;
	int NNZ = ReadMatrixFile(filename, GS_mat, rows_multi_h, cols_multi_h, vals_multi_h, mat_rows, mat_cols);
	if(NNZ == 0)
		exit(1);

	//sort vectors by row
	cusp::detail::sort_by_row(rows_multi_h, cols_multi_h, vals_multi_h);
	CuspVectorInt_h row_sizes_h(mat_rows);
	CuspVectorInt_h row_IDs_h(mat_rows);
	CuspVectorInt_h row_size_index_h(rows_multi_h);
	//calcuate average
	int nnz_multi[NUM_DEVICES];
	memset(nnz_multi, 0, sizeof(nnz_multi));
	int max_row = 0, nnz_count = 0, offset = 0, avg_row = 0;
	for(int i=0; i<GS_mat.size(); ++i)
	{
		int size = GS_mat[i].size();
		if(size > max_row)
			max_row = size;
		nnz_count += size;
		
		row_sizes_h[i] = size;
		row_IDs_h[i] = i;

		for(int j=0; j<size; j++, offset++)
			row_size_index_h[offset] = size;
	}
	avg_row = nnz_count / mat_rows;

	//sort through entries and organize by row sizes
	thrust::stable_sort_by_key(row_sizes_h.begin(), row_sizes_h.end(), row_IDs_h.begin());
	thrust::stable_sort_by_key(row_size_index_h.begin(), row_size_index_h.end(), 
						thrust::make_zip_iterator(thrust::make_tuple(rows_multi_h.begin(), cols_multi_h.begin(), vals_multi_h.begin())) );

	for(int i=0; i<mat_rows; i++)
	{
		int size = row_sizes_h[i];
		nnz_multi[i%NUM_DEVICES] += size;
	}
	// fprintf(stderr, "max row size: %d\n", max_row);
	// fprintf(stderr, "standard deviation: %f\n", std_dev);
	fprintf(stderr, "nnz count: %d  NNZ returned: %d  offset: %d  nnz0: %d  nnz1: %d\n", nnz_count, NNZ, offset, nnz_multi[0], nnz_multi[1]);

	const int BINS = MAX_OFFSET;
	const int width = 4;
	int ell_width = (avg_row / width + (((avg_row % width) == 0) ? 0 : 1)) * width;
	int bin_size = ell_width;
	int overflow_size = mat_rows*ell_width*(BINS/2);

	//fprintf(stderr, "alpha: %f\n", DCSR_mat.alpha);
	fprintf(stderr, "rows: %d  cols: %d\n", mat_rows, mat_cols);
	fprintf(stderr, "bin_size: %d\n", ell_width);
	fprintf(stderr, "overflow memsize: %d\n", overflow_size);
	fprintf(stderr, "VECTOR_SIZE: %d\n", VECTOR_SIZE);

	//generate random vector
	x_vec_h.resize(mat_cols);
	for(int i=0; i<mat_cols; ++i)
		x_vec_h[i] = double(rand()) / RAND_MAX;

	#pragma omp parallel num_threads(NUM_DEVICES)
	{
		//set cuda device
		double startTime, endTime;
		int ID = omp_get_thread_num();
		cudaSetDevice(ID);
		//fprintf(stderr, "ID: %d\n", ID);

		#if(PRECISION == 32)
			#define PTYPE	float
		#elif(PRECISION == 64)
			#define PTYPE	double
		#endif

		CuspVectorInt_h rows_h, cols_h;
		CuspVectorInt_d rows_d, cols_d;
	#if(PRECISION == 32)
		CuspVectorS_d vals_d, x_vec_d, y_vec_d;
		CuspVectorS_h vals_h;
	#elif(PRECISION == 64)
		CuspVectorD_d vals_d, x_vec_d, y_vec_d;
		CuspVectorD_h vals_h;
	#endif

		//initialize streams
		createStreams(ID);
		//cudaPrintfInit();

		fprintf(stderr, "Loading matrix...\n");
		int partial_rows = ALIGN_UP(mat_rows, NUM_DEVICES) / NUM_DEVICES;		//NUM_DEVICES must be power of 2
		//if(ID == NUM_DEVICES-1)
		//	partial_rows -= (mat_rows % 2);
		fprintf(stderr, "ID: %d  matrix: (%d %d)   nnz: %d\n", ID, partial_rows, mat_cols, nnz_multi[ID]);

		//Setup and initialize matrices
		dcsr_matrix<int, PTYPE, cusp::device_memory, BINS> DCSR_mat;
		//bin_size = 8;
		DCSR_mat.resize(partial_rows, mat_cols, bin_size);
		//DCSR_mat.resize(mat_rows, mat_cols, bin_size);
		device::Initialize_Matrix(DCSR_mat);

		x_vec_d = x_vec_h;
		//partial vectors
		rows_h.resize(nnz_multi[ID]);
		cols_h.resize(nnz_multi[ID]);
		vals_h.resize(nnz_multi[ID]);
		int offset = 0, local_offset = 0, row_count = 0;
		for(int i=0; i<mat_rows; i++)
		{
			int size = row_sizes_h[i];
			if(i % NUM_DEVICES == ID)
			{
				for(int j=0; j<size; j++, offset++, local_offset++)
				{
					if(local_offset >= nnz_multi[ID] || row_count >= partial_rows)
						fprintf(stderr, "ERROR** ID: %d  local_offset: %d  row_count: %d\n", ID, local_offset, row_count);

					rows_h[local_offset] = row_count;			//row indices need to be adjusted
					cols_h[local_offset] = cols_multi_h[offset];
					vals_h[local_offset] = vals_multi_h[offset];
				}
				row_count++;
			}
			else
				offset += size;
		}
		//fprintf(stderr, "ID: %d  offset: %d  nnz: %d\n", ID, offset, nnz_multi[ID]);

		//DEBUG//
		// rows_h.resize(NNZ);
		// cols_h.resize(NNZ);
		// vals_h.resize(NNZ);
		// for(int i=0; i<mat_rows; i++)
		// {
		// 	int size = row_sizes_h[i];
		// 	for(int j=0; j<size; j++, offset++)
		// 	{
		// 		rows_h[offset] = i;						//row indices need to be adjusted
		// 		cols_h[offset] = cols_multi_h[offset];
		// 		vals_h[offset] = vals_multi_h[offset];
		// 	}
		// }
		//DEBUG//
		rows_d = rows_h;
		cols_d = cols_h;
		vals_d = vals_h;

		LoadMatrix(DCSR_mat, rows_d, cols_d, vals_d, nnz_multi[ID]);

		/****************************************************************************/
		//Free memory
		rows_d.clear();
		cols_d.clear();
		vals_d.clear();
		rows_d.shrink_to_fit();
		cols_d.shrink_to_fit();
		vals_d.shrink_to_fit();
		/****************************************************************************/

		//bin rows
		device::BinRows(DCSR_mat);
		//sort rows
		startTime = omp_get_wtime();
		device::SortMatrixRow(DCSR_mat, __multiStreams[ID]);
		safeSync();
		endTime = omp_get_wtime();
		fprintf(stderr, "ID: %d  DCSR sort time:  %f\n", ID, (endTime - startTime));

		//SPMV tests
		#define TEST_COUNT_MULTI	100
		//#pragma omp barrier

	#if(SPMV_TEST == 1)
		y_vec_d.resize(partial_rows, 0);
		//y_vec_d.resize(mat_rows, 0);
		startTime = omp_get_wtime();
		for(int i=0; i<TEST_COUNT_MULTI; i++)
		{
			device::spmv(DCSR_mat, x_vec_d, y_vec_d, __multiStreams[ID]);
		}
		safeSync();
		endTime = omp_get_wtime();
		//cudaPrintfDisplay(stdout, true);
		fprintf(stderr, "ID: %d   DCSR matrix SpMV time:  %f\n", ID, (endTime - startTime));
		y1_vec_h = y_vec_d;
	#endif

		//ADD tests
	// #if(RUN_ADD == 1)
	// 	//setup additional entries (10% of nonzeros from original matrix)
	// 	int num_iters = 5;
	// 	int num_slices = 50;
	// 	int slice = NNZ / 500;
	// 	double total_time = 0;
	// 	double add_time = 0;
	// 	double spmv_time = 0;

	// 	rows_h.resize(slice);
	// 	cols_h.resize(slice);
	// 	vals_h.resize(slice);
	// 	fprintf(stderr, "slice size: %d\n", slice);

	// 	CuspVectorInt_d Trows_d[num_slices], Tcols_d[num_slices];
	// #if(PRECISION == 32)
	// 	CuspVectorS_d Tvals_d[num_slices];
	// #elif(PRECISION == 64)
	// 	CuspVectorD_d Tvals_d[num_slices];
	// #endif

	// 	for(int n=0; n<num_slices; n++)
	// 	{
	// 		for(int i=0; i<slice; ++i)
	// 		{
	// 			rows_h[i] = rand() % mat_rows;
	// 			cols_h[i] = rand() % mat_cols;
	// 			vals_h[i] = double(rand()) / RAND_MAX;
	// 		}
			
	// 		Trows_d[n] = rows_h;
	// 		Tcols_d[n] = cols_h;
	// 		Tvals_d[n] = vals_h;

	// 		cusp::detail::sort_by_row(Trows_d[n], Tcols_d[n], Tvals_d[n]);
	// 	}

	// 	y_vec_d.resize(partial_rows, 0);

	// 	#if(RUN_DCSR == 1)
	// 		total_time = 0;
	// 		//startTime = omp_get_wtime();
	// 		for(int i=0; i<num_slices; i++)
	// 		{
	// 			startTime = omp_get_wtime();
	// 			device::UpdateMatrix(DCSR_mat, Trows_d[i], Tcols_d[i], Tvals_d[i], 1);
	// 			safeSync();
	// 			endTime = omp_get_wtime();
	// 			//fprintf(stderr, "update time:  %f\n", (endTime - startTime));
	// 			add_time += (endTime - startTime);

	// 			if(i > 0 && i % 25 == 0)
	// 			{
	// 				startTime = omp_get_wtime();
	// 				device::SortMatrixRow(DCSR_mat);
	// 				device::BinRows(DCSR_mat);
	// 				safeSync();
	// 				endTime = omp_get_wtime();
	// 				//fprintf(stderr, "sort time:  %f\n", (endTime - startTime));
	// 				add_time += (endTime - startTime);
	// 			}

	// 			startTime = omp_get_wtime();
	// 			for(int j=0; j<num_iters; j++)
	// 				device::spmv(DCSR_mat, x_vec_d, y_vec_d);
	// 			safeSync();
	// 			endTime = omp_get_wtime();
	// 			//fprintf(stderr, "spmv time:  %f\n", (endTime - startTime));
	// 			spmv_time += (endTime - startTime);
	// 		}
	// 		safeSync();
	// 		//endTime = omp_get_wtime();
	// 		fprintf(stderr, "Matrix ADD time:  %f\n", add_time);
	// 		fprintf(stderr, "Matrix spmv time:  %f\n", spmv_time);
	// 		fprintf(stderr, "Matrix iter total time:  %f\n", spmv_time + add_time);
	// 	#endif

	// #endif

		//destroyStreams();
		fprintf(stderr, "destroyStreams...\n");
		destroyStreams(ID);
	}
}

#endif