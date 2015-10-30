#include "CFA.h"

#define DEBUG		1
//#define OMP

#if(DEBUG == 1)
#define MV_MULTIPLY(A, B, C)	cusp::multiply(A, B, C);\
								fprintf(stdout, "(%dx%d) * (%dx1) -> (%dx1)\n", A.num_rows, A.num_cols, B.size(), C.size())

#define DEBUG_PRINT(A, B)		fprintf(stdout, A);\
								cusp::print(B);
#else
#define MULTIPLY(A, B, C)		cusp::multiply(A, B, C)
#define DEBUG_PRINT(A, B)
#endif

extern void Test(std::string filename);

#include "host_forms.inl"
#include "device_forms.inl"

// template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
// inline void subtract_vec(cusp::array1d<INDEX_TYPE, MEM_TYPE> &A,
// 							cusp::array1d<INDEX_TYPE, MEM_TYPE> &B,
// 							cusp::array1d<INDEX_TYPE, MEM_TYPE> &C)
// {
// 	thrust::transform(A.begin(), A.end(), B.begin(), C.begin(), thrust::minus<INDEX_TYPE>());
// }

// General GPU Device CUDA Initialization
// int gpuDeviceInit(int devID)
// {
//     int deviceCount;
//     checkCudaErrors(cudaGetDeviceCount(&deviceCount));

//     if (deviceCount == 0)
//     {
//         fprintf(stdout, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
//         exit(-1);
//     }

//     if (devID < 0)
//        devID = 0;
        
//     if (devID > deviceCount-1)
//     {
//         fprintf(stdout, "\n");
//         fprintf(stdout, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
//         fprintf(stdout, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
//         fprintf(stdout, "\n");
//         return -devID;
//     }

//     cudaDeviceProp deviceProp;
//     checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );

//     if (deviceProp.major < 1)
//     {
//         fprintf(stdout, "gpuDeviceInit(): GPU device does not support CUDA.\n");
//         exit(-1);
//     }
    
//     checkCudaErrors( cudaSetDevice(devID) );
//     checkCudaErrors( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );
//     printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

//     return devID;
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::Init_CPU()
{
#if BUILD_TYPE == CPU
	size_t vec_size = sigma.num_rows;
	VOID_vec.resize(vec_size, 0);
	NOT_FALSE_vec.resize(vec_size, 1);
	FALSE_vec.resize(vec_size, 0);
	BOOL_vec.resize(vec_size, 0);
	NUM_vec.resize(vec_size, 0); 
	LIST_vec.resize(vec_size, 0);
	tb.resize(1, 0);
	fb.resize(1, 0);
	AND_vec1.resize(r.size(), 0);
	AND_vec2.resize(r.size(), 0);

	for(int i=0; i<ARG_MAX; ++i)
	{
		v[i].resize(vec_size, 0);
		a[i].resize(sigma.num_cols, 0);
		
		if(Call[i].size() != r.size())
			Call[i].resize(r.size(), 0);
		if(PrimList[i].size() != r.size())
			PrimList[i].resize(r.size(), 0);
	}

	a_var.resize(sigma.num_cols, 0);
	vf.resize(vec_size, 0);
    a_set.resize(sigma.num_cols, 0);
    v_set.resize(vec_size, 0);
    v_cond.resize(vec_size, 0);
    v_list.resize(vec_size, 0);

	for(int i=0; i<vec_size; ++i)
	{
		if(i == vec_size - 5)			//list
			LIST_vec[i] = 1;			
		else if(i == vec_size - 4)		//void
			VOID_vec[i] = 1;
		else if(i == vec_size - 3)		//#t
			BOOL_vec[i] = 1;
		else if(i == vec_size - 2)		//#f
		{
			NOT_FALSE_vec[i] = 0;
			BOOL_vec[i] = 1;
			FALSE_vec[i] = 1;
		}
		else if(i == vec_size - 1)		//NUM
			NUM_vec[i] = 1;
	}

	s.resize(r.size(), 0);
	s_indices.resize(r.size(), 0);
#endif
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::Init_GPU()
{
#if BUILD_TYPE == GPU
	cudaPrintfInit();

	size_t vec_size = sigma.num_rows;
	VOID_vec.resize(vec_size, 0);
	NOT_FALSE_vec.resize(vec_size, 1);
	FALSE_vec.resize(vec_size, 0);
	BOOL_vec.resize(vec_size, 0);
	NUM_vec.resize(vec_size, 0); 
	LIST_vec.resize(vec_size, 0);
	tb.resize(1, 0);
	fb.resize(1, 0);
	AND_vec1.resize(r.size(), 0);
	AND_vec2.resize(r.size(), 0);

	for(int i=0; i<ARG_MAX; ++i)
	{
		v[i].resize(vec_size, 0);
		a[i].resize(sigma.num_cols, 0);
		
		if(Call[i].size() != r.size())
			Call[i].resize(r.size(), 0);
		if(PrimList[i].size() != r.size())
			PrimList[i].resize(r.size(), 0);
	}

	for(int i=0; i<NUM_STREAMS; ++i)
	{
		a_var[i].resize(sigma.num_cols, 0);
	    vf[i].resize(vec_size, 0);
	}
    a_set.resize(sigma.num_cols, 0);
    v_set.resize(vec_size, 0);
    v_cond.resize(vec_size, 0);
    v_list.resize(vec_size, 0);

	for(int i=0; i<vec_size; ++i)
	{
		if(i == vec_size - 5)			//list
			LIST_vec[i] = 1;			
		else if(i == vec_size - 4)		//void
			VOID_vec[i] = 1;
		else if(i == vec_size - 3)		//#t
			BOOL_vec[i] = 1;
		else if(i == vec_size - 2)		//#f
		{
			NOT_FALSE_vec[i] = 0;
			BOOL_vec[i] = 1;
			FALSE_vec[i] = 1;
		}
		else if(i == vec_size - 1)		//NUM
			NUM_vec[i] = 1;
	}

    a_indices.resize(sigma.num_cols, 0);
	v_indices.resize(vec_size, 0);
	for(int i=0; i<NUM_STREAMS; ++i)
	{
		index_count[i].resize(8, 0);
		temp_row_indices[i].resize(vec_size, 0);
		temp_col_indices[i].resize(sigma.num_cols, 0);
		s[i].resize(r.size(), 0);
		s_indices[i].resize(r.size(), 0);
		Fun_vec[i].resize(Fun.num_rows, 0);
		Body_vec[i].resize(Body.num_rows, 0);
		Arg_vec[i].resize(Arg[0].num_rows, 0);
		accum_var_vec[i].resize(sigma.num_cols);
		accum_vf_vec[i].resize(vec_size);

		update_queue[i].resize(QUEUE_SIZE);
	}
	Cond_vec.resize(CondTrue.num_rows);

	cudaStreamCreate(&stream_Call);
    cudaStreamCreate(&stream_List);
    cudaStreamCreate(&stream_Set);
    cudaStreamCreate(&stream_If);
    cudaStreamCreate(&stream_Num);
    cudaStreamCreate(&stream_Bool);
    cudaStreamCreate(&stream_Void);
#endif
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::ReadTestFile(const char* filename)
{
	int ID = omp_get_thread_num();

	std::ifstream tf(filename);
	if(tf.fail())
	{
		fprintf(stdout, "Invalid test file: %s\n", filename);
		exit(1);
	}

	fprintf(stdout, "Reading test file: %s\n", filename);
	while(!tf.eof())
	{
		int rows, cols, i, j;
		char buf[64], name[32];
		valid_Call.resize(ARG_MAX);
		valid_List.resize(ARG_MAX);

		tf.getline(buf, 64);
		if(tf.gcount() > 1)
		{
			std::sscanf(buf, "%s %d %d", name, &rows, &cols);

			std::vector< std::pair<int, int> > indices;
			tf.getline(buf, 64);
			while(tf.gcount() > 1)
			{
				std::sscanf(buf, "%d %d", &i, &j);
				if(i<0 || i>=rows || j<0 || j>=cols)
					fprintf(stdout, "ERROR Rows: %d Cols: %d \t (i:%d j:%d)\n", rows, cols, i, j);

				indices.push_back(std::make_pair(i,j));
				tf.getline(buf, 64);
			}
			std::sort(indices.begin(), indices.end(), compare_entry);

			//check name
			std::string mat_name(name), sname = "", snum = "";
			int mat_num = -1;
			for(int i=0; i<NUM_MATRIX_TYPES; ++i)
			{
				std::string find_name = MatrixTypeMap[i];
				if(mat_name.find(find_name) == 0)
				{
					sname = find_name;
					snum = mat_name.substr(sname.size());
					if(snum.size() > 0)
						mat_num = atoi(snum.c_str());
					break;
				}
			}

			cusp::coo_matrix<int, int, cusp::host_memory> A(rows, cols, indices.size());
			cusp::coo_matrix<int, int, cusp::host_memory> B(cols, rows, indices.size());

			for(int i=0; i<indices.size(); ++i)
			{
				A.row_indices[i] = indices[i].first;
				A.column_indices[i] = indices[i].second;
				A.values[i] = 1;
			}
			A.sort_by_row_and_column();
			cusp::transpose(A, B);
			B.sort_by_row_and_column();

			cusp::array1d<int, cusp::host_memory> vec;
			if(A.num_cols == 1)
			{
				vec.resize(A.num_rows, 0);
				for(int i=0; i<A.num_entries; ++i)
				{
					vec[A.row_indices[i]] = 1;
				}
			}

			if(ID == 0 && A.num_entries > 0)
			{
				fprintf(stderr, "\n%s (%d x %d) with %d entries\n", name, A.num_rows, A.num_cols, A.num_entries);
				fprintf(stderr, "B: (%d x %d)\n", B.num_rows, B.num_cols);
			}

			//parse name
			if(sname == "r")
				r = vec;
			else if(sname == "sigma")
			{
#if BUILD_TYPE == CPU
				sigma = B;
				if(ID == 0)
					print_matrix_info(sigma);
#else			
				if(ID == 0)
				{
					size_t entry_count_size = 32*sizeof(INDEX_TYPE);
					checkCudaErrors( cudaHostAlloc((void **)&entry_count_host, entry_count_size, 0));
					checkCudaErrors( cudaMalloc((void **)&entry_count_device, entry_count_size));
					memset(entry_count_host, 0, entry_count_size);
					checkCudaErrors( cudaMemcpy(entry_count_device, entry_count_host, entry_count_size, cudaMemcpyHostToDevice) );
				}

				#pragma omp barrier
				cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> temp;
				temp = B;
				LoadMatrix(temp, sigma);
				sigma.num_entries = CountEntries(sigma);

				if(ID == 0)
					print_matrix_info(sigma);
#endif
			}
			else if(sname == "CondTrue")
			{
				CondTrue = B;
				if(ID == 0)
					print_matrix_info(CondTrue);
			}
			else if(sname == "CondFalse")
			{
				CondFalse = B;
				if(ID == 0)
					print_matrix_info(CondFalse);
			}
			else if(sname == "Body")
			{
				Body = B;
				if(ID == 0)
					print_matrix_info(Body);
			}
			else if(sname == "Fun")
			{
				Fun = B;
				if(ID == 0)
					print_matrix_info(Fun);
			}
			else if(sname == "Arg")
			{
				Arg[mat_num-1] = B;
				if(ID == 0)
					print_matrix_info(Arg[mat_num-1]);
			}
			else if(sname == "Var")
			{
				Var[mat_num-1] = B;
				if(ID == 0)
					print_matrix_info(Var[mat_num-1]);
			}
			else if(sname == "Call")
			{
				if(A.num_entries > 0)
					valid_Call[mat_num] = true;
				else
					valid_Call[mat_num] = false;

				Call[mat_num] = vec;
				if(m_maxCall < mat_num)
					m_maxCall = mat_num;
			}
			else if(sname == "PrimBool")
				PrimBool = vec;
			else if(sname == "PrimNum")
				PrimNum = vec;
			else if(sname == "PrimVoid")
				PrimVoid = vec;
			else if(sname == "PrimList")
			{
				if(A.num_entries > 0)
					valid_List[mat_num] = true;
				else
					valid_List[mat_num] = false;

				PrimList[mat_num] = vec;
				if(m_maxList < mat_num)
					m_maxList = mat_num;
			}
			else if(sname == "If")
				IF = vec;
			else if(sname == "Set")
				SET = vec;
			else
				fprintf(stdout, "could not match input matrix: %s\n", name);
		}
	}

	tf.close();
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::Run_Analysis()
{
	int ID = omp_get_thread_num();
	fprintf(stdout, "\n\n\nStarting analysis: %d\n", ID);

	int prev_num_entries = sigma.num_entries;
	r_prime = r;
	int iter=0;
	bool sigma_change = true, r_change = true;
	fprintf(stderr, "m_maxCall: %d  m_maxList: %d\n", m_maxCall, m_maxList);
	#define ITER_COUNT	5

	#pragma omp parallel num_threads(NUM_STREAMS)
	do
	{
		// int ID = omp_get_thread_num();
		// fprintf(stderr, "thread ID: %d\n", ID);
		
		// if(ID == 0)
		// {
		 	iter++;
		 	fprintf(stdout, "\n\nITERATION %d\n\n", iter);
		// }

		int ID = omp_get_thread_num();
		fprintf(stderr, "thread ID: %d\n", ID);
		
		if(ID == 0)
		{	
			iter++;
			fprintf(stdout, "\n\nITERATION %d\n\n", iter);
		}

		//if(ID == 1)
			f_call();
		//if(ID == 2)
		 	f_list();
		//if(ID == 3)
			f_set();
		//if(ID == 4)
			f_if();
		//if(ID == 5)
			f_primBool();
		//if(ID == 6)
			f_primNum();
		//if(ID == 7)
			f_primVoid();
		//if(ID == 8)
			f_UpdateStore();

		if(ID == 0 && iter % ITER_COUNT == 0)
		{
			fprintf(stdout, "\nupdate sigma\n");
		#if BUILD_TYPE == GPU
			sigma.num_entries = CountEntries(sigma);

			if(prev_num_entries != sigma.num_entries)
				sigma_change = true;
			else
				sigma_change = false;
			prev_num_entries = sigma.num_entries;

			fprintf(stdout, "\nupdate r\n");
			int r_entries = thrust::count(r.begin(), r.end(), 1);
			int r_prime_entries = thrust::count(r_prime.begin(), r_prime.end(), 1);

			if(r_entries != r_prime_entries)
				r_change = true;
			else
				r_change = false;

			int val = sigma.matrix.coo.column_indices[0];
			fprintf(stderr, "coo entries: %d\n", val);

			//cusp::print(sigma.matrix.ell);
			cusp::print(update_queue[STREAM_VOID]);

			if(prev_num_entries != sigma.num_entries)
				sigma_change = true;
			else
				sigma_change = false;
			prev_num_entries = sigma.num_entries;

			fprintf(stdout, "\nupdate r\n");
			int r_entries = thrust::count(r.begin(), r.end(), 1);
			int r_prime_entries = thrust::count(r_prime.begin(), r_prime.end(), 1);

			if(r_entries != r_prime_entries)
				r_change = true;
			else
				r_change = false;

			int val = sigma.coo.column_indices[0];
			fprintf(stderr, "coo entries: %d\n", val);

			//DEBUG_PRINT("r: ", r);
			//DEBUG_PRINT("r_prime: ", r_prime);
			r = r_prime;
			fprintf(stderr, "sigma.num_entries: %d\n", sigma.num_entries);
		#else
			//sigma.num_entries = thrust::count_if(sigma.column_indices.values.begin(), sigma.column_indices.values.end(), is_non_negative());
			//thrust::fill(sigma.values.begin(), sigma.values.end(), 1);
			//cusp::print(sigma);

			if(prev_num_entries != sigma.num_entries)
				sigma_change = true;
			else
				sigma_change = false;
			prev_num_entries = sigma.num_entries;

			fprintf(stdout, "\nupdate r\n");
			int r_entries = thrust::count(r.begin(), r.end(), 1);
			int r_prime_entries = thrust::count(r_prime.begin(), r_prime.end(), 1);

			if(r_entries != r_prime_entries)
				r_change = true;
			else
				r_change = false;

			//DEBUG_PRINT("r: ", r);
			r = r_prime;
			fprintf(stderr, "sigma.num_entries: %d\n", sigma.num_entries);
		#endif
		}

	#pragma omp barrier
	} while(r_change || sigma_change);

	fprintf(stdout, "Analysis Complete...\n");
}


template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::WriteStore()
{
	cusp::coo_matrix<int, VALUE_TYPE, cusp::host_memory> store;
	CopyStore(sigma, store);
	fprintf(stderr, "copy complete\n");

	std::ofstream output("tests/output.dat");
	output << "sigma " << store.num_rows << " " << store.num_cols << std::endl;
	for(int i=0; i<store.num_entries; ++i)
	{
		output << store.row_indices[i] << " " << store.column_indices[i] << " " << std::endl;
	}

	output.close();
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::CopyStore(	cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat,
														cusp::coo_matrix<int, VALUE_TYPE, cusp::host_memory> &store)
{
#if BUILD_TYPE == GPU
	cusp::coo_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> temp;
	temp.resize(mat.num_rows, mat.num_cols, mat.num_entries);

    int offset = 0;
    for(int row=0; row<mat.num_rows; ++row)
    {
        INDEX_TYPE row_start = mat.row_offsets[row];
        INDEX_TYPE row_end = mat.row_offsets[row+1];

        for(int n=row_start; n < row_end; ++n, ++offset)
        {
            temp.row_indices[offset] = row;
            temp.column_indices[offset] = mat.column_indices[n];
        }
    }

    cusp::transpose(temp, store);
#else
	cusp::transpose(mat, store);
#endif
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::CopyStore(	hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat,
														cusp::coo_matrix<int, VALUE_TYPE, cusp::host_memory> &store)
{
#if BUILD_TYPE == GPU
	cusp::coo_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> temp;
	temp.resize(mat.num_rows, mat.num_cols, mat.num_entries);

	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::invalid_index;

    int offset = 0;
    for(int row=0; row < mat.num_rows; ++row)
    {
    	int num_cols_per_row = mat.matrix.ell.column_indices.num_cols;
    	int pitch = mat.matrix.ell.column_indices.pitch;

    	int offset = 0;
        for(int n=row; n < num_cols_per_row; n+=pitch)
        {
        	INDEX_TYPE val = mat.matrix.ell.column_indices.values[n];
        	if(val != invalid_index)
        	{
	        	temp.row_indices[offset] = row;
    	        temp.column_indices[offset] = mat.matrix.ell.column_indices.values[n];
    	        offset++;
    	    }
        }
    }

    int coo_size = mat.matrix.coo.column_indices[0];
    for(int n=1; n <= coo_size; n++, offset++)
    {
    	temp.row_indices[offset] = mat.matrix.coo.row_indices[n];
    	temp.column_indices[offset] = mat.matrix.coo.column_indices[n];
    }

        for(int n=row; n < num_cols_per_row; n+=pitch, offset++)
        {
        	temp.row_indices[offset] = row;
            temp.column_indices[offset] = mat.matrix.ell.column_indices.values[n];
        }
    }

    cusp::transpose(temp, store);
#else
	cusp::transpose(mat.matrix, store);
#endif
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::CopyStore(	dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat,
														cusp::coo_matrix<int, VALUE_TYPE, cusp::host_memory> &store)
{
	cusp::coo_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> temp;
	temp.resize(mat.num_rows, mat.num_cols, mat.num_entries);

    int offset = 0;
    for(int row=0; row<mat.num_rows; ++row)
    {
        INDEX_TYPE row_start = (*mat.row_offsets)[row];
        INDEX_TYPE row_end = (*mat.row_offsets)[row+1];
        INDEX_TYPE row_size = mat.row_sizes[row];

        for(int n=0; n < row_size; ++n, ++offset)
        {
        	INDEX_TYPE col = (*mat.column_indices)[row_start + n];
            temp.row_indices[offset] = row;
            temp.column_indices[offset] = col;
        }
    }

    cusp::transpose(temp, store);
}

void Test(std::string filename)
{
	double startTime, endTime;
#if BUILD_TYPE == CPU
	CFA<int, int, cusp::host_memory> Analysis;

	Analysis.ReadTestFile(filename.c_str());
	Analysis.Init_CPU();

	startTime = omp_get_wtime();
	Analysis.Run_Analysis();
	endTime = omp_get_wtime();

	fprintf(stdout, "Run Time: %f seconds\n", endTime - startTime);
	Analysis.WriteStore();

#elif BUILD_TYPE == GPU
	//#pragma omp parallel num_threads(NUM_GPUS)
	{
		int ID = omp_get_thread_num();
		gpuDeviceInit(1);
		fprintf(stderr, "thread ID: %d\n", ID);
		CFA<int, int, cusp::device_memory> Analysis;

		Analysis.ReadTestFile(filename.c_str());
		Analysis.Init_GPU();

	//#pragma omp barrier

		if(ID == 0)
			startTime = omp_get_wtime();
		
		Analysis.Run_Analysis();

		if(ID == 0)
		{
			endTime = omp_get_wtime();
			fprintf(stdout, "Run Time: %f seconds\n", endTime - startTime);
			Analysis.WriteStore();
		}
		cudaPrintfEnd();
	}
#endif
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
unsigned int CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::CountEntries(cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory> &mat)
{
	cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> temp(mat);
	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, INDEX_TYPE, cusp::device_memory>::invalid_index;

	int num_entries = 0;
	int pitch = temp.column_indices.pitch;
	for(int col=0; col<temp.column_indices.num_cols; ++col)
	{
		int offset = pitch*col;
		for(int row=0; row<temp.num_rows; ++row, ++offset)
		{
			if(temp.column_indices.values[offset] != invalid_index)
				num_entries++;
		}
	}
	temp.num_entries = num_entries;

	return num_entries;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
unsigned int CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::CountEntries(hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat)
{
    INDEX_TYPE val = sigma.matrix.coo.column_indices[0];
    fprintf(stderr, "val: %d\n", val);
	return thrust::reduce(sigma.row_sizes.begin(), sigma.row_sizes.end()) + val;

    //INDEX_TYPE val = mat.matrix.coo.column_indices[0];
    cusp::hyb_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> temp = mat.matrix;
	unsigned int val = mat.num_entries = thrust::count_if(mat.matrix.ell.column_indices.values.begin(), mat.matrix.ell.column_indices.values.end(), is_non_negative());
	fprintf(stderr, "count entries: %d\n", val);
	return val;
}

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
unsigned int CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::CountEntries(dell_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> &mat)
{
	INDEX_TYPE val = sigma.coo.column_indices[0];
	return thrust::reduce(sigma.row_sizes.begin(), sigma.row_sizes.end()) + val;
}
