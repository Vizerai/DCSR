#if(BUILD_TYPE == CPU)

//*******************************************************************************//
template <typename VALUE_TYPE>
void AND_OP(	const cusp::array1d<VALUE_TYPE, cusp::host_memory> &A, 
				const cusp::array1d<VALUE_TYPE, cusp::host_memory> &B, 
				std::vector<VALUE_TYPE> &vec)
{
	assert(A.size() == B.size());
	vec.clear();

	for(int i=0; i<A.size(); ++i)
	{	
		if(A[i] > 0 && B[i] > 0)
			vec.push_back(i);
	}
}

template <typename VALUE_TYPE>
void AccumVec(	cusp::array1d<VALUE_TYPE, cusp::host_memory> &a,
				const cusp::array1d<VALUE_TYPE, cusp::host_memory> &b)
{
	assert(a.size() == b.size());
	//a += b
	for(int i=0; i<a.size(); ++i)
		if(b[i])
			a[i] = 1;
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProduct(	const cusp::array1d<VALUE_TYPE, cusp::host_memory> &a,
					const cusp::array1d<VALUE_TYPE, cusp::host_memory> &b,
					cusp::csr_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> &mat)
{
	INDEX_TYPE num_entries_a=0, num_entries_b=0;
	std::vector<INDEX_TYPE> a_vec, b_vec;
	for(INDEX_TYPE i=0; i<a.size(); ++i)
	{
		if(a[i])
		{	
			a_vec.push_back(i);
			num_entries_a++;
		}
	}
	for(INDEX_TYPE i=0; i<b.size(); ++i)
	{
		if(b[i])
		{
			b_vec.push_back(i);
			num_entries_b++;
		}
	}

	//fprintf(stderr, "num_entries: %d %d\n", num_entries_a, num_entries_b);
	mat.resize(a.size(), b.size(), num_entries_a*num_entries_b);
	INDEX_TYPE row_offset = 0;
	for(int i=0; i<a.size(); ++i)
	{
		mat.row_offsets[i] = row_offset;
		if(a[i])
		{
			for(int j=0; j<b_vec.size(); ++j,++row_offset)
			{
				mat.column_indices[row_offset] = b_vec[j];
				mat.values[row_offset] = 1;
			}
		}
	}
	mat.row_offsets[a.size()] = row_offset;
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
void OuterProduct(	const cusp::array1d<VALUE_TYPE, cusp::host_memory> &a,
					const cusp::array1d<VALUE_TYPE, cusp::host_memory> &b,
					cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory> &mat)
{
	INDEX_TYPE num_entries_a=0, num_entries_b=0;
	std::vector<INDEX_TYPE> a_vec, b_vec;
	for(INDEX_TYPE i=0; i<a.size(); ++i)
	{
		if(a[i])
		{	
			a_vec.push_back(i);
			num_entries_a++;
		}
	}
	for(INDEX_TYPE i=0; i<b.size(); ++i)
	{
		if(b[i])
		{
			b_vec.push_back(i);
			num_entries_b++;
		}
	}

	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::host_memory>::invalid_index;

	fprintf(stderr, "num_entries: %d %d\n", num_entries_a, num_entries_b);
	int num_rows = a.size(), num_cols = b.size();
	mat.resize(num_rows, num_cols, num_entries_a*num_entries_b, num_entries_b);
	int num_cols_per_row = mat.column_indices.num_cols, pitch = mat.column_indices.pitch;
	for(int row=0; row < num_rows; ++row)
	{
		int offset = row;
		if(a[row])
		{
			for(int n=0; n < num_cols_per_row; ++n, offset+=pitch)
			{
				if(n < b_vec.size())
				{
					mat.column_indices.values[offset] = b_vec[n];
					mat.values.values[offset] = 1;
				}
				else
					mat.column_indices.values[offset] = invalid_index;
			}
		}
	}

	assert(mat.num_entries == num_entries_a*num_entries_b);
	//print_matrix_info(mat);
}

//*******************************************************************************//
//Host Forms

//F_call
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_call_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s, const int j)
{
	temp_vec.resize(Fun.num_rows);
	cusp::multiply(Fun, s, temp_vec);
	cusp::multiply(sigma, temp_vec, vf);

	temp_vec.resize(Arg[0].num_rows);
	for(int i=0; i<j; ++i)
	{
		cusp::multiply(Arg[i], s, temp_vec);
		cusp::multiply(sigma, temp_vec, v[i]);
		cusp::multiply(Var[i], vf, a[i]);
	}

	temp_Mat[0] = sigma;
	for(int i=0; i<j; ++i)
	{
		OuterProduct(v[i], a[i], temp_Mat[2]);
		if(i%2 == 0)
			cusp::add(temp_Mat[0], temp_Mat[2], temp_Mat[1]);
		else
			cusp::add(temp_Mat[1], temp_Mat[2], temp_Mat[0]);
	}
	sigma = (j%2 == 1) ? temp_Mat[1] : temp_Mat[0];

	//r_prime
	temp_vec.resize(Body.num_rows);
	cusp::multiply(Body, vf, temp_vec);
	AccumVec(r_prime, temp_vec);
}

//f_call
template <>
void CFA<int, int, cusp::host_memory>::f_call()
{
	std::vector<int> search_vec;
	//fprintf(stdout, "f_call\n");
	for(int j=1; j<=m_maxCall; ++j)
	{
		if(valid_Call[j])
		{
			AND_OP(r_prime, Call[j], search_vec);
			fprintf(stdout, "f_call_%d: %d\n", j, search_vec.size());

			for(int i=0; i<search_vec.size(); ++i)
			{
				thrust::fill(s.begin(), s.end(), 0);
				s[search_vec[i]] = 1;
				f_call_host(s, j);
			}
		}
	}
}

//f_list
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_list_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s, const int j)
{
	//vf = s[i]
	temp_vec.resize(Fun.num_rows);
	cusp::multiply(Fun, s, temp_vec);
	cusp::multiply(sigma, temp_vec, vf);
	cusp::multiply(Var[0], vf, a_var);

	temp_vec.resize(Arg[0].num_rows);
	thrust::fill(v_list.begin(), v_list.end(), 0);
	for(int i=0; i<j; ++i)
	{
		cusp::multiply(Arg[i], s, temp_vec);
		cusp::multiply(sigma, temp_vec, v[i]);
		AccumVec(v_list, v[i]);
	}
	AccumVec(v_list, LIST_vec);
	OuterProduct(v_list, a_var, temp_Mat[0]);
	cusp::add(temp_Mat[0], sigma, temp_Mat[1]);
	sigma = temp_Mat[1];

	//r_prime
	temp_vec.resize(Body.num_rows);
	cusp::multiply(Body, vf, temp_vec);
	AccumVec(r_prime, temp_vec);
}

//entry point
template <>
void CFA<int, int, cusp::host_memory>::f_list()
{
	std::vector<int> search_vec;
	//fprintf(stdout, "f_list\n");
	for(int j=0; j<=m_maxList; ++j)
	{
		if(valid_List[j])
		{
			AND_OP(r_prime, PrimList[j], search_vec);
			fprintf(stdout, "f_list_%d: %d\n", j, search_vec.size());

			for(int i=0; i<search_vec.size(); ++i)
			{
				thrust::fill(s.begin(), s.end(), 0);
				s[search_vec[i]] = 1;
				f_list_host(s, j);
			}
		}
	}
}


//F_set
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_set_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s)
{
	temp_vec.resize(Fun.num_rows);
	cusp::multiply(Fun, s, temp_vec);
	cusp::multiply(sigma, temp_vec, vf);
	cusp::multiply(Var[0], vf, a_var);
	cusp::multiply(Arg[0], s, a_set);
	temp_vec.resize(Arg[1].num_rows);
	cusp::multiply(Arg[1], s, temp_vec);
	cusp::multiply(sigma, temp_vec, v_set);

	//sigma + (a_var (X) void) + (a_set (X) v_set)
	OuterProduct(VOID_vec, a_var, temp_Mat[0]);
	OuterProduct(v_set, a_set, temp_Mat[1]);
	cusp::add(temp_Mat[0], temp_Mat[1], temp_Mat[2]);
	cusp::add(temp_Mat[2], sigma, temp_Mat[3]);
	sigma = temp_Mat[3];

	//r_prime
	temp_vec.resize(Body.num_rows);
	cusp::multiply(Body, vf, temp_vec);
	AccumVec(r_prime, temp_vec);
}

//entry point
template <>
void CFA<int, int, cusp::host_memory>::f_set()
{
	std::vector<int> search_vec;
	AND_OP(r_prime, SET, search_vec);
	fprintf(stdout, "f_set: %d\n", search_vec.size());
	for(int i=0; i<search_vec.size(); ++i)
	{
		thrust::fill(s.begin(), s.end(), 0);
		s[search_vec[i]] = 1;
		f_set_host(s);
	}
}

//f_if
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_if_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s)
{
	temp_vec.resize(Arg[0].num_rows);
	cusp::multiply(Arg[0], s, temp_vec);
	cusp::multiply(sigma, temp_vec, v_cond);

	int tb = thrust::inner_product(v_cond.begin(), v_cond.end(), NOT_FALSE_vec.begin(), 0);
	int fb = thrust::inner_product(v_cond.begin(), v_cond.end(), FALSE_vec.begin(), 0);

	temp_vec.resize(CondTrue.num_rows);
	if(tb && fb)
	{
		cusp::multiply(CondTrue, s, temp_vec);
		AccumVec(r_prime, temp_vec);
		cusp::multiply(CondFalse, s, temp_vec);
		AccumVec(r_prime, temp_vec);
	}
	else if(tb)
	{
		cusp::multiply(CondTrue, s, temp_vec);
		AccumVec(r_prime, temp_vec);
	}
	else if(fb)
	{
		cusp::multiply(CondFalse, s, temp_vec);
		AccumVec(r_prime, temp_vec);
	}
}

//entry point
template <>
void CFA<int, int, cusp::host_memory>::f_if()
{
	std::vector<int> search_vec;
	AND_OP(r_prime, IF, search_vec);
	fprintf(stdout, "f_if: %d\n", search_vec.size());
	for(int i=0; i<search_vec.size(); ++i)
	{
		thrust::fill(s.begin(), s.end(), 0);
		s[search_vec[i]] = 1;
		f_if_host(s);
	}
}

//f_primNum
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primNum_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s)
{
	temp_vec.resize(Fun.num_rows);
	cusp::multiply(Fun, s, temp_vec);
	cusp::multiply(sigma, temp_vec, vf);
	cusp::multiply(Var[0], vf, a_var);

	//sigma + (a_var (X) NUM)
	OuterProduct(NUM_vec, a_var, temp_Mat[0]);
	cusp::add(temp_Mat[0], sigma, temp_Mat[1]);
	sigma = temp_Mat[1];

	//r_prime
	temp_vec.resize(Body.num_rows);
	cusp::multiply(Body, vf, temp_vec);
	AccumVec(r_prime, temp_vec);
}

//entry point
template <>
void CFA<int, int, cusp::host_memory>::f_primNum()
{
	std::vector<int> search_vec;
	AND_OP(r_prime, PrimNum, search_vec);
	fprintf(stdout, "f_primNum: %d\n", search_vec.size());
	for(int i=0; i<search_vec.size(); ++i)
	{
		thrust::fill(s.begin(), s.end(), 0);
		s[search_vec[i]] = 1;
		f_primNum_host(s);
	}
}

//f_primBool
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primBool_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s)
{
	temp_vec.resize(Fun.num_rows);
	cusp::multiply(Fun, s, temp_vec);
	cusp::multiply(sigma, temp_vec, vf);
	cusp::multiply(Var[0], vf, a_var);

	//sigma + (a_var (X) #T#F)
	OuterProduct(BOOL_vec, a_var, temp_Mat[0]);
	cusp::add(temp_Mat[0], sigma, temp_Mat[1]);
	sigma = temp_Mat[1];

	//r_prime
	temp_vec.resize(Body.num_rows);
	cusp::multiply(Body, vf, temp_vec);
	AccumVec(r_prime, temp_vec);
}

//entry point
template <>
void CFA<int, int, cusp::host_memory>::f_primBool()
{
	std::vector<int> search_vec;
	AND_OP(r_prime, PrimBool, search_vec);
	fprintf(stdout, "f_primBool: %d\n", search_vec.size());
	for(int i=0; i<search_vec.size(); ++i)
	{
		thrust::fill(s.begin(), s.end(), 0);
		s[search_vec[i]] = 1;
		f_primBool_host(s);
	}
}

//f_primVoid
template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
void CFA<INDEX_TYPE, VALUE_TYPE, MEM_TYPE>::f_primVoid_host(const cusp::array1d<VALUE_TYPE, cusp::host_memory> &s)
{
	temp_vec.resize(Fun.num_rows);
	cusp::multiply(Fun, s, temp_vec);
	cusp::multiply(sigma, temp_vec, vf);
	cusp::multiply(Var[0], vf, a_var);

	//sigma + (a_var (X) VOID)
	OuterProduct(VOID_vec, a_var, temp_Mat[0]);
	cusp::add(temp_Mat[0], sigma, temp_Mat[1]);
	sigma = temp_Mat[1];

	//r_prime
	temp_vec.resize(Body.num_rows);
	cusp::multiply(Body, vf, temp_vec);
	AccumVec(r_prime, temp_vec);
}

//entry point
template <>
void CFA<int, int, cusp::host_memory>::f_primVoid()
{
	std::vector<int> search_vec;
	AND_OP(r_prime, PrimVoid, search_vec);
	fprintf(stdout, "f_PrimVoid: %d\n", search_vec.size());
	for(int i=0; i<search_vec.size(); ++i)
	{
		thrust::fill(s.begin(), s.end(), 0);
		s[search_vec[i]] = 1;
		f_primVoid_host(s);
	}
}

#endif