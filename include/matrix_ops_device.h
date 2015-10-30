#ifndef MATRIX_OPS_DEVICE
#define MATRIX_OPS_DEVICE

#define QUEUE_SIZE		512

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
AccumMat(	const 	INDEX_TYPE num_rows,
			const 	INDEX_TYPE num_cols,
			const 	INDEX_TYPE num_cols_per_row,
			const 	INDEX_TYPE pitch,
			const 	INDEX_TYPE *column_indices,
					VALUE_TYPE *a)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE offset = row;
		for(int n=0; n < num_cols_per_row; ++n)
		{
			int col = column_indices[offset];
			if(col != invalid_index)
			{
				a[row] = 1;
			}
			offset += pitch;
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
column_select(	const INDEX_TYPE num_rows,
				const INDEX_TYPE *A_row_offsets,
				const INDEX_TYPE *A_column_indices,
				const VALUE_TYPE *s,
				VALUE_TYPE *y)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	const INDEX_TYPE index = s[0];

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE offset = A_row_offsets[row];
		INDEX_TYPE num_cols_per_row = A_row_offsets[row + 1] - offset;

		VALUE_TYPE val = 0;
		for(int n=0; n < num_cols_per_row; ++n, ++offset)
		{
			int col = A_column_indices[offset];
			if(col == index)
			{
				val = 1;
			}
		}

		y[row] = val;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
column_select_if(	const INDEX_TYPE num_rows,
					const INDEX_TYPE *A_row_offsets,
					const INDEX_TYPE *A_column_indices,
					const VALUE_TYPE *s,
					const VALUE_TYPE *cond,
					VALUE_TYPE *y)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	//check conditional value
	if(cond[0] == 0)
		return;

	const int index = s[0];

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE offset = A_row_offsets[row];
		INDEX_TYPE num_cols_per_row = A_row_offsets[row + 1] - offset;

		VALUE_TYPE val = 0;
		for(int n=0; n < num_cols_per_row; ++n, ++offset)
		{
			int col = A_column_indices[offset];
			if(col == index)
			{
				val = 1;
			}
		}

		y[row] = val;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
ell_add(	const INDEX_TYPE num_rows,
			const INDEX_TYPE num_cols,
			const INDEX_TYPE A_num_cols_per_row,
			const INDEX_TYPE B_num_cols_per_row,
			const INDEX_TYPE C_num_cols_per_row,
			const INDEX_TYPE A_pitch,
			const INDEX_TYPE B_pitch,
			const INDEX_TYPE C_pitch,
			const INDEX_TYPE *A_column_indices,
			const INDEX_TYPE *B_column_indices,
			INDEX_TYPE *C_column_indices,
			VALUE_TYPE *C_values)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	//__shared__ int entry_count[BLOCK_THREAD_SIZE];
	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory>::invalid_index;

	//entry_count[tID] = 0;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		int a_index = row;
		int b_index = row;
		int c_index = row;
		int A_col = A_column_indices[a_index];
		int B_col = B_column_indices[b_index];
		for(int n=0; n < C_num_cols_per_row; ++n, c_index+=C_pitch)
		{
			if(A_col == invalid_index && B_col == invalid_index)
			{
				C_column_indices[c_index] = invalid_index;
			}
			else if(A_col != invalid_index && B_col == invalid_index)
			{
				C_column_indices[c_index] = A_col;
				C_values[c_index] = 1;
				
				a_index += A_pitch;

				if(a_index < A_num_cols_per_row * A_pitch)
					A_col = A_column_indices[a_index];
				else
					A_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(A_col == invalid_index && B_col != invalid_index)
			{
				C_column_indices[c_index] = B_col;
				C_values[c_index] = 1;
				
				b_index += B_pitch;

				if(b_index < B_num_cols_per_row * B_pitch)
					B_col = B_column_indices[b_index];
				else
					B_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(A_column_indices[a_index] < B_column_indices[b_index])
			{
				C_column_indices[c_index] = A_col;
				C_values[c_index] = 1;
				
				a_index += A_pitch;

				if(a_index < A_num_cols_per_row * A_pitch)
					A_col = A_column_indices[a_index];
				else
					A_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(B_column_indices[b_index] < A_column_indices[a_index])
			{
				C_column_indices[c_index] = B_col;
				C_values[c_index] = 1;
				
				b_index += B_pitch;

				if(b_index < B_num_cols_per_row * B_pitch)
					B_col = B_column_indices[b_index];
				else
					B_col = invalid_index;
				//entry_count[tID]++;
			}
			else if(A_column_indices[a_index] == B_column_indices[b_index])
			{
				C_column_indices[c_index] = A_col;
				C_values[c_index] = 1;
				
				a_index += A_pitch;
				b_index += B_pitch;

				if(a_index < A_num_cols_per_row * A_pitch)
					A_col = A_column_indices[a_index];
				else
					A_col = invalid_index;
				if(b_index < B_num_cols_per_row * B_pitch)
					B_col = B_column_indices[b_index];
				else
					B_col = invalid_index;
				//entry_count[tID]++;
			}
		}
	}
	//__syncthreads();

	// reduce_sum<INDEX_TYPE>(entry_count, BLOCK_THREAD_SIZE, tID);
	// if(tID == 0)
	// 	*C_num_entries = entry_count[0];
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
LoadEllMatrix(	const INDEX_TYPE num_rows,
				const INDEX_TYPE num_entries,
				const INDEX_TYPE num_cols_per_row,
				const INDEX_TYPE pitch,
				const INDEX_TYPE *src_row_offsets,
				const INDEX_TYPE *src_column_indices,
				const VALUE_TYPE *src_values,
				INDEX_TYPE *dst_column_indices,
				VALUE_TYPE *dst_values)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	const INDEX_TYPE invalid_index = cusp::ell_matrix<INDEX_TYPE, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE row_start = src_row_offsets[row];
		INDEX_TYPE row_end = src_row_offsets[row + 1];
		INDEX_TYPE offset = row;

		dst_column_indices[row] = (row_end - row_start);
		for(int j=row_start; j < row_end; ++j, offset+=pitch)
		{
			dst_column_indices[offset] = src_column_indices[j];
			dst_values[offset] = src_values[j];
		}

		while(offset < num_cols_per_row * pitch)
		{
			dst_column_indices[offset] = invalid_index;
			offset += pitch;
		}
	}
}

}	//namespace device

#endif
