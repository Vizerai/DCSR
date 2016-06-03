#ifndef SPARSE_UPDATE_DEVICE
#define SPARSE_UPDATE_DEVICE

#define QUEUE_SIZE		512
#define WARP_SIZE 		32
#define LOG_WARP_SIZE	5

namespace device
{

__constant__ int chunk_sizes[8];

//*****************************************************************************//
//update matrices with arrays of row, column and value indices
//*****************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
UpdateMatrix_dcsr(	const INDEX_TYPE num_rows,
							const INDEX_TYPE pitch,
							const INDEX_TYPE alpha,
							const INDEX_TYPE *src_rows,
							const INDEX_TYPE *src_cols,
							const VALUE_TYPE *src_vals,
							const INDEX_TYPE *offsets,
							INDEX_TYPE *Aj,
							VALUE_TYPE *Ax,
							INDEX_TYPE *row_offsets,
							INDEX_TYPE *row_sizes)
{
	const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
	const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
	const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
	const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
	const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;             	// vector index within the block
	const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

	__shared__ volatile INDEX_TYPE new_segment[VECTORS_PER_BLOCK*2];

	for(INDEX_TYPE row=vector_id; row < num_rows; row+=num_vectors)
	{
		INDEX_TYPE Aoffset = 0;
		INDEX_TYPE rlA = row_sizes[row];
		INDEX_TYPE A_idx = 0;
		INDEX_TYPE A_row_start = row_offsets[row*2];
		INDEX_TYPE A_row_end = row_offsets[row*2 + 1];
		INDEX_TYPE free_mem = 0;

		const INDEX_TYPE B_row_start = offsets[row];
		const INDEX_TYPE B_row_end = offsets[row+1];
		const INDEX_TYPE rlB = B_row_end - B_row_start;

		if(rlB > 0)
		{
			//check to see if first bin has any memory allocated
			if(rlA == 0 && A_row_start == A_row_end)
			{
				//allocate new space for bin
				if(thread_lane == 0)
				{
					INDEX_TYPE new_size = rlB + alpha;
					INDEX_TYPE new_addr = atomicAdd(&row_sizes[num_rows], new_size);	//increase global memory pointer
					
					new_segment[vector_lane*2] = new_addr;
					new_segment[vector_lane*2 + 1] = new_addr + new_size;
					row_offsets[row*2] = new_addr;
					row_offsets[row*2 + 1] = new_addr + new_size;
				}

				A_idx = A_row_start = new_segment[vector_lane*2];
				A_row_end = new_segment[vector_lane*2 + 1];
			}
			else
			{
				if(rlA > 0)
				{
					while(A_idx < rlA)
					{
						A_idx += A_row_end - A_row_start;
						if(A_idx < rlA)
						{
							Aoffset++;
							A_row_start = row_offsets[Aoffset*pitch + row*2];
							A_row_end = row_offsets[Aoffset*pitch + row*2 + 1];
						}
					}

					A_idx = A_row_end + rlA - A_idx;
				}
				else
					A_idx = A_row_start;

				free_mem = A_row_end - A_idx;
				//allocate new space for bin
				if(thread_lane == 0 && free_mem < rlB)
				{
					//INDEX_TYPE new_size = max(rlB - free_mem, (A_row_end - A_row_start) * 2);
					INDEX_TYPE new_size = rlB - free_mem + alpha;
					INDEX_TYPE new_addr = atomicAdd(&row_sizes[num_rows], new_size);	//increase global memory pointer
					
					new_segment[vector_lane*2] = new_addr;
					new_segment[vector_lane*2 + 1] = new_addr + new_size;
					row_offsets[(Aoffset+1)*pitch + row*2] = new_addr;
					row_offsets[(Aoffset+1)*pitch + row*2 + 1] = new_addr + new_size;
				}
			}

			A_idx += thread_lane;
			for(INDEX_TYPE B_idx = B_row_start + thread_lane; B_idx < B_row_end; B_idx+=THREADS_PER_VECTOR, A_idx+=THREADS_PER_VECTOR)
			{
				if(A_idx >= A_row_end)
				{
					INDEX_TYPE pos = A_idx - A_row_end;
					Aoffset++;
					A_row_start = new_segment[vector_lane*2];
					A_row_end = new_segment[vector_lane*2 + 1];
					// A_row_start = row_offsets[Aoffset*pitch + row*2];
					// A_row_end = row_offsets[Aoffset*pitch + row*2 + 1];
					A_idx = A_row_start + pos;
				}

				Aj[A_idx] = src_cols[B_idx];
				Ax[A_idx] = src_vals[B_idx];
			}

			if(thread_lane == 0)
				row_sizes[row] += rlB;
		}
	}
}

//permuted variant
//takes in permuted row sizes and permuted row IDs as well
template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
UpdateMatrixP_dcsr(	const INDEX_TYPE num_rows,
					const INDEX_TYPE pitch,
					const INDEX_TYPE alpha,
					const INDEX_TYPE *src_rows,
					const INDEX_TYPE *src_cols,
					const VALUE_TYPE *src_vals,
					const INDEX_TYPE *offsets,
					INDEX_TYPE *Aj,
					VALUE_TYPE *Ax,
					INDEX_TYPE *row_offsets,
					INDEX_TYPE *row_sizes)
{
	// const INDEX_TYPE tID = blockDim.x*blockIdx.x + threadIdx.x;
	// const INDEX_TYPE grid_size = blockDim.x * gridDim.x;				//grid size

	const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
	const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
	const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
	const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
	const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;             	// vector index within the block
	const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

	__shared__ volatile INDEX_TYPE new_segment[VECTORS_PER_BLOCK];

	for(INDEX_TYPE row=vector_id; row < num_rows; row+=num_vectors)
	{
		INDEX_TYPE Aoffset = 0;
		INDEX_TYPE rlA = row_sizes[row];
		INDEX_TYPE A_idx = 0;
		INDEX_TYPE A_row_start = row_offsets[row*2];
		INDEX_TYPE A_row_end = row_offsets[row*2 + 1];
		INDEX_TYPE free_mem = 0;

		const INDEX_TYPE B_row_start = offsets[row];
		const INDEX_TYPE B_row_end = offsets[row+1];
		const INDEX_TYPE rlB = B_row_end - B_row_start;

		if(rlB > 0)
		{
			//check to see if first bin has any memory allocated
			if(rlA == 0 && A_row_start == A_row_end)
			{
				//allocate new space for bin
				if(thread_lane == 0)
				{
					INDEX_TYPE new_size = rlB + alpha;
					INDEX_TYPE new_addr = atomicAdd(&row_sizes[num_rows], new_size);	//increase global memory pointer
					
					new_segment[vector_lane] = new_addr;
					row_offsets[row*2] = new_addr;
					row_offsets[row*2 + 1] = new_addr + new_size;
				}

				A_idx = new_segment[vector_lane];
			}
			else
			{
				if(rlA > 0)
				{
					while(A_idx < rlA)
					{
						A_idx += A_row_end - A_row_start;
						if(A_idx < rlA)
						{
							Aoffset++;
							A_row_start = row_offsets[Aoffset*pitch + row*2];
							A_row_end = row_offsets[Aoffset*pitch + row*2 + 1];
						}
					}

					A_idx = A_row_end + rlA - A_idx;
				}
				else
					A_idx = A_row_start;

				free_mem = A_row_end - A_idx;
				//allocate new space for bin
				if(thread_lane == 0 && free_mem < rlB)
				{
					if(Aoffset + 1 >= BINS)
					{
						cuPrintf("segment overlow**  Aoffset: %d  row: %d  rlA: %d  rlB: %d\n", Aoffset, row, rlA, rlB);
						//break;
					}

					INDEX_TYPE new_size = rlB - free_mem + alpha;
					INDEX_TYPE new_addr = atomicAdd(&row_sizes[num_rows], new_size);	//increase global memory pointer

					row_offsets[(Aoffset+1)*pitch + row*2] = new_addr;
					row_offsets[(Aoffset+1)*pitch + row*2 + 1] = new_addr + new_size;
				}
			}

			for(INDEX_TYPE B_idx = B_row_start; B_idx < B_row_end; B_idx++, A_idx++)
			{
				if(A_idx >= A_row_end)
				{
					INDEX_TYPE pos = A_idx - A_row_end;
					Aoffset++;
					A_row_start = row_offsets[Aoffset*pitch + row*2];
					A_row_end = row_offsets[Aoffset*pitch + row*2 + 1];
					A_idx = A_row_start + pos;
				}

				Aj[A_idx] = src_cols[B_idx];
				Ax[A_idx] = src_vals[B_idx];
			}

			row_sizes[row] += rlB;
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
UpdateMatrix_hyb_B(	const INDEX_TYPE num_rows,
					const INDEX_TYPE num_cols,
					const INDEX_TYPE num_cols_per_row,
					const INDEX_TYPE pitch,
					const INDEX_TYPE *src_rows,
					const INDEX_TYPE *src_cols,
					const INDEX_TYPE N,
					INDEX_TYPE *rs,
					INDEX_TYPE *column_indices,
					INDEX_TYPE *overflow_rows,
					INDEX_TYPE *overflow_cols)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{
		for(INDEX_TYPE i=0; i < N; i++)
		{
			if(src_rows[i] == row)
			{
				INDEX_TYPE offset = row;
				INDEX_TYPE col = src_cols[i];
				INDEX_TYPE rl = rs[row];
				bool valid = true;

				// for(INDEX_TYPE j=0; j < rl && valid; j++)
				// {
				// 	if(column_indices[offset + j*pitch] == col)
				// 	{
				// 		valid = false;
				// 		break;
				// 	}
				// }

				if(rl < num_cols_per_row && valid)
				{
					column_indices[offset + rl*pitch] = col;
					rs[row] += 1;
					valid = false;
				}
				else if(valid) 	//overflow
				{
					bool ovf_valid = true;
					// for(INDEX_TYPE i=1; i <= rs[num_rows]; ++i)
					// {
					// 	if(overflow_cols[i] == col && overflow_rows[i] == row)
					// 	{
					// 		ovf_valid = false;
					// 		break;
					// 	}
					// }

					if(ovf_valid)
					{
						INDEX_TYPE index = atomicAdd(&rs[num_rows], 1);
						overflow_rows[index] = row;
						overflow_cols[index] = col;
					}
				}
			}
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
UpdateMatrix_hyb(	const INDEX_TYPE num_rows,
					const INDEX_TYPE num_cols,
					const INDEX_TYPE num_cols_per_row,
					const INDEX_TYPE pitch,
					const INDEX_TYPE *src_rows,
					const INDEX_TYPE *src_cols,
					const VALUE_TYPE *src_vals,
					const INDEX_TYPE *offsets,
					INDEX_TYPE *row_sizes,
					INDEX_TYPE *column_indices,
					VALUE_TYPE *vals,
					INDEX_TYPE *overflow_rows,
					INDEX_TYPE *overflow_cols,
					VALUE_TYPE *overflow_vals)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{
		INDEX_TYPE row_start = offsets[row];
		INDEX_TYPE row_end = offsets[row+1];

		for(INDEX_TYPE j = row_start; j < row_end; j++)
		{
			INDEX_TYPE offset = row;
			INDEX_TYPE col = src_cols[j];
			VALUE_TYPE val = src_vals[j];
			INDEX_TYPE rl = row_sizes[row];

			if(rl < num_cols_per_row)
			{
				column_indices[offset + rl*pitch] = col;
				vals[offset + rl*pitch] = val;
				row_sizes[row] += 1;
			}
			else
			{
				INDEX_TYPE index = atomicAdd(&row_sizes[num_rows], 1);
				row_sizes[row] += 1;
				overflow_rows[index] = row;
				overflow_cols[index] = col;
				overflow_vals[index] = val;
			}
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
UpdateMatrix_dell(	const INDEX_TYPE num_rows,
					const INDEX_TYPE pitch,
					const INDEX_TYPE num_cols_per_row,
					const INDEX_TYPE *src_rows,
					const INDEX_TYPE *src_cols,
					const VALUE_TYPE *src_vals,
					const INDEX_TYPE N,
					INDEX_TYPE *ell_column_indices,
					VALUE_TYPE *ell_values,
					INDEX_TYPE *Aj,
					VALUE_TYPE *Ax,
					INDEX_TYPE *row_offsets,
					INDEX_TYPE *row_sizes)
{
	const INDEX_TYPE tID = blockDim.x*blockIdx.x + threadIdx.x;
	const INDEX_TYPE grid_size = blockDim.x * gridDim.x;				//grid size

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{
		//loop of N inputs looking for elements from the respective row
		for(INDEX_TYPE i=0; i < N; i++)
		{
			if(src_rows[i] == row)
			{
				INDEX_TYPE col = src_cols[i];
				VALUE_TYPE val = src_vals[i];
				INDEX_TYPE rl = row_sizes[row];

				bool valid = true;
				INDEX_TYPE offset = 0;
				INDEX_TYPE r_idx = 0, idx, row_start, row_end;
				idx = row_start = row_offsets[row*2];
				row_end = row_offsets[row*2 + 1];

				for(idx = 0; idx < rl && idx < num_cols_per_row && valid; idx++)
				{
					if(ell_column_indices[row + idx*pitch] == col)
					{
						valid = false;
						break;
					}
				}
				if(rl < num_cols_per_row && valid)
				{
					ell_column_indices[row + idx*pitch] = col;
					ell_values[row + idx*pitch] = val;
					row_sizes[row] += 1;
					valid = false;
				}

				while(r_idx < rl && valid)
				{
					for(idx = row_start; idx < row_end && r_idx < rl; idx++, r_idx++)
					{
						if(Aj[idx] == col)
						{
							valid = false;
							break;
						}
					}

					if(idx >= row_end)
					{
						offset++;
						idx = row_start = row_offsets[offset*pitch + row*2];
						row_end = row_offsets[offset*pitch + row*2+1];
					}
				}

				if(valid)
				{
					//if non allocated memory section then allocate new section of memory for this row
					if(row_start == row_end)
					{
						//allocate new space for chunk
						INDEX_TYPE new_size = max((row_offsets[(offset-1)*pitch + row*2 + 1] - row_offsets[(offset-1)*pitch + row*2]) * (offset+1), 32);
						INDEX_TYPE new_add = atomicAdd(&row_sizes[num_rows], new_size);	//increase global memory pointer

						//allocate new row chunk...
						idx = row_offsets[offset*pitch + row*2] = new_add;
						row_offsets[offset*pitch + row*2+1] = new_add + new_size;
					}

					Aj[idx] = col;
					Ax[idx] = val;
					row_sizes[row] += 1;
				}
			}
		}
	}
}

//*******************************************************************************//
//Load matrix from coo matrix.  Assume no duplicate entries.
//*******************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
LoadMatrix_dcsr_B_coo(	const INDEX_TYPE num_rows,
						const INDEX_TYPE chunk_size,
						const INDEX_TYPE pitch,
						const float alpha,
						const INDEX_TYPE *src_rows,
						const INDEX_TYPE *src_cols,
						const INDEX_TYPE N,
						INDEX_TYPE *ci,
						INDEX_TYPE *cl,
						INDEX_TYPE *ca,
						INDEX_TYPE *rs,
						INDEX_TYPE *cols)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;		//thread ID
	//const int lID = threadIdx.x;								//lane ID
	const int grid_size = blockDim.x * gridDim.x;				//grid size

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{

	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
LoadMatrix_dcsr_coo(	const INDEX_TYPE num_rows,
						const INDEX_TYPE chunk_size,
						const INDEX_TYPE pitch,
						const float alpha,
						const INDEX_TYPE *src_rows,
						const INDEX_TYPE *src_cols,
						const VALUE_TYPE *src_vals,
						const INDEX_TYPE N,
						INDEX_TYPE *ci,
						INDEX_TYPE *cl,
						INDEX_TYPE *ca,
						INDEX_TYPE *rs,
						INDEX_TYPE *cols,
						VALUE_TYPE *vals)
{
	// const int tID = blockDim.x * blockIdx.x + threadIdx.x;		//thread ID
	// const int btID = threadIdx.x;								//block thread ID
	// const int grid_size = blockDim.x * gridDim.x;				//grid size

	// __shared__ volatile INDEX_TYPE rs_s[BLOCK_THREAD_SIZE];

	// for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	// {

	// }
}

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// __launch_bounds__(BLOCK_THREAD_SIZE,1)
// __global__ void 
// LoadMatrix_hyb_B_coo(	const INDEX_TYPE num_rows,
// 						const INDEX_TYPE num_cols,
// 						const INDEX_TYPE num_cols_per_row,
// 						const INDEX_TYPE pitch,
// 						const INDEX_TYPE *src_rows,
// 						const INDEX_TYPE *src_cols,
// 						const INDEX_TYPE N,
// 						INDEX_TYPE *rs,
// 						INDEX_TYPE *column_indices,
// 						VALUE_TYPE *vals,
// 						INDEX_TYPE *overflow_rows,
// 						INDEX_TYPE *overflow_cols)
// {
// 	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
// 	const int grid_size = blockDim.x * gridDim.x;

// 	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
// 	{
// 		for(INDEX_TYPE i=0; i < N; i++)
// 		{
// 			if(src_rows[i] == row)
// 			{
// 				INDEX_TYPE offset = row;
// 				INDEX_TYPE col = src_cols[i];
// 				INDEX_TYPE rl = rs[row];

// 				if(rl < num_cols_per_row)
// 				{
// 					column_indices[offset + rl*pitch] = col;
// 					rs[row] += 1;
// 				}
// 				else //overflow
// 				{
// 					INDEX_TYPE index = atomicAdd(&rs[num_rows], 1);
// 					rs[row] += 1;
// 					overflow_rows[index] = row;
// 					overflow_cols[index] = col;
// 				}
// 			}
// 		}
// 	}
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
LoadMatrix_hyb_coo(	const INDEX_TYPE num_rows,
					const INDEX_TYPE num_cols,
					const INDEX_TYPE num_cols_per_row,
					const INDEX_TYPE pitch,
					const INDEX_TYPE *rows,
					const INDEX_TYPE *cols,
					const VALUE_TYPE *vals,
					const INDEX_TYPE *offsets,
					INDEX_TYPE *rs,
					INDEX_TYPE *Aj,
					VALUE_TYPE *Ax,
					INDEX_TYPE *ovf_rows,
					INDEX_TYPE *ovf_cols,
					VALUE_TYPE *ovf_vals)
{
    const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    __shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];
    __shared__ volatile INDEX_TYPE ovf_idx[VECTORS_PER_BLOCK];

	for(INDEX_TYPE row=vector_id; row < num_rows; row+=num_vectors)
	{
		INDEX_TYPE row_start, row_end;
		if(thread_lane < 2)
			ptrs[vector_lane][thread_lane] = offsets[row + thread_lane];

		row_start = ptrs[vector_lane][0];
		row_end = ptrs[vector_lane][1];
		INDEX_TYPE size = row_end - row_start;

		for(INDEX_TYPE i=row_start+thread_lane; i < (row_start + num_cols_per_row) && i < row_end; i+=THREADS_PER_VECTOR)
		{
			INDEX_TYPE offset = i - row_start;
			Aj[row + offset*pitch] = cols[i];
			Ax[row + offset*pitch] = vals[i];
		}

		if(size > num_cols_per_row)
		{
			if(thread_lane == 0)
				ovf_idx[vector_lane] = atomicAdd(&rs[num_rows], size - num_cols_per_row);

			INDEX_TYPE offset = (size - num_cols_per_row);
			INDEX_TYPE start = ovf_idx[vector_lane], end = ovf_idx[vector_lane] + offset;
			for(INDEX_TYPE i=start + thread_lane; i < end; i+=THREADS_PER_VECTOR, offset+=THREADS_PER_VECTOR)
			{
				ovf_rows[i] = row;
				ovf_cols[i] = cols[offset];
				ovf_vals[i] = vals[offset];
			}
		}

		if(thread_lane == 0)
			rs[row] += size;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
LoadMatrix_hyb_csr(	const INDEX_TYPE num_rows,
					const INDEX_TYPE num_cols,
					const INDEX_TYPE num_cols_per_row,
					const INDEX_TYPE pitch,
					const INDEX_TYPE *cols,
					const VALUE_TYPE *vals,
					const INDEX_TYPE *offsets,
					INDEX_TYPE *rs,
					INDEX_TYPE *Aj,
					VALUE_TYPE *Ax,
					INDEX_TYPE *ovf_rows,
					INDEX_TYPE *ovf_cols,
					VALUE_TYPE *ovf_vals)
{
    const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    __shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];
    __shared__ volatile INDEX_TYPE ovf_idx[VECTORS_PER_BLOCK];

	for(INDEX_TYPE row=vector_id; row < num_rows; row+=num_vectors)
	{
		INDEX_TYPE row_start, row_end;
		if(thread_lane < 2)
			ptrs[vector_lane][thread_lane] = offsets[row + thread_lane];

		row_start = ptrs[vector_lane][0];
		row_end = ptrs[vector_lane][1];
		INDEX_TYPE size = row_end - row_start;

		for(INDEX_TYPE i=row_start+thread_lane; i < (row_start + num_cols_per_row) && i < row_end; i+=THREADS_PER_VECTOR)
		{
			INDEX_TYPE offset = i - row_start;
			Aj[row + offset*pitch] = cols[i];
			Ax[row + offset*pitch] = vals[i];
		}

		if(size > num_cols_per_row)
		{
			if(thread_lane == 0)
				ovf_idx[vector_lane] = atomicAdd(&rs[num_rows], size - num_cols_per_row);

			INDEX_TYPE offset = (size - num_cols_per_row);
			INDEX_TYPE start = ovf_idx[vector_lane], end = ovf_idx[vector_lane] + offset;
			for(INDEX_TYPE i=start + thread_lane; i < end; i+=THREADS_PER_VECTOR, offset+=THREADS_PER_VECTOR)
			{
				ovf_rows[i] = row;
				ovf_cols[i] = cols[offset];
				ovf_vals[i] = vals[offset];
			}
		}

		if(thread_lane == 0)
			rs[row] += size;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
LoadMatrix_csr_coo(	const INDEX_TYPE num_rows,
					const INDEX_TYPE *src_rows,
					const INDEX_TYPE *src_cols,
					const VALUE_TYPE *src_vals,
					const INDEX_TYPE N,
					INDEX_TYPE *T_i,
					INDEX_TYPE *A_j,
					VALUE_TYPE *A_x)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE i=tID; i < N; i+=grid_size)
	{
		T_i[i] = src_rows[i];
		A_j[i] = src_cols[i];
		A_x[i] = src_vals[i];
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
LoadMatrix_csr_count_rows(	const INDEX_TYPE num_rows,
							const INDEX_TYPE *src_rows,
							const INDEX_TYPE N,
							INDEX_TYPE *row_offsets)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row=tID; row < num_rows; row+=grid_size)
	{
		for(INDEX_TYPE i=0; i < N; i++)
		{
			if(src_rows[i] == row)
			{
				row_offsets[row] += 1;
			}
		}
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
ConvertMatrix_DELL_CSR(	const INDEX_TYPE num_rows,
						const INDEX_TYPE chunk_size,
						const INDEX_TYPE pitch,
						const INDEX_TYPE *ci,
						const INDEX_TYPE *cl,
						const INDEX_TYPE *ca,
						const INDEX_TYPE *rs,
						const INDEX_TYPE *src_column_indices,
						const VALUE_TYPE *src_values,
						INDEX_TYPE *row_offsets,
						INDEX_TYPE *dst_column_indices,
						VALUE_TYPE *dst_values)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
        INDEX_TYPE rl = rs[row];
        INDEX_TYPE r_idx = 0;
        INDEX_TYPE cID = row / chunk_size;
        bool next_chunk = false;

        do
        {
            INDEX_TYPE next_cID = ci[cID];
            //INDEX_TYPE offset = A_ca[cID] + (row % chunk_size);
            INDEX_TYPE offset = ca[cID] + (row & (chunk_size-1))*pitch;
            INDEX_TYPE clength = cl[cID];

            INDEX_TYPE csr_row_start = row_offsets[row];
            //csr_row_end = row_offsets[row+1];

            for(INDEX_TYPE c_idx = 0; c_idx < clength && r_idx < rl; ++c_idx, ++r_idx)
            {
            	dst_column_indices[csr_row_start + r_idx] = src_column_indices[offset + c_idx];
                dst_values[csr_row_start + r_idx] = src_values[offset + c_idx];
            }

            if(next_cID > 0 && r_idx < rl)
            {
                next_chunk = true;
                cID = next_cID;
            }
            else
                next_chunk = false;

        } while(next_chunk);
    }
}

//*******************************************************************************//
//Sort DELL matrix rows
//*******************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
SetRowIndices(	const INDEX_TYPE num_rows,
				const INDEX_TYPE pitch,
				INDEX_TYPE *Ai,
				const INDEX_TYPE *A_row_offsets,
				const INDEX_TYPE *A_rs)
{
    __shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];

    const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
    {
    	INDEX_TYPE r_idx = 0;
        const INDEX_TYPE rl = A_rs[row];

    	#pragma unroll
        for(INDEX_TYPE offset = 0; offset < BINS && r_idx < rl; offset++)
        {
            // use two threads to fetch A_row_offsets[row] and A_row_offsets[row+1]
            // this is considerably faster than the straightforward version
            #if(THREADS_PER_VECTOR >= 2)
            if(thread_lane < 2)
            {
                ptrs[vector_lane][thread_lane] = A_row_offsets[offset*pitch + row*2 + thread_lane];
                //ptrs[vector_lane][1] = A_row_offsets[offset*pitch + row*2 + 1];
            }
            #else
            {
                ptrs[vector_lane][0] = A_row_offsets[offset*pitch + row*2];
                ptrs[vector_lane][1] = A_row_offsets[offset*pitch + row*2 + 1];
            }
            #endif

            const INDEX_TYPE row_start = ptrs[vector_lane][0];          					//same as: row_start = A_row_offsets[row];
            const INDEX_TYPE row_end = min(ptrs[vector_lane][1], row_start + rl - r_idx);	//same as: row_end   = A_row_offsets[row+1];

            for(INDEX_TYPE jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
            	Ai[jj] = row;

        	r_idx += (row_end - row_start);
        }
    }
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
// __launch_bounds__(BLOCK_THREAD_SIZE,1)
// __global__ void
// CompactIndices(	const INDEX_TYPE num_rows,
// 				const INDEX_TYPE pitch,
// 				INDEX_TYPE *T_cols,
// 				VALUE_TYPE *T_vals,
// 				const INDEX_TYPE *temp_offsets,
// 				const INDEX_TYPE *Ai,
// 				const VALUE_TYPE *Ax,
// 				const INDEX_TYPE *A_row_sizes,
// 				const INDEX_TYPE *A_row_offsets)
// {
// 	__shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];

//     const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
//     const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
//     const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
//     const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
//     const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
//     const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

//     for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
//     {
// 		INDEX_TYPE r_idx = 0, T_idx;
// 		const INDEX_TYPE rl = A_row_sizes[row];
// 		const INDEX_TYPE Tstart = temp_offsets[row];
// 		//const INDEX_TYPE Tend = temp_offsets[row + 1];
// 		T_idx = Tstart + thread_lane;

//     	#pragma unroll
//         for(INDEX_TYPE offset = 0; offset < BINS && r_idx < rl; offset++)
//         {
//             // use two threads to fetch A_row_offsets[row] and A_row_offsets[row+1]
//             // this is considerably faster than the straightforward version
//             if(THREADS_PER_VECTOR >= 2)
//             {
// 				if(thread_lane < 2)
// 					ptrs[vector_lane][thread_lane] = A_row_offsets[offset*pitch + row*2 + thread_lane];
// 	        }
// 			else
// 			{
// 				ptrs[vector_lane][0] = A_row_offsets[offset*pitch + row*2];
// 				ptrs[vector_lane][1] = A_row_offsets[offset*pitch + row*2 + 1];
// 			}

//             const INDEX_TYPE row_start = ptrs[vector_lane][0];          					//same as: row_start = A_row_offsets[row];
//             const INDEX_TYPE row_end = min(ptrs[vector_lane][1], row_start + rl - r_idx);	//same as: row_end   = A_row_offsets[row+1];

//             for(INDEX_TYPE jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR, T_idx += THREADS_PER_VECTOR)
//             {
//             	T_cols[T_idx] = Ai[jj];
//             	T_vals[T_idx] = Ax[jj];
//             }

//         	r_idx += (row_end - row_start);
//         }
//     }
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
CompactIndices(	const INDEX_TYPE num_rows,
				const INDEX_TYPE pitch,
				INDEX_TYPE *T_cols,
				VALUE_TYPE *T_vals,
				const INDEX_TYPE *temp_offsets,
				const INDEX_TYPE *Aj,
				const VALUE_TYPE *Ax,
				const INDEX_TYPE *A_Prow_sizes,
				const INDEX_TYPE *A_row_ids,
				const INDEX_TYPE *A_row_offsets,
				const INDEX_TYPE *A_row_sizes)
{
	__shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];

    const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    INDEX_TYPE start_range, end_range;
    switch(THREADS_PER_VECTOR)
    {
    	case 1:
            start_range = c_bin_offsets[0];
            end_range = c_bin_offsets[1];
            break;
        case 2:
            start_range = c_bin_offsets[1];
            end_range = c_bin_offsets[2];
            break;
        case 4:
            start_range = c_bin_offsets[2];
            end_range = c_bin_offsets[3];
            break;
        case 8:
            start_range = c_bin_offsets[3];
            end_range = c_bin_offsets[4];
            break;
        case 16:
            start_range = c_bin_offsets[4];
            end_range = c_bin_offsets[5];
            break;
        case 32:
            start_range = c_bin_offsets[5];
            end_range = c_bin_offsets[7];			//read offsets for 2 bins (32+ and 256+ bins)
            break;
    }

    for(INDEX_TYPE iter = start_range + vector_id; iter < end_range; iter += num_vectors)
    {
		INDEX_TYPE r_idx = 0, T_idx;
		const INDEX_TYPE row = A_row_ids[iter];
		const INDEX_TYPE rl = A_Prow_sizes[iter];
		const INDEX_TYPE Tstart = temp_offsets[iter];		//permuted version
		//const INDEX_TYPE Tstart = temp_offsets[row];
		//const INDEX_TYPE Tend = temp_offsets[row + 1];

    	#pragma unroll
        for(INDEX_TYPE offset = 0; offset < BINS && r_idx < rl; offset++)
        {
            // use two threads to fetch A_row_offsets[row] and A_row_offsets[row+1]
            // this is considerably faster than the straightforward version
            if(THREADS_PER_VECTOR >= 2)
            {
				if(thread_lane < 2)
					ptrs[vector_lane][thread_lane] = A_row_offsets[offset*pitch + row*2 + thread_lane];
	        }
			else
			{
				ptrs[vector_lane][0] = A_row_offsets[offset*pitch + row*2];
				ptrs[vector_lane][1] = A_row_offsets[offset*pitch + row*2 + 1];
			}

            const INDEX_TYPE row_start = ptrs[vector_lane][0];          					//same as: row_start = A_row_offsets[row];
            const INDEX_TYPE row_end = min(ptrs[vector_lane][1], row_start + rl - r_idx);	//same as: row_end   = A_row_offsets[row+1];

			T_idx = Tstart + r_idx + thread_lane;
            for(INDEX_TYPE jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR, T_idx += THREADS_PER_VECTOR)
            {
				T_cols[T_idx] = Aj[jj];
				T_vals[T_idx] = Ax[jj];
            }

        	r_idx += (row_end - row_start);
        }
    }
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int BINS>
// __launch_bounds__(BLOCK_THREAD_SIZE,1)
// __global__ void
// SetOffsets(	const INDEX_TYPE num_rows,
// 			const INDEX_TYPE pitch,
// 			INDEX_TYPE *A_o,
// 			INDEX_TYPE *A_rs,
// 			const INDEX_TYPE *temp_offsets)
// {
// 	const int tID = blockDim.x * blockIdx.x + threadIdx.x;			// global thread index
// 	const int grid_size = blockDim.x * gridDim.x;
    
//     for(INDEX_TYPE row = tID; row < num_rows; row += grid_size)
//     {
//         A_o[row*2] = temp_offsets[row];
//         A_o[row*2 + 1] = temp_offsets[row+1];

//         //reset other indices
//         #pragma unroll
//         for(int offset=1; offset<BINS; offset++)
//         {
//         	A_o[offset*pitch + row*2] = -1;
//         	A_o[offset*pitch + row*2 + 1] = -1;
//     	}
//     }

//     if(tID == 0)
//     {
//     	A_rs[num_rows] = temp_offsets[num_rows];		//memory allocation pointer
//     }
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
SetOffsets(	const INDEX_TYPE num_rows,
			const INDEX_TYPE pitch,
			INDEX_TYPE *A_o,
			INDEX_TYPE *A_rs,
			const INDEX_TYPE *A_rids,
			const INDEX_TYPE *temp_offsets)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;			// global thread index
	const int grid_size = blockDim.x * gridDim.x;
    
    for(INDEX_TYPE iter = tID; iter < num_rows; iter += grid_size)
    {
    	const INDEX_TYPE row = A_rids[iter];
        A_o[row*2] = temp_offsets[iter];
        A_o[row*2 + 1] = temp_offsets[iter+1];

        //reset other indices
        #pragma unroll
        for(int offset=1; offset<BINS; offset++)
        {
        	A_o[offset*pitch + row*2] = -1;
        	A_o[offset*pitch + row*2 + 1] = -1;
    	}
    }

    if(tID == 0)
    {
    	A_rs[num_rows] = temp_offsets[num_rows];		//memory allocation pointer
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
SetRowData(		const INDEX_TYPE num_rows,
				const INDEX_TYPE pitch,
				INDEX_TYPE *A_ro,
				INDEX_TYPE *A_rs,
				const INDEX_TYPE *temp_offsets)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;			// global thread index
	const int grid_size = blockDim.x * gridDim.x;
    
    for(INDEX_TYPE row = tID; row < num_rows; row += grid_size)
    {   
        A_ro[row*2] = temp_offsets[row];
        A_ro[row*2 + 1] = temp_offsets[row+1];
        A_rs[row] = temp_offsets[row+1] - temp_offsets[row];
    }

    if(tID == 0)
		A_rs[num_rows] = temp_offsets[num_rows];
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
DCSRtoCSROffsets(	const INDEX_TYPE num_rows,
					const INDEX_TYPE pitch,
					const INDEX_TYPE *A_ro,
					INDEX_TYPE *CSR_offsets)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;			// global thread index
	const int grid_size = blockDim.x * gridDim.x;
    
    for(INDEX_TYPE row = tID; row < num_rows; row += grid_size)
    {
		CSR_offsets[row] = A_ro[row*2];
    }

    if(tID == 0)
    	CSR_offsets[num_rows] = A_ro[(num_rows-1)*2 + 1];			//ending offset
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
SetBins(	const INDEX_TYPE num_rows,
			INDEX_TYPE *bins,
			INDEX_TYPE *row_ids,
			INDEX_TYPE *row_sizes)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;			// global thread index
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row = tID; row < num_rows; row += grid_size)
	{
		INDEX_TYPE rl = row_sizes[row], bin;
		if(rl < 2)
			bin = 0;
		else if(rl >= 2 && rl < 4)
			bin = 1;
		else if(rl >= 4 && rl < 8)
			bin = 2;
		else if(rl >= 8 && rl < 16)
			bin = 3;
		else if(rl >= 16 && rl < 32)
			bin = 4;
		else if(rl >= 32 && rl < 512)
			bin = 5;
		else
			bin = 6;

		bins[row] = bin;
		row_ids[row] = row;
	}
}

//*******************************************************************************//
//Initialize dcsr or dcsr_B matrix
//*******************************************************************************//
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
InitializeMatrix_dcsr(	const INDEX_TYPE num_rows,
						const INDEX_TYPE chunk_length,
						INDEX_TYPE *row_sizes,
						INDEX_TYPE *row_offsets)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	row_offsets[row*2] = row * chunk_length;
    	row_offsets[row*2+1] = row * chunk_length + chunk_length;
    }

	if(tID == 0)
	{
		row_sizes[num_rows] = num_rows*chunk_length;
	}
}

}	//namespace device

#endif