#ifndef PRIMITIVES_DEVICE_H
#define PRIMITIVES_DEVICE_H

#define WARP_SIZE	32

//shared memory size = BLOCK_SIZE
//shared memory reduce
template <typename VALUE_TYPE, int BLOCK_SIZE>
inline __device__ void reduce_sum(VALUE_TYPE *a, const int tID)
{
	int n = BLOCK_SIZE;
	while(n > 1)
	{
		int half = n / 2;
		if(tID < half)
			a[tID] += a[n - tID - 1];
		n = n - half;
		__syncthreads();
	}
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
FILL(	VALUE_TYPE *a,
		const VALUE_TYPE value,
		const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(int i=tID; i<size; i+=grid_size)
		a[i] = value;
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
AND_OP(	const VALUE_TYPE *A,
		const VALUE_TYPE *B,
		VALUE_TYPE *C,
		const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(int i=tID; i<size; i+=grid_size)
		C[i] = A[i] & B[i];
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
get_indices(	const VALUE_TYPE *a,
				VALUE_TYPE *b,
				const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	__shared__ int index;
	index = 0;
	__syncthreads();

	for(int i=tID; i<size; i+=grid_size)
	{
		if(a[i] != 0)
		{
			int old_index = atomicAdd(&index, 1);
			b[old_index] = i;
		}
	}
}

//b size = a size + 1
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
gather_reduce(	const VALUE_TYPE *a,
				VALUE_TYPE *b,
				INDEX_TYPE *index_count,
				const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	__shared__ int count[BLOCK_THREAD_SIZE];
	__shared__ int index;
	count[tID] = 0;
	index = 0;
	__syncthreads();

	for(int i=tID; i<size; i+=grid_size)
	{
		if(a[i] != 0)
		{
			int old_index = atomicAdd(&index, 1);
			b[old_index] = i;
			count[tID]++;
		}
	}
	__syncthreads();

	reduce_sum<VALUE_TYPE, BLOCK_THREAD_SIZE>(count, tID);

	if(tID == 0)
		index_count[0] = count[0];
}

template <typename INDEX_TYPE, typename VALUE_TYPE, int BLOCK_SIZE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
count(	const VALUE_TYPE *a,
		const VALUE_TYPE val,
		INDEX_TYPE *res,
		const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	__shared__ INDEX_TYPE count[BLOCK_SIZE];
	count[tID] = 0;
	__syncthreads();

	for(int i=tID; i < size; i+=grid_size)
	{
		if(a[i] == val)
			count[tID]++;
	}
	__syncthreads();

	reduce_sum<VALUE_TYPE, BLOCK_SIZE>(count, tID);

	if(tID == 0)
		res[0] = count[0];
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
AccumVec(	VALUE_TYPE *a,
			const VALUE_TYPE *b,
			const int size)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	//a += b
	for(int i=tID; i<size; i+=grid_size)
	{
		if(b[i])
			a[i] = 1;
		//a[i] += b[i];
	}
}

template <typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void 
InnerProductStore(	const VALUE_TYPE *a,
					const VALUE_TYPE *b,
					const int size,
					VALUE_TYPE *c)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;

	for(int index=tID; index < size; index+=grid_size)
	{
		if(a[index] != 0 && b[index] != 0)
		{
			c[0] = 1;
		}
	}
}

//a (.) Mat -> b
template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void InnerProduct(	const VALUE_TYPE *a,
								VALUE_TYPE *b,
								const INDEX_TYPE size_a,
								const INDEX_TYPE size_b,
								const INDEX_TYPE num_rows,
								const INDEX_TYPE num_cols,
								const INDEX_TYPE num_cols_per_row,
								const INDEX_TYPE pitch,
								const INDEX_TYPE *column_indices)
{
	const int tID = blockDim.x * blockIdx.x + threadIdx.x;
	const int grid_size = blockDim.x * gridDim.x;
	
	const INDEX_TYPE invalid_index = cusp::ell_matrix<int, VALUE_TYPE, cusp::device_memory>::invalid_index;

	for(int row=tID; row<size_a; row+=grid_size)
	{
		int offset = row;
		for(int n=0; n < num_cols_per_row; ++n, offset+=pitch)
		{
			int col = column_indices[offset];
			if(col != invalid_index)
			{
				if(a[row] == 1)
					b[col] = 1;
			}
		}
	}
}


///////////////////////////////////////////////////////////////////////////////////////////
//Count indices
///////////////////////////////////////////////////////////////////////////////////////////

// segmented reduction in shared memory
template <typename IndexType, typename ValueType>
__device__ ValueType segreduce_warp(const IndexType thread_lane, IndexType index, ValueType val, IndexType * indices, ValueType * vals)
{
    indices[threadIdx.x] = index;
    vals[threadIdx.x] = val;

    if( thread_lane >=  1 && index == indices[threadIdx.x -  1] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  1]; } 
    if( thread_lane >=  2 && index == indices[threadIdx.x -  2] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  2]; }
    if( thread_lane >=  4 && index == indices[threadIdx.x -  4] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  4]; }
    if( thread_lane >=  8 && index == indices[threadIdx.x -  8] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  8]; }
    if( thread_lane >= 16 && index == indices[threadIdx.x - 16] ) { vals[threadIdx.x] = val = val + vals[threadIdx.x - 16]; }

    return val;
}

template <typename IndexType, typename ValueType>
__device__ void segreduce_block(const IndexType * idx, ValueType * val)
{
    ValueType left = 0;
    if( threadIdx.x >=   1 && idx[threadIdx.x] == idx[threadIdx.x -   1] ) { left = val[threadIdx.x -   1]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();  
    if( threadIdx.x >=   2 && idx[threadIdx.x] == idx[threadIdx.x -   2] ) { left = val[threadIdx.x -   2]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   4 && idx[threadIdx.x] == idx[threadIdx.x -   4] ) { left = val[threadIdx.x -   4]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=   8 && idx[threadIdx.x] == idx[threadIdx.x -   8] ) { left = val[threadIdx.x -   8]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  16 && idx[threadIdx.x] == idx[threadIdx.x -  16] ) { left = val[threadIdx.x -  16]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >=  32 && idx[threadIdx.x] == idx[threadIdx.x -  32] ) { left = val[threadIdx.x -  32]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();  
    if( threadIdx.x >=  64 && idx[threadIdx.x] == idx[threadIdx.x -  64] ) { left = val[threadIdx.x -  64]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 128 && idx[threadIdx.x] == idx[threadIdx.x - 128] ) { left = val[threadIdx.x - 128]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
    if( threadIdx.x >= 256 && idx[threadIdx.x] == idx[threadIdx.x - 256] ) { left = val[threadIdx.x - 256]; } __syncthreads(); val[threadIdx.x] += left; left = 0; __syncthreads();
}

template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
count_sorted_indices_kernel(	const IndexType total_size,
		                     	const IndexType interval_size,
		                     	const IndexType * I, 
		                        IndexType * X,
		                        IndexType * temp_indices,
		                    	IndexType * temp_vals)
{
    __shared__ volatile IndexType indices[48 *(BLOCK_SIZE/32)];
    __shared__ volatile ValueType vals[BLOCK_SIZE];

    const IndexType thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;                        // global thread index
    const IndexType thread_lane = threadIdx.x & (WARP_SIZE-1);                                  // thread index within the warp
    const IndexType warp_id     = thread_id   / WARP_SIZE;                                      // global warp index

    const IndexType interval_begin = warp_id * interval_size;                                   // warp's offset into I,J,V
    const IndexType interval_end   = thrust::min(interval_begin + interval_size, total_size);	// end of warps's work

    const IndexType idx = 16 * (threadIdx.x/32 + 1) + threadIdx.x;                              // thread's index into padded rows array

    indices[idx - 16] = -1;                                                                     // fill padding with invalid row index

    if(interval_begin >= interval_end)                                                          // warp has no work to do 
        return;

    if (thread_lane == 31)
    {
        // initialize the carry in values
        indices[idx] = I[interval_begin]; 
		vals[threadIdx.x] = ValueType(0);
    }
  
    for(IndexType n = interval_begin + thread_lane; n < interval_end; n += WARP_SIZE)
    {
        IndexType index = I[n];                                       // index (i)
        ValueType val = 1;
        
        if (thread_lane == 0)
        {
            if(index == indices[idx + 31])
                val += vals[threadIdx.x + 31];                      	// row continues
            else
                X[indices[idx + 31]] += vals[threadIdx.x + 31];  		// row terminated
        }
        
        indices[idx] = index;
        vals[threadIdx.x] = val;

        if(index == indices[idx -  1]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  1]; } 
        if(index == indices[idx -  2]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  2]; }
        if(index == indices[idx -  4]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  4]; }
        if(index == indices[idx -  8]) { vals[threadIdx.x] = val = val + vals[threadIdx.x -  8]; }
        if(index == indices[idx - 16]) { vals[threadIdx.x] = val = val + vals[threadIdx.x - 16]; }

        if(thread_lane < 31 && index != indices[idx + 1])
        	X[index] += vals[threadIdx.x];                                            // row terminated
    }

    if(thread_lane == 31)
    {
        // write the carry out values
        temp_indices[warp_id] = indices[idx];
        temp_vals[warp_id] = vals[threadIdx.x];
    }
}


// The second level of the segmented reduction operation
template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
count_sorted_indices_update_kernel(	const IndexType num_warps,
                              		const IndexType * temp_indices,
                              		const ValueType * temp_vals,
                                	ValueType * X)
{
    __shared__ IndexType indices[BLOCK_SIZE + 1];    
    __shared__ ValueType vals[BLOCK_SIZE + 1];    

    const IndexType end = num_warps - (num_warps & (BLOCK_SIZE - 1));

    if (threadIdx.x == 0)
    {
        indices[BLOCK_SIZE] = (IndexType) -1;
        vals[BLOCK_SIZE] = (ValueType)  0;
    }
    
    __syncthreads();

    IndexType i = threadIdx.x;

    while (i < end)
    {
        // do full blocks
        indices[threadIdx.x] = temp_indices[i];
        vals[threadIdx.x] = temp_vals[i];

        __syncthreads();

        segreduce_block(indices, vals);

        if (indices[threadIdx.x] != indices[threadIdx.x + 1])
            X[indices[threadIdx.x]] += vals[threadIdx.x];

        __syncthreads();

        i += BLOCK_SIZE; 
    }

    if (end < num_warps){
        if (i < num_warps){
            indices[threadIdx.x] = temp_indices[i];
            vals[threadIdx.x] = temp_vals[i];
        } else {
            indices[threadIdx.x] = (IndexType) -1;
            vals[threadIdx.x] = (ValueType)  0;
        }

        __syncthreads();
   
        segreduce_block(indices, vals);

        if (i < num_warps)
            if (indices[threadIdx.x] != indices[threadIdx.x + 1])
                X[indices[threadIdx.x]] += vals[threadIdx.x];
    }
}

template <typename IndexType, typename ValueType>
__global__ void
count_sorted_indices_serial_kernel(	const IndexType size,
                    				const IndexType * I,
                    				IndexType * X)
{
    for(IndexType n = 0; n < size; n++)
    {
		X[I[n]] += 1;
    }
}


#endif