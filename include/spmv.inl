#ifndef SPMV_DEVICE
#define SPMV_DEVICE

#define QUEUE_SIZE		512

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_ellb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE num_cols_per_row,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * Aj,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE invalid_index = cusp::ell_matrix<int, INDEX_TYPE, cusp::device_memory>::invalid_index;

    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	VALUE_TYPE sum = 0;
        INDEX_TYPE offset = row;
        for(INDEX_TYPE n = 0; n < num_cols_per_row; n++)
        {
            const INDEX_TYPE col = Aj[offset];
            if(col != invalid_index)
            {
	            if(x[col] != 0)
	            	sum = 1;
            }

            offset += pitch;
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_hybb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE num_cols_per_row,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * A_ell_column_indices,
        	const INDEX_TYPE * A_coo_row_indices,
            const INDEX_TYPE * A_coo_column_indices,
            const INDEX_TYPE * A_rs,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE invalid_index = cusp::ell_matrix<int, INDEX_TYPE, cusp::device_memory>::invalid_index;

    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	VALUE_TYPE sum = 0;
        INDEX_TYPE offset = row;
        INDEX_TYPE rl = A_rs[row];
        INDEX_TYPE r_idx = 0;

        for(INDEX_TYPE n = 0; n < num_cols_per_row && r_idx < rl; ++n, ++r_idx)
        {
            const INDEX_TYPE col = A_ell_column_indices[offset];
            if(col != invalid_index)
            {
	            if(x[col] != 0)
	            {
                	sum = 1;
                    break;
                }
            }
            else
            	break;

            offset += pitch;
        }

        int overflow_size = A_coo_column_indices[0];
        for(int n=1; n <= overflow_size && r_idx < rl; n++)
        {
        	if(A_coo_row_indices[n] == row)
        	{
                r_idx++;
        		const INDEX_TYPE col = A_coo_column_indices[n];
        		if(x[col] != 0)
	            {
                	sum = 1;
                    break;
                }
        	}
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_hyb(   const INDEX_TYPE num_rows,
            const INDEX_TYPE num_cols_per_row,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * A_ell_column_indices,
            const VALUE_TYPE * A_ell_values,
            const INDEX_TYPE * A_coo_row_indices,
            const INDEX_TYPE * A_coo_column_indices,
            const VALUE_TYPE * A_coo_values,
            const INDEX_TYPE * A_rs,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE invalid_index = cusp::ell_matrix<int, INDEX_TYPE, cusp::device_memory>::invalid_index;
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
        VALUE_TYPE sum = 0;
        INDEX_TYPE offset = row;
        INDEX_TYPE rl = A_rs[row];
        INDEX_TYPE r_idx = 0;

        for(INDEX_TYPE n = 0; n < num_cols_per_row && r_idx < rl; ++n, ++r_idx)
        {
            const INDEX_TYPE col = A_ell_column_indices[offset];
            if(col != invalid_index)
            {
                sum += A_ell_values[offset] * x[col];
            }
            else
                break;

            offset += pitch;
        }

        int overflow_size = A_coo_column_indices[0];
        for(int n=1; n <= overflow_size && r_idx < rl; n++)
        {
            if(A_coo_row_indices[n] == row)
            {
                r_idx++;
                sum += A_coo_values[n] * x[A_coo_column_indices[n]];
            }
        }

        y[row] = sum;
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_csrb(	const INDEX_TYPE num_rows,
            const INDEX_TYPE * A_row_offsets,
			const INDEX_TYPE * A_column_indices,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
    	INDEX_TYPE row_start = A_row_offsets[row];
    	INDEX_TYPE row_end = A_row_offsets[row + 1];

    	VALUE_TYPE sum = 0;
        for(INDEX_TYPE j=row_start; j < row_end; ++j)
        {
        	INDEX_TYPE col = A_column_indices[j];
        	if(x[col] != 0)
	    		sum = 1;
        }

        y[row] = sum;
	}
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmv_csr(   const INDEX_TYPE num_rows,
            const INDEX_TYPE * A_row_offsets,
            const INDEX_TYPE * A_column_indices,
            const VALUE_TYPE * A_values,
            const VALUE_TYPE * x, 
                  VALUE_TYPE * y)
{
    const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
    const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

    for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
    {
        INDEX_TYPE row_start = A_row_offsets[row];
        INDEX_TYPE row_end = A_row_offsets[row + 1];

        VALUE_TYPE sum = 0;
        for(INDEX_TYPE j=row_start; j < row_end; ++j)
        {
            sum += A_values[j] * x[A_column_indices[j]];
        }

        y[row] = sum;
    }
}

// template <typename INDEX_TYPE, typename VALUE_TYPE>
// __launch_bounds__(BLOCK_THREADS_MAX,1)
// __global__ void
// spmv_dcsr_b(    const INDEX_TYPE num_rows,
//                 const INDEX_TYPE chunk_width,
//                 const INDEX_TYPE pitch,
//                 const INDEX_TYPE * Matrix_MD,
//                 const INDEX_TYPE * A_ci,
//                 const INDEX_TYPE * A_cl,
//                 const INDEX_TYPE * A_ca,
//                 const INDEX_TYPE * A_rs,
//                 const INDEX_TYPE * A_cols,  
//                 const VALUE_TYPE * x,
//                       VALUE_TYPE * y)
// {
//     const INDEX_TYPE tID = blockDim.x * blockIdx.x + threadIdx.x;
//     const INDEX_TYPE grid_size = gridDim.x * blockDim.x;

//     for(INDEX_TYPE row=tID; row < num_rows; row += grid_size)
//     {
//         INDEX_TYPE rl = A_rs[row];
//         VALUE_TYPE sum = 0;
//         INDEX_TYPE r_idx = 0;
//         bool next_chunk = false;

//         do
//         {
//             INDEX_TYPE cID = row / chunk_width;
//             INDEX_TYPE next_cID = A_ci[cID];
//             INDEX_TYPE offset = A_ca[cID] + (row % chunk_width);

//             for(INDEX_TYPE c_idx = 0; c_idx < A_cl[cID]; ++c_idx, ++r_idx)
//             {
//                 INDEX_TYPE col = A_cols[offset + c_idx*pitch];
//                 if(x[col] != 0)
//     	    	{
//                 	sum = 1;
//                     break;          //break out because it is a binary matrix and the value of this dot product is 1
//                 }
//             }

//             if(next_cID > 0 && r_idx < rl && sum == 0)
//             {
//                 next_chunk = true;
//                 cID = next_cID;
//             }
//             else
//                 next_chunk = false;

//         } while(next_chunk);
//         y[row] = sum;
// 	}
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
spmv_dcsr(  const INDEX_TYPE num_rows,
            const INDEX_TYPE pitch,
            const INDEX_TYPE * A_row_offsets,
            const INDEX_TYPE * A_row_sizes,
            const INDEX_TYPE * Aj,
            const VALUE_TYPE * Ax,
            const VALUE_TYPE * x,
                  VALUE_TYPE * y)
{
    __shared__ volatile VALUE_TYPE sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];

    const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
    {
    //     INDEX_TYPE r_idx = 0;
    //     const INDEX_TYPE rl = A_row_sizes[row];

    //     // initialize local sum
    //     VALUE_TYPE sum = 0;

    //     #pragma unroll
    //     for(INDEX_TYPE offset = 0; offset < BINS && r_idx < rl; offset++)
    //     {
    //         // use two threads to fetch A_row_offsets[row] and A_row_offsets[row+1]
    //         // this is considerably faster than the straightforward version
    //         if(THREADS_PER_VECTOR >= 2)
    //         {
    //             if(thread_lane < 2)
    //                 ptrs[vector_lane][thread_lane] = A_row_offsets[offset*pitch + row*2 + thread_lane];
    //         }
    //         else
    //         {
    //             ptrs[vector_lane][0] = A_row_offsets[offset*pitch + row*2];
    //             ptrs[vector_lane][1] = A_row_offsets[offset*pitch + row*2 + 1];
    //         }

    //         const INDEX_TYPE row_start = ptrs[vector_lane][0];                                  //same as: row_start = A_row_offsets[row];
    //         const INDEX_TYPE row_end   = min(ptrs[vector_lane][1], row_start + rl - r_idx);     //same as: row_end   = A_row_offsets[row+1];

    //         if(THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
    //         {
    //             // ensure aligned memory access to Aj and Ax
    //             INDEX_TYPE jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

    //             // accumulate local sums
    //             if(jj >= row_start && jj < row_end)
    //                 sum += Ax[jj] * x[Aj[jj]];

    //             // accumulate local sums
    //             for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
    //                 sum += Ax[jj] * x[Aj[jj]];
    //         }
    //         else
    //         {
    //             // accumulate local sums
    //             for(INDEX_TYPE jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
    //                 sum += Ax[jj] * x[Aj[jj]];
    //         }

    //         r_idx += (row_end - row_start);
    //     }

    //     // store local sum in shared memory
    //     sdata[threadIdx.x] = sum;

    //     // reduce local sums to row sum
    //     if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
    //     if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
    //     if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
    //     if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
    //     if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
       
    //     // first thread writes the result
    //     if (thread_lane == 0)
    //         y[row] = sdata[threadIdx.x];

        /* TEST */
        if(thread_lane == 0)
            y[row] = 1;
        /* TEST */
    }

}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
spmv_dcsr_bin_scalar(   const INDEX_TYPE num_rows,
                        const INDEX_TYPE pitch,
                        const INDEX_TYPE * A_row_offsets,
                        const INDEX_TYPE * A_Prow_sizes,
                        const INDEX_TYPE * Aj,
                        const VALUE_TYPE * Ax,
                        const VALUE_TYPE * x, 
                              VALUE_TYPE * y,
                        const INDEX_TYPE * row_ids)
{
    const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK;
    const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE vector_id   = thread_id;                                       // global vector index
    //const INDEX_TYPE vector_lane = threadIdx.x;                                     // vector index within the block
    const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    INDEX_TYPE start_range = c_bin_offsets[0];
    INDEX_TYPE end_range = c_bin_offsets[1];

    for(INDEX_TYPE iter = start_range + vector_id; iter < end_range; iter += num_vectors)
    {
        INDEX_TYPE r_idx = 0;
        const INDEX_TYPE row = row_ids[iter];
        const INDEX_TYPE rl = A_Prow_sizes[iter];

        // initialize local sum
        VALUE_TYPE sum = 0;

        #pragma unroll
        for(INDEX_TYPE offset = 0; offset < BINS && r_idx < rl; offset++)
        {
            const INDEX_TYPE row_start = A_row_offsets[offset*pitch + row*2];
            const INDEX_TYPE row_end = min(A_row_offsets[offset*pitch + row*2 + 1], row_start + rl - r_idx);     //same as: row_end   = A_row_offsets[row+1];

            // accumulate local sums
            for(INDEX_TYPE jj = row_start; jj < row_end; jj++)
                sum += Ax[jj] * x[Aj[jj]];

            r_idx += (row_end - row_start);
        }
       
        //write the result
        y[row] = sum;
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
spmv_dcsr_bin(  const INDEX_TYPE num_rows,
                const INDEX_TYPE pitch,
                const INDEX_TYPE * A_row_offsets,
                const INDEX_TYPE * A_Prow_sizes,
                const INDEX_TYPE * Aj,
                const VALUE_TYPE * Ax,
                const VALUE_TYPE * x, 
                      VALUE_TYPE * y,
                const INDEX_TYPE * row_ids)
{
    __shared__ volatile VALUE_TYPE sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];

    const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    //__shared__ volatile INDEX_TYPE start_range[VECTORS_PER_BLOCK], end_range[VECTORS_PER_BLOCK];
    INDEX_TYPE start_range, end_range;

    switch(THREADS_PER_VECTOR)
    {
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
            end_range = c_bin_offsets[6];
            break;
    }

    for(INDEX_TYPE iter = start_range + vector_id; iter < end_range; iter += num_vectors)
    {
        INDEX_TYPE r_idx = 0;
        const INDEX_TYPE row = row_ids[iter];
        const INDEX_TYPE rl = A_Prow_sizes[iter];

        // initialize local sum
        VALUE_TYPE sum = 0;

        #pragma unroll
        for(INDEX_TYPE offset = 0; offset < BINS && r_idx < rl; offset++)
        {
            // use two threads to fetch A_row_offsets[row] and A_row_offsets[row+1]
            // this is considerably faster than the straightforward version
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = A_row_offsets[offset*pitch + row*2 + thread_lane];

            const INDEX_TYPE row_start = ptrs[vector_lane][0];                                  //same as: row_start = A_row_offsets[row];
            const INDEX_TYPE row_end   = min(ptrs[vector_lane][1], row_start + rl - r_idx);     //same as: row_end   = A_row_offsets[row+1];

            if(THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
            {
                // ensure aligned memory access to Aj and Ax
                INDEX_TYPE jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

                // accumulate local sums
                if(jj >= row_start && jj < row_end)
                    sum += Ax[jj] * x[Aj[jj]];

                // accumulate local sums
                for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                    sum += Ax[jj] * x[Aj[jj]];
            }
            else
            {
                // accumulate local sums
                for(INDEX_TYPE jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                    sum += Ax[jj] * x[Aj[jj]];
            }

            r_idx += (row_end - row_start);
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
       
        // first thread writes the result
        if(thread_lane == 0)
            y[row] = sdata[threadIdx.x];
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(512,1)
__global__ void
spmv_dcsr_bin_large(    const INDEX_TYPE num_rows,
                        const INDEX_TYPE pitch,
                        const INDEX_TYPE * A_row_offsets,
                        const INDEX_TYPE * A_Prow_sizes,
                        const INDEX_TYPE * Aj,
                        const VALUE_TYPE * Ax,
                        const VALUE_TYPE * x, 
                              VALUE_TYPE * y,
                        const INDEX_TYPE * row_ids)
{
    //multi warp version (1 vector per block)
    const INDEX_TYPE WARPS_PER_BLOCK = THREADS_PER_VECTOR / WARP_SIZE;
    __shared__ volatile VALUE_TYPE sdata[THREADS_PER_VECTOR + WARP_SIZE / 2];       // padded to avoid reduction conditionals

    //const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;  // global thread index
    const INDEX_TYPE thread_lane = threadIdx.x;                                     // thread index within the vector
    const INDEX_TYPE vector_id   = blockIdx.x;                                      // global vector index
    const INDEX_TYPE warp_id     = threadIdx.x / WARP_SIZE;                         // warp id
    const INDEX_TYPE warp_lane   = threadIdx.x & (WARP_SIZE - 1);                   // lane id within warp
    const INDEX_TYPE num_vectors = gridDim.x;                                       // total number of active vectors

    __shared__ volatile INDEX_TYPE bins[WARPS_PER_BLOCK][2];
    //__shared__ volatile INDEX_TYPE start_range[VECTORS_PER_BLOCK], end_range[VECTORS_PER_BLOCK];
    INDEX_TYPE start_range, end_range;
    start_range = c_bin_offsets[6];
    end_range = c_bin_offsets[7];

    for(INDEX_TYPE iter = start_range + vector_id; iter < end_range; iter += num_vectors)
    {
        const INDEX_TYPE row = row_ids[iter];
        const INDEX_TYPE rl = A_Prow_sizes[iter];

        INDEX_TYPE idx = 0;

        // initialize local sum
        VALUE_TYPE sum = 0;

        #pragma unroll
        for(INDEX_TYPE offset = 0; offset < BINS && idx < rl; offset++)
        {
            if(warp_lane < 2)
                bins[warp_id][warp_lane] = A_row_offsets[offset*pitch + row*2 + warp_lane];

            const INDEX_TYPE row_start = bins[warp_id][0];
            const INDEX_TYPE row_end = min(bins[warp_id][1], row_start + rl - idx);

            // ensure aligned memory access to Aj and Ax
            INDEX_TYPE jj = row_start - (row_start & (31)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum += Ax[jj] * x[Aj[jj]];

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += Ax[jj] * x[Aj[jj]];

            idx += (row_end - row_start);
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;

        // reduce local sums to row sum
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
       
        __syncthreads();

        // first warp writes the result
        if(warp_id == 0)
        {
            //sm_30 //
            // if(warp_lane < WARPS_PER_BLOCK)
            //     sum = sdata[warp_lane*WARP_SIZE];
            // else
            //     sum = 0;

            // reduce local sums to row sum
            //warpScanDown32(sum);
            //sm_30 //

            //hack for sm_20 architecture machines
            if(warp_lane == 0)
            {
                sum = 0;
                for(int i=0; i<WARPS_PER_BLOCK; i++)
                    sum += sdata[i*WARP_SIZE];
            }

            if(warp_lane == 0)
                y[row] = sum;
        }
    }
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
// __launch_bounds__(BLOCK_THREADS_MAX,1)
// __global__ void
// spmv_dcsr_sorted(   const INDEX_TYPE num_rows,
//                     const INDEX_TYPE pitch,
//                     const INDEX_TYPE * A_row_offsets,
//                     const INDEX_TYPE * A_row_sizes,
//                     const INDEX_TYPE * Aj,
//                     const VALUE_TYPE * Ax,
//                     const VALUE_TYPE * x, 
//                           VALUE_TYPE * y)
// {
//     __shared__ volatile VALUE_TYPE sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
//     __shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];

//     const INDEX_TYPE THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

//     const INDEX_TYPE thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
//     const INDEX_TYPE thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
//     const INDEX_TYPE vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
//     const INDEX_TYPE vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
//     const INDEX_TYPE num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

//     for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
//     {
//         INDEX_TYPE r_idx = 0;
//         const INDEX_TYPE rl = A_row_sizes[row];

//         // initialize local sum
//         VALUE_TYPE sum = 0;

//         // use two threads to fetch A_row_offsets[row] and A_row_offsets[row+1]
//         // this is considerably faster than the straightforward version
//         #if(THREADS_PER_VECTOR >= 2)
//         if(thread_lane < 2)
//         {
//             ptrs[vector_lane][thread_lane] = A_row_offsets[row*2 + thread_lane];
//         }
//         #else
//             ptrs[vector_lane][0] = A_row_offsets[row*2];
//             ptrs[vector_lane][1] = A_row_offsets[row*2 + 1];
//         #endif

//         const INDEX_TYPE row_start = ptrs[vector_lane][0];                                  //same as: row_start = A_row_offsets[row];
//         const INDEX_TYPE row_end   = min(ptrs[vector_lane][1], row_start + rl - r_idx);     //same as: row_end   = A_row_offsets[row+1];

//         if(THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
//         {
//             // ensure aligned memory access to Aj and Ax
//             INDEX_TYPE jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

//             // accumulate local sums
//             if(jj >= row_start && jj < row_end)
//                 sum += Ax[jj] * x[Aj[jj]];

//             // accumulate local sums
//             for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
//                 sum += Ax[jj] * x[Aj[jj]];
//         }
//         else
//         {
//             // accumulate local sums
//             for(INDEX_TYPE jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
//                 sum += Ax[jj] * x[Aj[jj]];
//         }

//         r_idx += (row_end - row_start);

//         // store local sum in shared memory
//         sdata[threadIdx.x] = sum;

//         // reduce local sums to row sum
//         if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
//         if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
//         if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
//         if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
//         if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
       
//         // first thread writes the result
//         if (thread_lane == 0)
//             y[row] = sdata[threadIdx.x];
//     }
// }

template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(VECTORS_PER_BLOCK * THREADS_PER_VECTOR,1)
__global__ void
spmv_csr_vector_kernel(const IndexType num_rows,
                       const IndexType * Ap, 
                       const IndexType * Aj, 
                       const ValueType * Ax, 
                       const ValueType * x, 
                             ValueType * y)
{
    __shared__ volatile ValueType sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];
    
    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        #if(THREADS_PER_VECTOR >= 2)
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[row + thread_lane];
        #else
            ptrs[vector_lane][0] = Ap[row];
            ptrs[vector_lane][1] = Ap[row + 1];
        #endif

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // initialize local sum
        ValueType sum = 0;
     
        if (THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
        {
            // ensure aligned memory access to Aj and Ax
            IndexType jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(jj >= row_start && jj < row_end)
                sum += Ax[jj] * x[Aj[jj]];

            // accumulate local sums
            for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += Ax[jj] * x[Aj[jj]];
        }
        else
        {
            // accumulate local sums
            for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                sum += Ax[jj] * x[Aj[jj]];
        }

        // store local sum in shared memory
        sdata[threadIdx.x] = sum;
        
        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        if (THREADS_PER_VECTOR >  8) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        if (THREADS_PER_VECTOR >  4) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        if (THREADS_PER_VECTOR >  2) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        if (THREADS_PER_VECTOR >  1) sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
       
        // first thread writes the result
        if (thread_lane == 0)
            y[row] = sdata[threadIdx.x];
    }
}

}	//namespace device

#endif