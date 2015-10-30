#ifndef SPMM_DEVICE
#define SPMM_DEVICE

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR, unsigned int SORT_SIZE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmm_dcsr_kernel( const INDEX_TYPE M,
                  const INDEX_TYPE N,
                  const INDEX_TYPE K,
                  const INDEX_TYPE pitchA,
                  const INDEX_TYPE pitchB,
                  const INDEX_TYPE pitchC,
                  const INDEX_TYPE * A_row_offsets,
                  const INDEX_TYPE * A_row_sizes,
                  const INDEX_TYPE * Aj,
                  const VALUE_TYPE * Ax,
                  const INDEX_TYPE * B_row_offsets,
                  const INDEX_TYPE * B_row_sizes,
                  const INDEX_TYPE * Bj,
                  const VALUE_TYPE * Bx,
                  const INDEX_TYPE * MM_rows,
                  const INDEX_TYPE * MM_sizes,
                  const INDEX_TYPE * MM_bins,
                  INDEX_TYPE * C_row_offsets,
                  INDEX_TYPE * C_row_sizes,
                  INDEX_TYPE * Cj,
                  VALUE_TYPE * Cx)
{
    const INDEX_TYPE THREADS_PER_BLOCK  = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const INDEX_TYPE thread_id          = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const INDEX_TYPE thread_lane        = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const INDEX_TYPE vector_id          = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const INDEX_TYPE vector_lane        = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const INDEX_TYPE num_vectors        = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors
    const INDEX_TYPE warp_id            = threadIdx.x / WARP_SIZE;
    const INDEX_TYPE warp_lane          = threadIdx.x & (WARP_SIZE-1);
    const INDEX_TYPE num_warps          = THREADS_PER_BLOCK / WARP_SIZE;

#if(SORT == BITONIC)
    __shared__ volatile INDEX_TYPE  s_j[SORT_SIZE * VECTORS_PER_BLOCK];       //column indices
    __shared__ volatile VALUE_TYPE  s_x[SORT_SIZE * VECTORS_PER_BLOCK];       //values
#elif(SORT == RADIX)
    __shared__ volatile INDEX_TYPE  s_j[SORT_SIZE * VECTORS_PER_BLOCK * 2];       //column indices
    __shared__ volatile VALUE_TYPE  s_x[SORT_SIZE * VECTORS_PER_BLOCK * 2];       //values
#endif

    __shared__ volatile INDEX_TYPE  Aptrs[VECTORS_PER_BLOCK][2];
    __shared__ volatile INDEX_TYPE  Bptrs[VECTORS_PER_BLOCK][2];
    __shared__ volatile INDEX_TYPE  s_pos[num_warps];
    __shared__ volatile INDEX_TYPE  s_carryIdx[num_warps];
    __shared__ volatile VALUE_TYPE  s_carryVal[num_warps];

    // Specialize WarpReduce for type int
    typedef cub::WarpReduce<int> WarpReduce;
    // Allocate WarpReduce shared memory for one warp
    __shared__ typename WarpReduce::TempStorage temp_storage;

    INDEX_TYPE start_range, end_range;
    switch(SORT_SIZE)
    {
        case 32:
            start_range = MM_bins[1];
            end_range = MM_bins[2];
            break;
        case 64:
            start_range = MM_bins[2];
            end_range = MM_bins[3];
            break;
        case 128:
            start_range = MM_bins[3];
            end_range = MM_bins[4];
            break;
        case 256:
            start_range = MM_bins[4];
            end_range = MM_bins[5];
            break;
        case 512:
            start_range = MM_bins[5];
            end_range = MM_bins[6];
            break;
        case 1024:
            start_range = MM_bins[6];
            end_range = MM_bins[7];
            break;
        case 2048:
            start_range = MM_bins[7];
            end_range = MM_bins[8];
            break;
        default:
            break;
    }
    __syncthreads();

    //Search row i of A matrix
    for(INDEX_TYPE iter=start_range + vector_id; iter < end_range; iter+=num_vectors)
    {
        INDEX_TYPE Arow = MM_rows[iter];
        //INDEX_TYPE MMrl = MM_sizes[iter];

        INDEX_TYPE Ar_idx = 0;
        const INDEX_TYPE Arl = A_row_sizes[Arow];

        //compute partial products for row i
        #pragma unroll
        for(INDEX_TYPE Aoffset = 0; Aoffset < BINS && Ar_idx < Arl; Aoffset++)
        {
            if(thread_lane < 2)
                Aptrs[vector_lane][thread_lane] = A_row_offsets[Aoffset*pitchA + Arow*2 + thread_lane];

            const INDEX_TYPE A_row_start = Aptrs[vector_lane][0];                                     //same as: row_start = A_row_offsets[row];
            const INDEX_TYPE A_row_end   = min(Aptrs[vector_lane][1], A_row_start + Arl - Ar_idx);    //same as: row_end   = A_row_offsets[row+1];
            INDEX_TYPE sk = vector_lane*SORT_SIZE;

            for(INDEX_TYPE jj=A_row_start; jj < A_row_end; jj++)
            {
                INDEX_TYPE Brow = Aj[jj];
                INDEX_TYPE Br_idx = 0;
                const INDEX_TYPE Brl = B_row_sizes[Brow];
                const VALUE_TYPE Aval = Ax[jj];

                //Check entries in row j of B matrix
                #pragma unroll
                for(INDEX_TYPE Boffset = 0; Boffset < BINS && Br_idx < Brl; Boffset++)
                {
                    if(thread_lane < 2)
                        Bptrs[vector_lane][thread_lane] = B_row_offsets[Boffset*pitchB + Brow*2 + thread_lane];

                    const INDEX_TYPE B_row_start = Bptrs[vector_lane][0];
                    const INDEX_TYPE B_row_end   = min(Bptrs[vector_lane][1], B_row_start + Brl - Br_idx);

                    if(THREADS_PER_VECTOR == 32 && B_row_end - B_row_start > 32)
                    {
                        // ensure aligned memory access to Aj and Ax
                        INDEX_TYPE kk = B_row_start - (B_row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;
                        sk = __shfl(sk, warp_lane - thread_lane) + thread_lane;

                        // accumulate local sums
                        if(kk >= B_row_start && kk < B_row_end)
                        {
                            s_x[sk - (B_row_start & (THREADS_PER_VECTOR - 1))] = Bx[kk] * Aval;
                            s_j[sk - (B_row_start & (THREADS_PER_VECTOR - 1))] = Bj[kk];
                        }
                        sk += THREADS_PER_VECTOR - (B_row_start & (THREADS_PER_VECTOR - 1));

                        // accumulate local sums
                        for(kk += THREADS_PER_VECTOR; kk < B_row_end; kk += THREADS_PER_VECTOR)
                        {
                            s_x[sk] = Bx[kk] * Aval;
                            s_j[sk] = Bj[kk];

                            sk += min((B_row_end - (kk - thread_lane)), THREADS_PER_VECTOR);
                        }
                    }
                    else
                    {
                        sk = __shfl(sk, warp_lane - thread_lane) + thread_lane;
                        // accumulate local sums
                        for(INDEX_TYPE kk = B_row_start + thread_lane; kk < B_row_end; kk += THREADS_PER_VECTOR)
                        {
                            s_x[sk] = Bx[kk] * Aval;
                            s_j[sk] = Bj[kk];

                            sk += min((B_row_end - (kk - thread_lane)), THREADS_PER_VECTOR);
                        }
                    }
                    Br_idx += (B_row_end - B_row_start);
                }
            }
            Ar_idx += (A_row_end - A_row_start);
        }

        // __syncthreads();
        // sort and reduce partial products

        //for(int i=warp_id*(WARP_SIZE/THREADS_PER_VECTOR); i < warp_id*(WARP_SIZE/THREADS_PER_VECTOR) + (WARP_SIZE/THREADS_PER_VECTOR); i++)
        for(int i=warp_id; i < VECTORS_PER_BLOCK; i+=num_warps)
        {
            if(thread_lane == 0)
                s_carryIdx[i] = -1;

            if(MM_sizes[iter] < SORT_SIZE)
            {
                for(int k=MM_sizes[iter] + thread_lane; k < SORT_SIZE; k+=WARP_SIZE)
                {
                    s_j[i*SORT_SIZE + k] = INT_MAX;
                }
            }

            //Sort elements
#if(SORT == BITONIC)
            switch(SORT_SIZE)
            {
                case 32:
                    bitonicSort32_Key(&s_j[i*SORT_SIZE], &s_x[i*SORT_SIZE], thread_lane);
                    break;
                case 64:
                    bitonicSort64_Key(&s_j[i*SORT_SIZE], &s_x[i*SORT_SIZE], thread_lane);
                    break;
                case 128:
                    bitonicSort128_Key(&s_j[i*SORT_SIZE], &s_x[i*SORT_SIZE], thread_lane);
                    break;
                case 256:
                    bitonicSort256_Key(&s_j[i*SORT_SIZE], &s_x[i*SORT_SIZE], thread_lane);
                    break;
                case 512:
                    bitonicSort512_Key(&s_j[i*SORT_SIZE], &s_x[i*SORT_SIZE], thread_lane);
                    break;
                case 1024:
                    bitonicSort1024_Key(&s_j[i*SORT_SIZE], &s_x[i*SORT_SIZE], thread_lane);
                    break;
                // case 2048:
                //     warpSort2048_Key(&s_j[i*SORT_SIZE], &s_x[i*SORT_SIZE], thread_lane);
                //     break;
                default:
                    break;
            }
#elif(SORT == RADIX)


#endif


            //Reduce Elements
            if(thread_lane == 0)
                s_pos[i] = i*SORT_SIZE;

            for(int k=i*SORT_SIZE + thread_lane; k<i*SORT_SIZE + SORT_SIZE; k+=WARP_SIZE)
            {
                char head = 0, pos = 0;
                INDEX_TYPE idx = s_j[k];
                VALUE_TYPE val = s_x[k];
                if(thread_lane > 0)
                {
                    if(idx != s_j[k-1])
                        head = 1;
                }
                else
                    head = 1;

                typedef cub::WarpReduce<int> WarpReduce;
                val = WarpReduce(temp_storage).HeadSegmentedSum(val, head);
                if(idx == INT_MAX)
                    pos = head = 0;
                else
                    pos = head;

                if(thread_lane == 0 && idx == s_carryIdx[i])
                {
                    pos = 0;
                    val += s_carryVal[i];
                }
                warpScanUp32(pos, thread_lane);

                //write idx and val to shared memory buffers
                if(head && idx != INT_MAX)
                {
                    s_j[s_pos[i]+pos-1] = idx;
                    s_x[s_pos[i]+pos-1] = val;
                }

                //write carry values
                if(head && idx == s_j[k + WARP_SIZE-thread_lane-1])
                {
                    s_carryIdx[i] = idx;
                    s_carryVal[i] = val;
                }
                //update running position index
                if(thread_lane == WARP_SIZE-1)
                    s_pos[i] += pos;
            }

            //store results in C matrix
            INDEX_TYPE C_row_start, C_row_end;
            if(thread_lane == 0)
            {
                //INDEX_TYPE Crl = (s_pos[i] - i*SORT_SIZE);
                INDEX_TYPE new_size = (s_pos[i] - i*SORT_SIZE);//ALIGN_UP(Crl, 32);
                INDEX_TYPE new_addr = atomicAdd(&C_row_sizes[M], new_size);  //increase global memory pointer

                C_row_start = new_addr;
                C_row_end = new_addr + (s_pos[i] - i*SORT_SIZE);
                C_row_offsets[Arow*2] = new_addr;
                C_row_offsets[Arow*2 + 1] = new_addr + new_size;
            }
            C_row_start = __shfl(C_row_start, 0);
            C_row_end = __shfl(C_row_end, 0);
            
            for(INDEX_TYPE C_idx = C_row_start+thread_lane, S_idx = i*SORT_SIZE+thread_lane; C_idx < C_row_end; C_idx+=WARP_SIZE, S_idx+=WARP_SIZE)
            {
                Cj[C_idx] = s_j[S_idx];
                Cx[C_idx] = s_x[S_idx];
            }
            if(thread_lane == 0)
                C_row_sizes[Arow] = (s_pos[i] - i*SORT_SIZE);
        }
        //__syncthreads();
    }
}

//caclulate partial row sizes for an MxK * KxN matrix multiplciation
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
CalcSortSizes_SPMM( const INDEX_TYPE M,
                    const INDEX_TYPE N,
                    const INDEX_TYPE K,
                    const INDEX_TYPE pitchA,
                    const INDEX_TYPE * A_row_offsets,
                    const INDEX_TYPE * A_row_sizes,
                    const INDEX_TYPE * Aj,
                    const INDEX_TYPE * B_row_sizes,
                    INDEX_TYPE *MM_row_sizes)
{
    __shared__ volatile INDEX_TYPE ptrs[VECTORS_PER_BLOCK][2];

    const INDEX_TYPE THREADS_PER_BLOCK  = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const INDEX_TYPE thread_id          = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;     // global thread index
    const INDEX_TYPE thread_lane        = threadIdx.x & (THREADS_PER_VECTOR - 1);           // thread index within the vector
    const INDEX_TYPE vector_id          = thread_id / THREADS_PER_VECTOR;                   // global vector index
    const INDEX_TYPE vector_lane        = threadIdx.x / THREADS_PER_VECTOR;                 // vector index within the block
    const INDEX_TYPE num_vectors        = VECTORS_PER_BLOCK * gridDim.x;                    // total number of active vectors

    for(INDEX_TYPE row = vector_id; row < M; row += num_vectors)
    {
        INDEX_TYPE r_idx = 0;
        INDEX_TYPE sum = 0;
        const INDEX_TYPE rl = A_row_sizes[row];

        #pragma unroll
        for(INDEX_TYPE offset = 0; offset < BINS && r_idx < rl; offset++)
        {
            // use two threads to fetch A_row_offsets[row] and A_row_offsets[row+1]
            // this is considerably faster than the straightforward version
            if(thread_lane < 2)
                ptrs[vector_lane][thread_lane] = A_row_offsets[offset*pitchA + row*2 + thread_lane];

            const INDEX_TYPE row_start = ptrs[vector_lane][0];                                  //same as: row_start = A_row_offsets[row];
            const INDEX_TYPE row_end   = min(ptrs[vector_lane][1], row_start + rl - r_idx);     //same as: row_end   = A_row_offsets[row+1];

            if(THREADS_PER_VECTOR == 32 && row_end - row_start > 32)
            {
                // ensure aligned memory access to Aj and Ax
                INDEX_TYPE jj = row_start - (row_start & (THREADS_PER_VECTOR - 1)) + thread_lane;

                // accumulate local sums
                if(jj >= row_start && jj < row_end)
                    sum += B_row_sizes[Aj[jj]];

                // accumulate local sums
                for(jj += THREADS_PER_VECTOR; jj < row_end; jj += THREADS_PER_VECTOR)
                    sum += B_row_sizes[Aj[jj]];
            }
            else
            {
                // accumulate local sums
                for(INDEX_TYPE jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
                    sum += B_row_sizes[Aj[jj]];
            }

            r_idx += (row_end - row_start);
        }

        if(THREADS_PER_VECTOR == 32)    sum += __shfl_down(sum, 16, THREADS_PER_VECTOR);
        if(THREADS_PER_VECTOR >= 16)    sum += __shfl_down(sum, 8, THREADS_PER_VECTOR);
        if(THREADS_PER_VECTOR >= 8)     sum += __shfl_down(sum, 4, THREADS_PER_VECTOR);
        if(THREADS_PER_VECTOR >= 4)     sum += __shfl_down(sum, 2, THREADS_PER_VECTOR);
        if(THREADS_PER_VECTOR >= 2)     sum += __shfl_down(sum, 1, THREADS_PER_VECTOR);

        if(thread_lane == 0)
            MM_row_sizes[row] = sum;
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE>
__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void
SetBins_SPMM(   const INDEX_TYPE num_rows,
                INDEX_TYPE *bins,
                INDEX_TYPE *row_ids,
                const INDEX_TYPE *row_sizes)
{
    const int tID = blockDim.x * blockIdx.x + threadIdx.x;                  // global thread index
    const int grid_size = blockDim.x * gridDim.x;

    for(INDEX_TYPE row = tID; row < num_rows; row += grid_size)
    {
        INDEX_TYPE rl = row_sizes[row], bin;
        if(rl == 0)
            bin = 0;
        else if(rl >= 1 && rl <= 32)
            bin = 1;
        else if(rl > 32 && rl <= 64)
            bin = 2;
        else if(rl > 64 && rl <= 128)
            bin = 3;
        else if(rl > 128 && rl <= 256)
            bin = 4;
        else if(rl > 256 && rl <= 512)
            bin = 5;
        else if(rl > 512 && rl <= 1024)
            bin = 6;
        else if(rl > 1024 && rl <= 2048)
            bin = 7;
        else
            bin = 8;

        bins[row] = bin;
        row_ids[row] = row;
    }
}

} //namespace device

#endif