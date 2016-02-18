#pragma once

namespace device
{

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR, unsigned int SORT_SIZE>
__device__ __forceinline__ void ComputePartials(
        const INDEX_TYPE    Arow,
        const INDEX_TYPE    pitchA,
        const INDEX_TYPE    pitchB,
        const INDEX_TYPE    *A_row_offsets,
        const INDEX_TYPE    *A_row_sizes,
        const INDEX_TYPE    *Aj,
        const VALUE_TYPE    *Ax,
        const INDEX_TYPE    *B_row_offsets,
        const INDEX_TYPE    *B_row_sizes,
        const INDEX_TYPE    *Bj,
        const VALUE_TYPE    *Bx,
        INDEX_TYPE          *s_j,
        VALUE_TYPE          *s_x,
        const INDEX_TYPE    thread_lane,
        const INDEX_TYPE    vector_lane,
        const INDEX_TYPE    warp_lane)
{
    INDEX_TYPE Ar_idx = 0;
    const INDEX_TYPE Arl = A_row_sizes[Arow];

    // #pragma unroll
    // for(INDEX_TYPE Aoffset = 0; Aoffset < BINS && Ar_idx < Arl; Aoffset++)
    {
        // if(thread_lane < 2)
        //      Aptrs[vector_lane][thread_lane] = A_row_offsets[Arow*2 + thread_lane];
        //      Aptrs[vector_lane][thread_lane] = A_row_offsets[Aoffset*pitchA + Arow*2 + thread_lane];

        // const INDEX_TYPE A_row_start = Aptrs[vector_lane][0];                                     //same as: row_start = A_row_offsets[row];
        // const INDEX_TYPE A_row_end   = min(Aptrs[vector_lane][1], A_row_start + Arl - Ar_idx);    //same as: row_end   = A_row_offsets[row+1];
        const INDEX_TYPE A_row_start = A_row_offsets[Arow*2];
        const INDEX_TYPE A_row_end   = min(A_row_offsets[Arow*2 + 1], A_row_start + Arl - Ar_idx);
        INDEX_TYPE sk = 0;

        for(INDEX_TYPE jj=A_row_start; jj < A_row_end; jj++)
        {
            INDEX_TYPE Brow = Aj[jj];
            INDEX_TYPE Br_idx = 0;
            const INDEX_TYPE Brl = B_row_sizes[Brow];
            const VALUE_TYPE Aval = Ax[jj];

            //Check entries in row j of B matrix
            //#pragma unroll
            //for(INDEX_TYPE Boffset = 0; Boffset < BINS && Br_idx < Brl; Boffset++)
            {
                //if(thread_lane < 2)
                    //Bptrs[vector_lane][thread_lane] = B_row_offsets[Brow*2 + thread_lane];
                    // Bptrs[vector_lane][thread_lane] = B_row_offsets[Boffset*pitchB + Brow*2 + thread_lane];

                // const INDEX_TYPE B_row_start = B_row_offsets[Brow*2];
                // const INDEX_TYPE B_row_end   = min(Bptrs[vector_lane][1], B_row_start + Brl - Br_idx);
                const INDEX_TYPE B_row_start = B_row_offsets[Brow*2];
                const INDEX_TYPE B_row_end   = min(B_row_offsets[Brow*2 + 1], B_row_start + Brl - Br_idx);

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
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, unsigned int THREADS_PER_BLOCK, unsigned int SORT_SIZE>
__device__ __forceinline__ void ComputePartials_block(
        const INDEX_TYPE    Arow,
        const INDEX_TYPE    pitchA,
        const INDEX_TYPE    pitchB,
        const INDEX_TYPE    *A_row_offsets,
        const INDEX_TYPE    *A_row_sizes,
        const INDEX_TYPE    *Aj,
        const VALUE_TYPE    *Ax,
        const INDEX_TYPE    *B_row_offsets,
        const INDEX_TYPE    *B_row_sizes,
        const INDEX_TYPE    *Bj,
        const VALUE_TYPE    *Bx,
        INDEX_TYPE          *s_j,
        VALUE_TYPE          *s_x,
        const INDEX_TYPE    thread_lane)
{
    INDEX_TYPE Ar_idx = 0;
    const INDEX_TYPE Arl = A_row_sizes[Arow];
    __shared__ INDEX_TYPE sk_temp;

    // #pragma unroll
    // for(INDEX_TYPE Aoffset = 0; Aoffset < BINS && Ar_idx < Arl; Aoffset++)
    {
        // const INDEX_TYPE A_row_start = Aptrs[vector_lane][0];                                     //same as: row_start = A_row_offsets[row];
        // const INDEX_TYPE A_row_end   = min(Aptrs[vector_lane][1], A_row_start + Arl - Ar_idx);    //same as: row_end   = A_row_offsets[row+1];
        const INDEX_TYPE A_row_start = A_row_offsets[Arow*2];
        const INDEX_TYPE A_row_end   = min(A_row_offsets[Arow*2 + 1], A_row_start + Arl - Ar_idx);
        INDEX_TYPE sk = 0;

        for(INDEX_TYPE jj=A_row_start; jj < A_row_end; jj++)
        {
            INDEX_TYPE Brow = Aj[jj];
            INDEX_TYPE Br_idx = 0;
            const INDEX_TYPE Brl = B_row_sizes[Brow];
            const VALUE_TYPE Aval = Ax[jj];
            if(thread_lane == 0)
                sk_temp = sk;
            __syncthreads();

            //Check entries in row j of B matrix
            //#pragma unroll
            //for(INDEX_TYPE Boffset = 0; Boffset < BINS && Br_idx < Brl; Boffset++)
            {
                // const INDEX_TYPE B_row_start = B_row_offsets[Brow*2];
                // const INDEX_TYPE B_row_end   = min(Bptrs[vector_lane][1], B_row_start + Brl - Br_idx);
                const INDEX_TYPE B_row_start = B_row_offsets[Brow*2];
                const INDEX_TYPE B_row_end   = min(B_row_offsets[Brow*2 + 1], B_row_start + Brl - Br_idx);

                sk = sk_temp + thread_lane;
                // accumulate local sums
                for(INDEX_TYPE kk = B_row_start + thread_lane; kk < B_row_end; kk += THREADS_PER_BLOCK)
                {
                    s_x[sk] = Bx[kk] * Aval;
                    s_j[sk] = Bj[kk];

                    sk += min((B_row_end - (kk - thread_lane)), THREADS_PER_BLOCK);
                }
                Br_idx += (B_row_end - B_row_start);
            }
        }
        Ar_idx += (A_row_end - A_row_start);
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int SORT_SIZE>
__device__ __forceinline__ void ReduceElements(
        const int wID,
        const INDEX_TYPE lane,
        const INDEX_TYPE row,
        const INDEX_TYPE M,
        INDEX_TYPE *s_j,
        VALUE_TYPE *s_x,
        INDEX_TYPE *C_row_offsets,
        INDEX_TYPE *C_row_sizes,
        INDEX_TYPE *Cj,
        VALUE_TYPE *Cx)
{
    INDEX_TYPE s_pos = 0;
    INDEX_TYPE s_carryIdx = -1;
    VALUE_TYPE s_carryVal = 0;

    for(int k=lane; k<SORT_SIZE; k+=WARP_SIZE) {
        char head = 0, pos = 0;
        INDEX_TYPE idx = s_j[k];
        VALUE_TYPE val = s_x[k];
        if(lane > 0) {
            if(idx != s_j[k-1])
                head = 1;
        }
        else
            head = 1;

        // Specialize WarpReduce for type int
        typedef cub::WarpReduce<int> WarpReduce;
        // Allocate WarpReduce shared memory for one warp
        __shared__ typename WarpReduce::TempStorage temp_storage;

        typedef cub::WarpReduce<int> WarpReduce;
        val = WarpReduce(temp_storage).HeadSegmentedSum(val, head);
        if(idx == INT_MAX)
            pos = head = 0;
        else
            pos = head;

        if(lane == 0 && idx == s_carryIdx) {
            pos = 0;
            val += s_carryVal;
        }
        warpScanUp32(pos, lane);

        //write idx and val to shared memory buffers if thread has head of segment
        if(head && idx != INT_MAX) {
            s_j[s_pos+pos-1] = idx;
            s_x[s_pos+pos-1] = val;
        }

        //write carry values
        s_carryIdx = idx;
        s_carryVal = val;

        int flags = __ballot(head);
        if(flags) {
            int last = WARP_SIZE-1 - __clz(flags);
            s_carryIdx =  __shfl(s_carryIdx, last);
            s_carryVal =  __shfl(s_carryVal, last);
        }
        else {
            s_carryIdx =  __shfl(s_carryIdx, 0);
            s_carryVal =  __shfl(s_carryVal, 0);   
        }

        //update running position index
        s_pos += pos;
        s_pos = __shfl(s_pos, WARP_SIZE-1);
    }
    __threadfence_block();

    //store results in C matrix
    INDEX_TYPE C_row_start, C_row_end;
    if(lane == 0) {
        // if(s_pos <= 0)
        //     printf("ERROR: row: %d s_pos: %d\n", row, s_pos);
        //printf("row: %d\n", row);
        C_row_sizes[row] = s_pos;
        INDEX_TYPE new_addr = atomicAdd(&C_row_sizes[M], s_pos);  //increase global memory pointer

        // if(new_addr >= 1661856)
        //     printf("ERROR: beyond max memory... %d\n", new_addr);

        C_row_start = new_addr;
        C_row_end = new_addr + s_pos;
        C_row_offsets[row*2] = C_row_start;
        C_row_offsets[row*2 + 1] = C_row_end;
    }
    C_row_start = __shfl(C_row_start, 0);
    C_row_end = __shfl(C_row_end, 0);
    
    for(INDEX_TYPE C_idx = C_row_start+lane, S_idx = lane; C_idx < C_row_end; C_idx+=WARP_SIZE, S_idx+=WARP_SIZE) {
        Cj[C_idx] = s_j[S_idx];
        Cx[C_idx] = s_x[S_idx];
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned int SORT_SIZE, int NT>
__device__ __forceinline__ void CopyToGlobal(
        const INDEX_TYPE row,
        const INDEX_TYPE num_rows,
        const INDEX_TYPE size,
        const INDEX_TYPE *s_j,
        const VALUE_TYPE *s_x,
        INDEX_TYPE *C_row_offsets,
        INDEX_TYPE *C_row_sizes,
        INDEX_TYPE *Cj,
        VALUE_TYPE *Cx)
{
    const int tID = threadIdx.x;
    __shared__ INDEX_TYPE temp[2];

    //store results in C matrix
    INDEX_TYPE C_row_start, C_row_end;
    if(!tID)
    {
        C_row_sizes[row] = size;
        INDEX_TYPE new_addr = atomicAdd(&C_row_sizes[num_rows], size);  //increase global memory pointer

        C_row_start = new_addr;
        C_row_end = new_addr + size;
        C_row_offsets[row*2] = C_row_start;
        C_row_offsets[row*2 + 1] = C_row_end;

        // if(num_rows == 188763 && C_row_end > 3775260)
        //     printf("ERROR: C_row_start: %d  C_row_end: %d\n", C_row_start, C_row_end);

        temp[0] = C_row_start;
        temp[1] = C_row_end;
    }
    __syncthreads();

    C_row_start = temp[0];
    C_row_end = temp[1];
    for(INDEX_TYPE C_idx = C_row_start+tID, S_idx = tID; C_idx < C_row_end; C_idx+=NT, S_idx+=NT)
    {
        // if(num_rows == 188763 && C_idx > 3775260)
        //     printf("ERROR:  C_idx: %d\n", C_idx);

        Cj[C_idx] = s_j[S_idx];
        Cx[C_idx] = s_x[S_idx];
    }
}

template<typename INDEX_TYPE, unsigned int SORT_SIZE>
__device__ __forceinline__ void getRanges(
        INDEX_TYPE &start_range,
        INDEX_TYPE &end_range,
        const INDEX_TYPE *MM_bins)
{
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
        //case 2048:    //hack to fix shared memory problem with double precision
        case 1280:
            start_range = MM_bins[7];
            end_range = MM_bins[8];
            break;
        case 4096:
            start_range = MM_bins[8];
            end_range = MM_bins[9];
            break;
        default:
            break;
    }
    
}

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
    //const INDEX_TYPE num_warps          = THREADS_PER_BLOCK / WARP_SIZE;

    __shared__ INDEX_TYPE s_j[SORT_SIZE * VECTORS_PER_BLOCK];      //column indices
    __shared__ VALUE_TYPE s_x[SORT_SIZE * VECTORS_PER_BLOCK];      //values

    INDEX_TYPE start_range, end_range;
    getRanges<INDEX_TYPE, SORT_SIZE> (start_range, end_range, MM_bins);
    __syncthreads();

    //Search row i of A matrix
    for(INDEX_TYPE iter=start_range + vector_id; iter < end_range; iter+=num_vectors)
    {
        INDEX_TYPE row = MM_rows[iter];
        //INDEX_TYPE MMrl = MM_sizes[iter];

        //compute partial products for row i
        ComputePartials<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, SORT_SIZE>
                (row, pitchA, pitchB, A_row_offsets, A_row_sizes, Aj, Ax, 
                B_row_offsets, B_row_sizes, Bj, Bx, &s_j[vector_lane*SORT_SIZE], &s_x[vector_lane*SORT_SIZE], 
                thread_lane, vector_lane, warp_lane);
        __threadfence_block();

        // sort and reduce partial products
        //for(int i=warp_id*(WARP_SIZE/THREADS_PER_VECTOR); i < warp_id*(WARP_SIZE/THREADS_PER_VECTOR) + (WARP_SIZE/THREADS_PER_VECTOR); i++)
        //for(int i=warp_id; i < VECTORS_PER_BLOCK; i+=num_warps)
        {
            int partial_size = MM_sizes[iter];
            if(partial_size < SORT_SIZE)
            {
                for(int k=partial_size + warp_lane; k < SORT_SIZE; k+=WARP_SIZE)
                    s_j[warp_id*SORT_SIZE + k] = INT_MAX;
            }

            // //Sort elements
            switch(SORT_SIZE) {
                case 32:
                    bitonicSort32_Key(&s_j[warp_id*SORT_SIZE], &s_x[warp_id*SORT_SIZE], warp_lane);
                    break;
                case 64:
                    bitonicSort64_Key(&s_j[warp_id*SORT_SIZE], &s_x[warp_id*SORT_SIZE], warp_lane);
                    break;
                case 128:
                    bitonicSort128_Key(&s_j[warp_id*SORT_SIZE], &s_x[warp_id*SORT_SIZE], warp_lane);
                    break;
                default:
                    break;
            }
            //Iterate<2,SORT_SIZE>::bitonicSortKey(&s_j[warp_id*SORT_SIZE], &s_x[warp_id*SORT_SIZE], warp_lane);

            //Reduce Elements
            ReduceElements<INDEX_TYPE, VALUE_TYPE, SORT_SIZE>
                    (warp_id, warp_lane, row, M, &s_j[vector_lane*SORT_SIZE], &s_x[vector_lane*SORT_SIZE], C_row_offsets, C_row_sizes, Cj, Cx);
        }
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, size_t VECTORS_PER_BLOCK, size_t THREADS_PER_BLOCK, size_t VALUES_PER_THREAD, size_t SORT_SIZE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmm_dcsr_med_kernel(   const INDEX_TYPE M,
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
    //const INDEX_TYPE thread_id          = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; // global thread index
    const INDEX_TYPE thread_lane        = threadIdx.x & (WARP_SIZE - 1);                // thread index within the vector
    const INDEX_TYPE vector_id_zero     = (THREADS_PER_BLOCK * blockIdx.x) / WARP_SIZE; // global vector index of the 1st vector (index 0) in this block
    const INDEX_TYPE vector_lane        = threadIdx.x / WARP_SIZE;                      // vector index within the block
    const INDEX_TYPE num_vectors        = VECTORS_PER_BLOCK * gridDim.x;                // total number of active vectors
    //const INDEX_TYPE warp_id            = threadIdx.x / WARP_SIZE;
    const INDEX_TYPE warp_lane          = threadIdx.x & (WARP_SIZE-1);
    //const INDEX_TYPE num_warps          = THREADS_PER_BLOCK / WARP_SIZE;

    __shared__ INDEX_TYPE  s_j[(SORT_SIZE+1) * VECTORS_PER_BLOCK];       //column indices   1 extra for comparison with end segment
    __shared__ VALUE_TYPE  s_x[(SORT_SIZE+1) * VECTORS_PER_BLOCK];       //values
    __shared__ INDEX_TYPE  s_sort_j[SORT_SIZE];
    __shared__ VALUE_TYPE  s_sort_x[SORT_SIZE];

    typedef cub::BlockRadixSort<INDEX_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD, VALUE_TYPE> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    INDEX_TYPE start_range, end_range;
    getRanges<INDEX_TYPE, SORT_SIZE> (start_range, end_range, MM_bins);
    __syncthreads();

    //Search row i of A matrix
    for(INDEX_TYPE iter=start_range + vector_id_zero; iter < end_range; iter+=num_vectors)
    {
        if(iter + vector_lane < end_range)
        {
            INDEX_TYPE Arow = MM_rows[iter+vector_lane];

            //compute partial products for row i
            ComputePartials<INDEX_TYPE, VALUE_TYPE, BINS, VECTORS_PER_BLOCK, WARP_SIZE, SORT_SIZE>
                    (Arow, pitchA, pitchB, A_row_offsets, A_row_sizes, Aj, Ax, 
                    B_row_offsets, B_row_sizes, Bj, Bx, &s_j[vector_lane*(SORT_SIZE+1)], &s_x[vector_lane*(SORT_SIZE+1)], 
                    thread_lane, vector_lane, warp_lane);
        }
        __syncthreads();

        // sort and reduce partial products
        for(int i=0; i < VECTORS_PER_BLOCK && iter+i < end_range; ++i)
        {
            INDEX_TYPE row = MM_rows[iter+i];
            int partial_size = MM_sizes[iter+i];

            for(int k=partial_size + threadIdx.x; k < SORT_SIZE+1; k+=THREADS_PER_BLOCK)
            {
                s_j[i*(SORT_SIZE+1) + k] = INT_MAX;
            }
            __syncthreads();

            INDEX_TYPE t_key[VALUES_PER_THREAD];
            VALUE_TYPE t_data[VALUES_PER_THREAD];
            //Sort elements
            #pragma unroll
            for(int k=0; k<VALUES_PER_THREAD; ++k)
            {
                t_key[k] = s_j[i*(SORT_SIZE+1) + threadIdx.x + k*THREADS_PER_BLOCK];
                t_data[k] = s_x[i*(SORT_SIZE+1) + threadIdx.x + k*THREADS_PER_BLOCK];
            }
            __syncthreads();

            //radix sort
            BlockRadixSort(temp_storage).Sort(t_key, t_data);

            //move back to shared memory after sort
            #pragma unroll
            for(int k=0; k<VALUES_PER_THREAD; ++k)
            {
                s_j[i*(SORT_SIZE+1) + threadIdx.x*VALUES_PER_THREAD + k] = t_key[k];
                s_x[i*(SORT_SIZE+1) + threadIdx.x*VALUES_PER_THREAD + k] = t_data[k];
            }
            __syncthreads();

            // if(row == 0)
            //     for(int k=threadIdx.x; k<SORT_SIZE; k+=THREADS_PER_BLOCK) {
            //         printf("(%d)  s_j: %d   s_x: %f\n", k, s_j[i*(SORT_SIZE+1) + k], s_x[i*(SORT_SIZE+1) + k]);
            //     }

            //Reduce Elements
            int ThreadCode = 0;
            int segs = KernelReduceByKeyPreprocess<INDEX_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD> (&s_j[i*(SORT_SIZE+1)], s_sort_j, partial_size, ThreadCode);
            KernelSegReduceApply<VALUE_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD> (ThreadCode, segs, &s_x[i*(SORT_SIZE+1)], s_sort_x);

            // if(row == 0 && threadIdx.x == 0)
            //     printf("partial_size: %d row segs: %d\n", partial_size, segs);

            //Copy to global mem
            CopyToGlobal<INDEX_TYPE, VALUE_TYPE, SORT_SIZE, THREADS_PER_BLOCK> (row, M, segs, s_sort_j, s_sort_x, C_row_offsets, C_row_sizes, Cj, Cx);
            __syncthreads();
        }
    }
}

//large dcsr kernel (1 vector = 1 block)
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, size_t THREADS_PER_BLOCK, size_t VALUES_PER_THREAD, size_t SORT_SIZE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmm_dcsr_large_kernel( const INDEX_TYPE M,
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
    //const INDEX_TYPE thread_id          = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; // global thread index
    const INDEX_TYPE thread_lane        = threadIdx.x;                                  // thread index within the vector (which is the block)
    const INDEX_TYPE vector_id          = blockIdx.x;                                   // global vector index
    //const INDEX_TYPE vector_lane        = threadIdx.x / WARP_SIZE;                    // vector index within the block
    const INDEX_TYPE num_vectors        = gridDim.x;                                    // total number of active vectors (1 vector per block for large kernel)

    __shared__ INDEX_TYPE  s_j[SORT_SIZE+1];       //column indices   1 extra for comparison with end segment
    __shared__ VALUE_TYPE  s_x[SORT_SIZE+1];       //values
    __shared__ INDEX_TYPE  s_sort_j[SORT_SIZE];
    __shared__ VALUE_TYPE  s_sort_x[SORT_SIZE];

    typedef cub::BlockRadixSort<INDEX_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD, VALUE_TYPE> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    INDEX_TYPE start_range, end_range;
    getRanges<INDEX_TYPE, SORT_SIZE> (start_range, end_range, MM_bins);
    __syncthreads();

    //Search row i of A matrix
    for(INDEX_TYPE iter=start_range + vector_id; iter < end_range; iter+=num_vectors)
    {
        INDEX_TYPE Arow = MM_rows[iter];

        //compute partial products for row i
        ComputePartials_block<INDEX_TYPE, VALUE_TYPE, BINS, THREADS_PER_BLOCK, SORT_SIZE>
                (Arow, pitchA, pitchB, A_row_offsets, A_row_sizes, Aj, Ax, 
                B_row_offsets, B_row_sizes, Bj, Bx, s_j, s_x, thread_lane);
        __syncthreads();

        // sort and reduce partial products
        {
            INDEX_TYPE row = MM_rows[iter];
            int partial_size = MM_sizes[iter];

            for(int k=partial_size + threadIdx.x; k < SORT_SIZE+1; k+=THREADS_PER_BLOCK)
            {
                s_j[k] = INT_MAX;
            }
            __syncthreads();

            INDEX_TYPE t_key[VALUES_PER_THREAD];
            VALUE_TYPE t_data[VALUES_PER_THREAD];
            //Sort elements
            #pragma unroll
            for(int k=0; k<VALUES_PER_THREAD; ++k)
            {
                t_key[k] = s_j[threadIdx.x + k*THREADS_PER_BLOCK];
                t_data[k] = s_x[threadIdx.x + k*THREADS_PER_BLOCK];
            }
            __syncthreads();

            //radix sort
            BlockRadixSort(temp_storage).Sort(t_key, t_data);

            //move back to shared memory after sort
            #pragma unroll
            for(int k=0; k<VALUES_PER_THREAD; ++k)
            {
                s_j[threadIdx.x*VALUES_PER_THREAD + k] = t_key[k];
                s_x[threadIdx.x*VALUES_PER_THREAD + k] = t_data[k];
            }
            __syncthreads();

            //Reduce Elements
            int ThreadCode = 0;
            int segs = KernelReduceByKeyPreprocess<INDEX_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD> (s_j, s_sort_j, partial_size, ThreadCode);
            KernelSegReduceApply<VALUE_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD> (ThreadCode, segs, s_x, s_sort_x);

            // if(threadIdx.x == 0 && (segs == 0 || partial_size == 0))
            //     printf("row: %d  partial_size: %d segs: %d\n", row, partial_size, segs);

            //Copy to global mem
            CopyToGlobal<INDEX_TYPE, VALUE_TYPE, SORT_SIZE, THREADS_PER_BLOCK> (row, M, segs, s_sort_j, s_sort_x, C_row_offsets, C_row_sizes, Cj, Cx);
            __syncthreads();
        }
    }
}

// template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, size_t THREADS_PER_BLOCK>
// __global__ void
// device_sort_kernel( int *temp_storage,
//                     size_t &temp_storage_bytes,
//                     INDEX_TYPE *key_buf1, 
//                     INDEX_TYPE *key_buf2,
//                     VALUE_TYPE *val_buf1,
//                     VALUE_TYPE *val_buf2,
//                     int *temp_vals,
//                     const int num_elements)
// {
//     cub::DoubleBuffer<INDEX_TYPE> dbuf_keys;
//     cub::DoubleBuffer<VALUE_TYPE> dbuf_vals;
//     dbuf_keys.d_buffers[0] = key_buf1;
//     dbuf_keys.d_buffers[1] = key_buf2;
//     dbuf_keys.selector = 0;
//     dbuf_vals.d_buffers[0] = val_buf1;
//     dbuf_vals.d_buffers[1] = val_buf2;
//     dbuf_vals.selector = 0;

//     size_t bytes = 0;
//     cub::DeviceRadixSort::SortPairs(NULL, bytes, dbuf_keys, dbuf_vals, num_elements);
//     if(bytes > temp_storage_bytes) {
//         SAFE_DELETE_ARRAY(temp_storage);
//         temp_storage = new int[bytes];
//         temp_storage_bytes = bytes;
//     }
//     cub::DeviceRadixSort::SortPairs((void *)temp_storage, temp_storage_bytes, dbuf_keys, dbuf_vals, num_elements);

//     if(blockIdx.x == 0 && threadIdx.x == 0) {
//         temp_vals[63] = dbuf_vals.selector;
//     }
// }

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, size_t THREADS_PER_BLOCK, size_t VALUES_PER_THREAD>
__global__ void
device_reduce_kernel(   int *temp_storage,
                        INDEX_TYPE  *key_buf,
                        VALUE_TYPE  *val_buf,
                        int         *threadCodes,
                        int         *blockCounts,
                        const int   row,
                        const int   num_rows,
                        const int   size,
                        INDEX_TYPE  *C_row_offsets,
                        INDEX_TYPE  *C_row_sizes)
{
    //const INDEX_TYPE thread_lane        = threadIdx.x;                                  // thread index within the vector (which is the block)
    //const INDEX_TYPE vector_id          = blockIdx.x;                                   // global vector index
    const INDEX_TYPE vector_lane        = threadIdx.x / WARP_SIZE;                      // vector index within the block
    const INDEX_TYPE warp_lane          = threadIdx.x & (WARP_SIZE-1);                  // lane within warp

    KernelReduceByKeyPreprocess_Global<INDEX_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD> (key_buf, size, threadCodes, blockCounts);

    if(blockIdx.x == 0 && vector_lane == 0) {
        int sum = 0;
        int limit = ROUND_UP(gridDim.x, WARP_SIZE);
        for(int i=warp_lane; i<limit; i+=WARP_SIZE) {
            int T = 0;
            if(i < gridDim.x)
                T = blockCounts[i];
            warpScanUp32(T, warp_lane);
            sum += T;
            if(i < gridDim.x)
                blockCounts[i] = sum;
            sum = __shfl(sum, WARP_SIZE-1);
        }

        if(warp_lane == WARP_SIZE-1) {
            C_row_sizes[row] = sum;
            INDEX_TYPE new_addr = atomicAdd(&C_row_sizes[num_rows], sum);  //increase global memory pointer

            INDEX_TYPE C_row_start = new_addr;
            INDEX_TYPE C_row_end = new_addr + sum;
            C_row_offsets[row*2] = C_row_start;
            C_row_offsets[row*2 + 1] = C_row_end;
        }
    }
}

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, size_t THREADS_PER_BLOCK, size_t VALUES_PER_THREAD>
__global__ void
device_reduceEmit_kernel(   int *temp_storage,
                            INDEX_TYPE *key_buf1, 
                            INDEX_TYPE *key_buf2,
                            VALUE_TYPE *val_buf1,
                            VALUE_TYPE *val_buf2,
                            int *threadCodes,
                            int *limits,
                            VALUE_TYPE *carryOut,
                            const int segs)
{
    cub::DoubleBuffer<INDEX_TYPE> dbuf_keys;
    cub::DoubleBuffer<VALUE_TYPE> dbuf_vals;
    dbuf_keys.d_buffers[0] = key_buf1;
    dbuf_keys.d_buffers[1] = key_buf2;
    dbuf_keys.selector = 0;
    dbuf_vals.d_buffers[0] = val_buf1;
    dbuf_vals.d_buffers[1] = val_buf2;
    dbuf_vals.selector = 0;

    KernelReduceByKeyEmit<INDEX_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD> (dbuf_keys.Current(), segs, threadCodes, limits, dbuf_keys.Alternate());
    KernelSegReduceApply_Global<VALUE_TYPE, THREADS_PER_BLOCK, VALUES_PER_THREAD> (threadCodes, segs, limits, dbuf_vals.Current(), dbuf_vals.Alternate(), carryOut);
}


//mega dcsr kernel for rows which cannot fit into shared memory (uses dynamic parallelism to assign kernels to each row)
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t BINS, size_t THREADS_PER_BLOCK, size_t VALUES_PER_THREAD, size_t SORT_SIZE>
__launch_bounds__(BLOCK_THREADS_MAX,1)
__global__ void
spmm_dcsr_mega_kernel(  const INDEX_TYPE M,
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
    //const INDEX_TYPE thread_id          = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; // global thread index
    const INDEX_TYPE thread_lane        = threadIdx.x;                                  // thread index within the vector (which is the block)
    const INDEX_TYPE vector_id          = blockIdx.x;                                   // global vector index
    //const INDEX_TYPE vector_lane        = threadIdx.x / WARP_SIZE;                    // vector index within the block
    const INDEX_TYPE num_vectors        = gridDim.x;                                    // total number of active vectors (1 vector per block for large kernel)

    __shared__ INDEX_TYPE *temp_idx_ptrs[4];
    __shared__ VALUE_TYPE *temp_val_ptrs[4];
    __shared__ int temp_vals[64];

    int memsize = 1;
    size_t temp_storage_bytes = 0;
    INDEX_TYPE *key_buf1 = NULL, *key_buf2 = NULL;
    VALUE_TYPE *val_buf1 = NULL, *val_buf2 = NULL;
    VALUE_TYPE *carryOut = NULL;
    int *threadCodes = NULL, *blockCounts = NULL, *temp_storage = NULL;
    cub::DoubleBuffer<INDEX_TYPE> dbuf_keys;
    cub::DoubleBuffer<VALUE_TYPE> dbuf_vals;

    INDEX_TYPE start_range, end_range;
    getRanges<INDEX_TYPE, SORT_SIZE> (start_range, end_range, MM_bins);
    __syncthreads();

    //Search row i of A matrix
    for(INDEX_TYPE iter=start_range + vector_id; iter < end_range; iter+=num_vectors)
    {
        INDEX_TYPE row = MM_rows[iter];
        INDEX_TYPE size = MM_sizes[iter];
        // if(thread_lane == 0)
        //     printf("row: %d size: %d iter: %d\n", row, size, iter);

        if(thread_lane == 0 && size > memsize) {
            SAFE_DELETE_ARRAY(key_buf1);
            SAFE_DELETE_ARRAY(key_buf2);
            SAFE_DELETE_ARRAY(val_buf1);
            SAFE_DELETE_ARRAY(val_buf2);

            while(memsize < size)
                memsize <<= 1;
            //printf("size: %d  memsize: %d  (%d)\n", size, memsize, blockIdx.x);

            key_buf1 = new INDEX_TYPE[memsize];
            key_buf2 = new INDEX_TYPE[memsize];
            val_buf1 = new VALUE_TYPE[memsize];
            val_buf2 = new VALUE_TYPE[memsize];
            temp_idx_ptrs[0] = key_buf1;
            temp_idx_ptrs[1] = key_buf2;
            temp_val_ptrs[0] = val_buf1;
            temp_val_ptrs[1] = val_buf2;

            int numBlocks = ROUND_UP(memsize, 1024);
            threadCodes = new int[memsize];
            blockCounts = new int[numBlocks];
            carryOut = new VALUE_TYPE[numBlocks];
            temp_idx_ptrs[2] = threadCodes;
            temp_idx_ptrs[3] = blockCounts;
            temp_val_ptrs[2] = carryOut;
        }
        __syncthreads();

        key_buf1 = temp_idx_ptrs[0];
        key_buf2 = temp_idx_ptrs[1];
        val_buf1 = temp_val_ptrs[0];
        val_buf2 = temp_val_ptrs[1];
        threadCodes = temp_idx_ptrs[2];
        blockCounts = temp_idx_ptrs[3];
        carryOut = temp_val_ptrs[2];
        dbuf_keys.d_buffers[0] = key_buf1;
        dbuf_keys.d_buffers[1] = key_buf2;
        dbuf_keys.selector = 0;
        dbuf_vals.d_buffers[0] = val_buf1;
        dbuf_vals.d_buffers[1] = val_buf2;
        dbuf_vals.selector = 0;

        //compute partial products for row i
        ComputePartials_block<INDEX_TYPE, VALUE_TYPE, BINS, THREADS_PER_BLOCK, SORT_SIZE>
                (row, pitchA, pitchB, A_row_offsets, A_row_sizes, Aj, Ax, 
                B_row_offsets, B_row_sizes, Bj, Bx, key_buf1, val_buf1, thread_lane);
        __syncthreads();

        //////////////////////////////////////
        // sort and reduce partial products

        //Sort elements
        if(thread_lane == 0) {
            // int numBlocks = (size+255) / 256;
            // size_t grid = numBlocks, blockSize = 256;
            // device_sort_kernel<INDEX_TYPE, VALUE_TYPE, BINS, 256> (temp_storage, temp_storage_bytes, key_buf1, key_buf2, val_buf1, val_buf2, size);
            size_t bytes = 0;
            cub::DeviceRadixSort::SortPairs(NULL, bytes, dbuf_keys, dbuf_vals, size);
            if(bytes > temp_storage_bytes) {
                SAFE_DELETE_ARRAY(temp_storage);
                temp_storage = new int[bytes/sizeof(int)+1];
                temp_storage_bytes = bytes;
            }
            cub::DeviceRadixSort::SortPairs((void *)temp_storage, temp_storage_bytes, dbuf_keys, dbuf_vals, size);
        }
        cudaDeviceSynchronize();

        //Reduce Elements
        if(thread_lane == 0) {
            int numBlocks = (size+1023) / 1024;
            size_t grid = numBlocks, blockSize = 256;
            device_reduce_kernel<INDEX_TYPE, VALUE_TYPE, BINS, 256, 4> <<<grid, blockSize>>> (temp_storage, dbuf_keys.Current(), dbuf_vals.Current(), 
                        threadCodes, blockCounts, row, M, size, C_row_offsets, C_row_sizes);
        }
        cudaDeviceSynchronize();

        //Write out results
        if(thread_lane == 0) {
            int numBlocks = (size+1023) / 1024;
            int segs = blockCounts[numBlocks-1];
            size_t grid = numBlocks, blockSize = 256;
            if(dbuf_keys.selector == 0)
                device_reduceEmit_kernel<INDEX_TYPE, VALUE_TYPE, BINS, 256, 4> <<<grid, blockSize>>> (temp_storage, key_buf1, key_buf2, val_buf1, val_buf2, 
                        threadCodes, blockCounts, carryOut, segs);
            else
                device_reduceEmit_kernel<INDEX_TYPE, VALUE_TYPE, BINS, 256, 4> <<<grid, blockSize>>> (temp_storage, key_buf2, key_buf1, val_buf2, val_buf1, 
                        threadCodes, blockCounts, carryOut, segs);
        }
        cudaDeviceSynchronize();

        // if(thread_lane == 0) {
        //     int segs = blockCounts[numBlocks-1];
        //     const int NT = 128;
        //     int numBlocks = ROUND_UP(segs, 128);
        //     size_t grid = numBlocks, blockSize = 256;
            
        //     KernelSegReduceSpine1<INDEX_TYPE, NT> <<<grid, blockSize>>> (temp_storage, key_buf1, key_buf2, val_buf1, val_buf2, threadCodes, blockCounts, segs);
        //     safeSync();

        //     if(numBlocks > 1) {
        //         KernelSegReduceSpine2<INDEX_TYPE, NT> <<<grid, blockSize>>> (temp_storage, key_buf2, key_buf1, val_buf2, val_buf1, threadCodes, blockCounts, segs);
        //     }
        // }
        // safeSync();
    }

    if(thread_lane == 0) {
        SAFE_DELETE_ARRAY(key_buf1);
        SAFE_DELETE_ARRAY(key_buf2);
        SAFE_DELETE_ARRAY(val_buf1);
        SAFE_DELETE_ARRAY(val_buf2);
        SAFE_DELETE_ARRAY(threadCodes);
        SAFE_DELETE_ARRAY(blockCounts);
        SAFE_DELETE_ARRAY(carryOut);
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