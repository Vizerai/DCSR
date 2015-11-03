#ifndef RADIX_SORT_INL
#define RADIX_SORT_INL

#define RADIX_BITS 		3			// number of bits sorted per pass
#define RADIX_BINS 		8 			// 2^RADIX_BITS

__device__ __forceinline__ void resetBins(
		volatile unsigned short *bins)
{
	#pragma unroll
	for(int i=0; i<RADIX_BINS; ++i)
	{
		bins[RADIX_BINS] = 0;
	}
}

/**
 * Bitfield-extract.
 */
template <typename UnsignedBits>
__device__ __forceinline__ unsigned int BFE(
	    UnsignedBits            source,
	    unsigned int            bit_start,
	    unsigned int            num_bits)
{
    unsigned int bits;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
}

//store initial digit and 31 shuffles
template <typename INDEX_TYPE, unsigned int BMASK>
__device__ __forceinline__ void shuflDigits(
		const INDEX_TYPE digit,
		INDEX_TYPE &mask)
{
	// Use predicate set from SHFL to guard against invalid peers
	mask = ((digit == BMASK) ? 1 : 0);
	mask += __shfl_up(mask, 16);
	mask += __shfl_up(mask, 8);
	mask += __shfl_up(mask, 4);
	mask += __shfl_up(mask, 2);
	mask += __shfl_up(mask, 1);

	//adjust for over addition
	//printf("lane: %d  bins[%d]: %d\n", bins[digit]);
	//bins[digit] -= (WARP_SIZE - (lane + 1));
	//printf("lane: %d  bins[%d]: %d\n", lane, digit, bins[digit]);
}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void radixSort32(volatile INDEX_TYPE *keys, volatile VALUE_TYPE *data, const int lane)
{
	short parity = 0;
	//const unsigned int end_bit = 8;
	unsigned int digit;
	unsigned int bin01, bin23, bin45, bin67;
	//volatile unsigned short bins[RADIX_BINS];// = {0,0,0,0}; // 0,0,0,0, 0,0,0,0, 0,0,0,0};

	int cur_bit = 0;
	#pragma unroll
	for(cur_bit=0; cur_bit < 30; cur_bit+=RADIX_BITS)
	{
		//resetBins(bins);
		bin01 = 0;
		bin23 = 0;
		bin45 = 0;
		bin67 = 0;

		//rank keys	
		INDEX_TYPE cur_key = keys[lane + parity];
		digit = BFE(cur_key, cur_bit, RADIX_BITS);

		unsigned int mask;
		mask = ((digit == 0x0) ? 1 : 0);
		mask = __ballot(mask);
		bin01 += __popc(mask);

		mask = ((digit == 0x1) ? 1 : 0);
		mask = __ballot(mask);
		bin01 += (__popc(mask) << 16);

		mask = ((digit == 0x2) ? 1 : 0);
		mask = __ballot(mask);
		bin23 += __popc(mask);

		mask = ((digit == 0x3) ? 1 : 0);
		mask = __ballot(mask);
		bin23 += (__popc(mask) << 16);

		mask = ((digit == 0x4) ? 1 : 0);
		mask = __ballot(mask);
		bin45 += __popc(mask);

		mask = ((digit == 0x5) ? 1 : 0);
		mask = __ballot(mask);
		bin45 += (__popc(mask) << 16);

		mask = ((digit == 0x6) ? 1 : 0);
		mask = __ballot(mask);
		bin67 += __popc(mask);

		//bin 7 is implicit
		// mask = ((digit == 0x7) ? 1 : 0);
		// mask = __ballot(mask);
		// bin67 += (__popc(mask) << 16);

		//get offset totals
		{
			short sum = 0, val1, val2;

			val1 = (bin01 & 0x0000FFFF);
			val2 = ((bin01 & 0xFFFF0000) >> 16);
			bin01 = (((sum+val1) << 16) | sum);
			sum += (val1 + val2);

			val1 = (bin23 & 0x0000FFFF);
			val2 = ((bin23 & 0xFFFF0000) >> 16);
			bin23 = (((sum+val1) << 16) | sum);
			sum += (val1 + val2);

			val1 = (bin45 & 0x0000FFFF);
			val2 = ((bin45 & 0xFFFF0000) >> 16);
			bin45 = (((sum+val1) << 16) | sum);
			sum += (val1 + val2);

			val1 = (bin67 & 0x0000FFFF);
			//val2 = (bin67 & 0xFFFF0000);		//last one not needed
			bin67 = (((sum+val1) << 16) | sum);
			//sum += (val1 + val2);
		}

		//shuffle keys and values
		//cur_key = keys[lane + parity];
		//digit = BFE(cur_key, cur_bit, RADIX_BITS);

		mask = ((digit == 0x0) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin01 += __popc(mask);

		mask = ((digit == 0x1) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin01 += (__popc(mask) << 16);

		mask = ((digit == 0x2) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin23 += __popc(mask);

		mask = ((digit == 0x3) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin23 += (__popc(mask) << 16);

		mask = ((digit == 0x4) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin45 += __popc(mask);

		mask = ((digit == 0x5) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin45 += (__popc(mask) << 16);

		mask = ((digit == 0x6) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin67 += __popc(mask);

		mask = ((digit == 0x7) ? 1 : 0);
		mask = __ballot(mask);
		mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
		bin67 += (__popc(mask) << 16);

		int offset = ((digit == 0x0) ? (bin01 & 0x0000FFFF) : 0);
		offset += ((digit == 0x1) ? (bin01 >> 16) : 0);
		offset += ((digit == 0x2) ? (bin23 & 0x0000FFFF) : 0);
		offset += ((digit == 0x3) ? (bin23 >> 16) : 0);
		offset += ((digit == 0x4) ? (bin45 & 0x0000FFFF) : 0);
		offset += ((digit == 0x5) ? (bin45 >> 16) : 0);
		offset += ((digit == 0x6) ? (bin67 & 0x0000FFFF) : 0);
		offset += ((digit == 0x7) ? (bin67 >> 16) : 0);
		//early exit condition
		if(__all(offset-1 == (lane+parity)))
		{
			if(parity)
			{
				//swap values back....
				offset += (parity^32)-1;
				keys[offset] = cur_key;
				data[offset] = data[lane + parity];
			}
			
			break;
		}
		offset += (parity^32)-1;

		keys[offset] = cur_key;
		data[offset] = data[lane + parity];

		parity ^= 32;
	}
}

// template<typename INDEX_TYPE, typename VALUE_TYPE>
// __device__ __forceinline__ void radixSort64(INDEX_TYPE *keys, VALUE_TYPE *data, const int lane)
// {
// 	int parity = 0;
// 	unsigned int cur_bit = 0;
// 	const unsigned int end_bit = 8;

// 	while(cur_bit < end_bit)
// 	{
// 		//rank keys
// 		unsigned int digit;
// 		unsigned int bins[RADIX_BINS] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

// 		INDEX_TYPE cur_key = keys[lane + parity];
// 		digit = BFE(cur_key, cur_bit, RADIX_BITS);
// 		bins[digit]++;
// 		shuflDigits(digit, bins, lane);

// 		cur_key = keys[lane + parity + 32];
// 		digit = BFE(cur_key, cur_bit, RADIX_BITS);
// 		bins[digit]++;
// 		shuflDigits(digit, bins, lane);

// 		if(lane == WARP_SIZE-1)
// 		{
// 			for(int i=0; i<RADIX_BINS; i++)
// 				printf("1 bins[%d]: %d\n", i, bins[i]);
// 		}

// 		//get final offsets
// 		if(lane == WARP_SIZE-1)
// 		{	
// 			unsigned int sum = 0;
// 			#pragma unroll
// 			for(int i=0; i<RADIX_BINS; i++)
// 			{
// 				unsigned int val = bins[i];
// 				bins[i] = sum;
// 				sum += val;
// 				printf("2 bins[%d]: %d\n", i, bins[i]);
// 			}
// 		}

// 		#pragma unroll
// 		for(int i=0; i<RADIX_BINS; i++)
// 			bins[i] = __shfl(bins[i], WARP_SIZE - 1);

// 		//shuffle keys and values
// 		cur_key = keys[lane + parity];
// 		digit = BFE(cur_key, cur_bit, RADIX_BITS);
// 		shuflDigits(digit, bins, lane);
// 		keys[bins[digit] + (parity^64)] = cur_key;
// 		data[bins[digit] + (parity^64)] = data[lane + parity];

// 		cur_key = keys[lane + parity + 32];
// 		digit = BFE(cur_key, cur_bit, RADIX_BITS);
// 		shuflDigits(digit, bins, lane);
// 		keys[bins[digit] + (parity^64)] = cur_key;
// 		data[bins[digit] + (parity^64)] = data[lane + parity + 32];

// 		parity ^= 64;
// 		cur_bit += RADIX_BITS;
// 	}
// }

// template<typename INDEX_TYPE, typename VALUE_TYPE>
// __device__ __forceinline__ void radixSort(INDEX_TYPE *key, VALUE_TYPE *data, const int lane, const int size)
// {
// 	unsigned int bins[16];

// 	while(cur_bit < end_bit)
// 	{
// 		for(int i=lane; i<size; i+=WARP_SIZE)
// 		{
// 			int digit;
// 			INDEX_TYPE cur = key[i];
			


// 		}


// 		for(int i=lane; i<size; i+=WARP_SIZE)
// 		{

// 		}

// 		bits += RADIX_BITS;
// 	}
// }

#endif