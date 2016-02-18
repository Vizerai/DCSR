#ifndef RADIX_SORT_INL
#define RADIX_SORT_INL

#define S_RADIX_BITS 	4					// number of bits sorted per pass
#define S_RADIX_BINS 	16 				// 2^RADIX_BITS
#define WIDTH 				32					// bin width
#define HALF_WIDTH		(WIDTH/2)		// half of bin width in bits
#define END_BIT 			30					//sort until 2^30

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

//ballot and popc based methods for determining number of digits that equal BMASK
template <unsigned int BMASK>
__device__ __forceinline__ int voteCount(
		const unsigned int digit,
		unsigned int &mask)
{
	mask = ((digit == BMASK) ? 1 : 0);
	mask = __ballot(mask);
	return __popc(mask);
}

template <unsigned int BMASK>
__device__ __forceinline__ int voteCount(
		const unsigned int digit,
		unsigned int &mask,
		const int lane)
{
	mask = ((digit == BMASK) ? 1 : 0);
	mask = __ballot(mask);
	mask = (mask & (0xFFFFFFFF >> (WARP_SIZE - lane - 1)));
	return __popc(mask);
}

template<typename INDEX_TYPE, typename VALUE_TYPE, int NUM_COUNTERS>
__device__ __forceinline__ void radixSort32(
			volatile INDEX_TYPE *keys, 
			volatile VALUE_TYPE *data,
			const int lane)
{
	short parity = 0;
	//const unsigned int end_bit = 8;
	unsigned int digit;
	//unsigned int bin01, bin23, bin45, bin67;
	unsigned int bins[NUM_COUNTERS];

	#pragma unroll
	for(int cur_bit=0; cur_bit < END_BIT; cur_bit+=S_RADIX_BITS)
	{
		// bin01 = 0;
		// bin23 = 0;
		// bin45 = 0;
		// bin67 = 0;
		#pragma unroll
		for(int i=0; i<NUM_COUNTERS; ++i) {
			bins[i] = 0;
		}

		//rank keys	
		INDEX_TYPE cur_key = keys[lane + parity];
		digit = BFE(cur_key, cur_bit, S_RADIX_BITS);

		unsigned int mask;
		bins[digit]++;
		// bin01 +=  voteCount<0x0>(digit, mask, lane);
		// bin01 += (voteCount<0x1>(digit, mask, lane) << HALF_WIDTH);
		// bin23 +=  voteCount<0x2>(digit, mask, lane);
		// bin23 += (voteCount<0x3>(digit, mask, lane) << HALF_WIDTH);
		// bin45 +=  voteCount<0x4>(digit, mask, lane);
		// bin45 += (voteCount<0x5>(digit, mask, lane) << HALF_WIDTH);
		// bin67 +=  voteCount<0x6>(digit, mask, lane);
		// bin67 += (voteCount<0x7>(digit, mask, lane) << HALF_WIDTH);

		//get offset totals
		{
			// unsigned int val1;
			// unsigned short sum = 0;//, val1, val2;

			// val1 = __shfl(bin01, WARP_SIZE-1);
			// sum = (val1 & 0x0000FFFF);
			// bin01 += (sum << HALF_WIDTH);
			// sum += ((val1 & 0xFFFF0000) >> HALF_WIDTH);

			// val1 = __shfl(bin23, WARP_SIZE-1);
			// bin23 += (sum);
			// sum += (val1 & 0x0000FFFF);
			// bin23 += (sum << HALF_WIDTH);
			// sum += ((val1 & 0xFFFF0000) >> HALF_WIDTH);

			// val1 = __shfl(bin45, WARP_SIZE-1);
			// bin45 += (sum);
			// sum += (val1 & 0x0000FFFF);
			// bin45 += (sum << HALF_WIDTH);
			// sum += ((val1 & 0xFFFF0000) >> HALF_WIDTH);

			// val1 = __shfl(bin67, WARP_SIZE-1);
			// bin67 += (sum);
			// sum += (val1 & 0x0000FFFF);
			// bin67 += (sum << HALF_WIDTH);
			// //sum += ((val1 & 0xFFFF0000) >> HALF_WIDTH);
		}

		//shuffle keys and values
		int offset = bins[(digit >> 1)] & 0x0000FFFF;
		// int offset = ((digit == 0x0) ? (bin01 & 0x0000FFFF) : 0);
		// offset += ((digit == 0x1) ? (bin01 >> HALF_WIDTH) : 0);
		// offset += ((digit == 0x2) ? (bin23 & 0x0000FFFF) : 0);
		// offset += ((digit == 0x3) ? (bin23 >> HALF_WIDTH) : 0);
		// offset += ((digit == 0x4) ? (bin45 & 0x0000FFFF) : 0);
		// offset += ((digit == 0x5) ? (bin45 >> HALF_WIDTH) : 0);
		// offset += ((digit == 0x6) ? (bin67 & 0x0000FFFF) : 0);
		// offset += ((digit == 0x7) ? (bin67 >> HALF_WIDTH) : 0);

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

template<typename INDEX_TYPE, typename VALUE_TYPE, int SIZE>
__device__ __forceinline__ void radixSort(volatile INDEX_TYPE *keys, volatile VALUE_TYPE *data, const int lane)
{
	short parity = 0;
	//const unsigned int end_bit = 8;
	unsigned int digit;
	unsigned int bin01, bin23, bin45, bin67, bin89, binAB, binCD, binEF;		//8 (2 part) bins for 16 total counters

	#pragma unroll
	for(int cur_bit=0; cur_bit < 30; cur_bit+=S_RADIX_BITS)
	{
		unsigned int early_exit = 0;
		//set bin values to 0 at beginning of pass
		bin01 = 0;
		bin23 = 0;
		bin45 = 0;
		bin67 = 0;
		bin89 = 0;
		binAB = 0;
		binCD = 0;
		binEF = 0;

		//rank keys
		#pragma unroll
		for(int shift=0; shift < SIZE; shift+=WARP_SIZE)
		{
			INDEX_TYPE cur_key = keys[lane + parity + shift];
			digit = BFE(cur_key, cur_bit, S_RADIX_BITS);

			unsigned int mask;
			bin01 +=  voteCount<0x0>(digit, mask);
			bin01 += (voteCount<0x1>(digit, mask) << HALF_WIDTH);
			bin23 +=  voteCount<0x2>(digit, mask);
			bin23 += (voteCount<0x3>(digit, mask) << HALF_WIDTH);
			bin45 +=  voteCount<0x4>(digit, mask);
			bin45 += (voteCount<0x5>(digit, mask) << HALF_WIDTH);
			bin67 +=  voteCount<0x6>(digit, mask);
			bin67 += (voteCount<0x7>(digit, mask) << HALF_WIDTH);
			bin89 +=  voteCount<0x8>(digit, mask);
			bin89 += (voteCount<0x9>(digit, mask) << HALF_WIDTH);
			binAB +=  voteCount<0xA>(digit, mask);
			binAB += (voteCount<0xB>(digit, mask) << HALF_WIDTH);
			binCD +=  voteCount<0xC>(digit, mask);
			binCD += (voteCount<0xD>(digit, mask) << HALF_WIDTH);
			binEF +=  voteCount<0xE>(digit, mask);
			//binEF += (voteCount<0xF>(digit, mask) << HALF_WIDTH);		//bin F is implicit
		}

		//get final offsets
		{
			unsigned short sum = 0, val1, val2;

			val1 = (bin01 & 0x0000FFFF);
			val2 = ((bin01 & 0xFFFF0000) >> HALF_WIDTH);
			bin01 = (((sum+val1) << HALF_WIDTH) | sum);		//bin 0 offset is set to 0 since it is at the beginning
			sum += (val1 + val2);

			val1 = (bin23 & 0x0000FFFF);
			val2 = ((bin23 & 0xFFFF0000) >> HALF_WIDTH);
			bin23 = (((sum+val1) << HALF_WIDTH) | sum);
			sum += (val1 + val2);

			val1 = (bin45 & 0x0000FFFF);
			val2 = ((bin45 & 0xFFFF0000) >> HALF_WIDTH);
			bin45 = (((sum+val1) << HALF_WIDTH) | sum);
			sum += (val1 + val2);

			val1 = (bin67 & 0x0000FFFF);
			val2 = (bin67 & 0xFFFF0000);
			bin67 = (((sum+val1) << HALF_WIDTH) | sum);
			sum += (val1 + val2);

			val1 = (bin89 & 0x0000FFFF);
			val2 = ((bin89 & 0xFFFF0000) >> HALF_WIDTH);
			bin89 = (((sum+val1) << HALF_WIDTH) | sum);
			sum += (val1 + val2);

			val1 = (binAB & 0x0000FFFF);
			val2 = ((binAB & 0xFFFF0000) >> HALF_WIDTH);
			binAB = (((sum+val1) << HALF_WIDTH) | sum);
			sum += (val1 + val2);

			val1 = (binCD & 0x0000FFFF);
			val2 = ((binCD & 0xFFFF0000) >> HALF_WIDTH);
			binCD = (((sum+val1) << HALF_WIDTH) | sum);
			sum += (val1 + val2);

			val1 = (binEF & 0x0000FFFF);
			//val2 = (binEF & 0xFFFF0000);		//last one not needed
			binEF = (((sum+val1) << HALF_WIDTH) | sum);
			//sum += (val1 + val2);
		}

		//shuffle keys and values
		#pragma unroll
		for(int shift=0; shift < SIZE; shift+=WARP_SIZE)
		{
			INDEX_TYPE cur_key = keys[lane + parity + shift];
			digit = BFE(cur_key, cur_bit, S_RADIX_BITS);

			unsigned int mask;
			bin01 +=  voteCount<0x0>(digit, mask, lane);
			bin01 += (voteCount<0x1>(digit, mask, lane) << HALF_WIDTH);
			bin23 +=  voteCount<0x2>(digit, mask, lane);
			bin23 += (voteCount<0x3>(digit, mask, lane) << HALF_WIDTH);
			bin45 +=  voteCount<0x4>(digit, mask, lane);
			bin45 += (voteCount<0x5>(digit, mask, lane) << HALF_WIDTH);
			bin67 +=  voteCount<0x6>(digit, mask, lane);
			bin67 += (voteCount<0x7>(digit, mask, lane) << HALF_WIDTH);
			bin89 +=  voteCount<0x8>(digit, mask, lane);
			bin89 += (voteCount<0x9>(digit, mask, lane) << HALF_WIDTH);
			binAB +=  voteCount<0xA>(digit, mask, lane);
			binAB += (voteCount<0xB>(digit, mask, lane) << HALF_WIDTH);
			binCD +=  voteCount<0xC>(digit, mask, lane);
			binCD += (voteCount<0xD>(digit, mask, lane) << HALF_WIDTH);
			binEF +=  voteCount<0xE>(digit, mask, lane);
			binEF += (voteCount<0xF>(digit, mask, lane) << HALF_WIDTH);

			//calculate offsets into key and data arrays
			int offset = ((digit == 0x0) ? (bin01 & 0x0000FFFF) : 0);
			offset += ((digit == 0x1) ? (bin01 >> HALF_WIDTH) : 0);
			offset += ((digit == 0x2) ? (bin23 & 0x0000FFFF) : 0);
			offset += ((digit == 0x3) ? (bin23 >> HALF_WIDTH) : 0);
			offset += ((digit == 0x4) ? (bin45 & 0x0000FFFF) : 0);
			offset += ((digit == 0x5) ? (bin45 >> HALF_WIDTH) : 0);
			offset += ((digit == 0x6) ? (bin67 & 0x0000FFFF) : 0);
			offset += ((digit == 0x7) ? (bin67 >> HALF_WIDTH) : 0);
			offset += ((digit == 0x8) ? (bin89 & 0x0000FFFF) : 0);
			offset += ((digit == 0x9) ? (bin89 >> HALF_WIDTH) : 0);
			offset += ((digit == 0xA) ? (binAB & 0x0000FFFF) : 0);
			offset += ((digit == 0xB) ? (binAB >> HALF_WIDTH) : 0);
			offset += ((digit == 0xC) ? (binCD & 0x0000FFFF) : 0);
			offset += ((digit == 0xD) ? (binCD >> HALF_WIDTH) : 0);
			offset += ((digit == 0xE) ? (binEF & 0x0000FFFF) : 0);
			offset += ((digit == 0xF) ? (binEF >> HALF_WIDTH) : 0);
			offset += (parity^SIZE)-1;

			//early exit condition
			if(__all(offset-(parity^SIZE) == (lane+parity+shift)))
				early_exit |= (1 << (shift/WARP_SIZE));

			keys[offset] = cur_key;
			data[offset] = data[lane + parity + shift];

			bin01 = __shfl(bin01, WARP_SIZE-1);
			bin23 = __shfl(bin23, WARP_SIZE-1);
			bin45 = __shfl(bin45, WARP_SIZE-1);
			bin67 = __shfl(bin67, WARP_SIZE-1);
			bin89 = __shfl(bin89, WARP_SIZE-1);
			binAB = __shfl(binAB, WARP_SIZE-1);
			binCD = __shfl(binCD, WARP_SIZE-1);
			binEF = __shfl(binEF, WARP_SIZE-1);
		}

		parity ^= SIZE;

		if(early_exit == (0xFFFFFFFF >> (WARP_SIZE-SIZE/WARP_SIZE)))
			break;
	}

	//if ended on odd parity, move values back to original buffer
	if(parity)
	{
		#pragma unroll
		for(int shift=lane; shift < SIZE; shift+=WARP_SIZE)
		{
			keys[shift] = keys[shift + SIZE];
			data[shift] = data[shift + SIZE];
		}
	}
}

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

// 		bits += S_RADIX_BITS;
// 	}
// }

#endif