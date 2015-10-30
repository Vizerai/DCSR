#ifndef RADIX_SORT_INL
#define RADIX_SORT_INL

//store initial digit and 31 shuffles
template <INDEX_TYPE>
__device__ __forceinline__ getDigits(
		const INDEX_TYPE digit,
		INDEX_TYPE *bins)
{
	INDEX_TYPE next = digit;
	bins[next] += 1;
	next = __shfl_up(digit, 1);
	bins[next] += 1;
	next = __shfl_up(digit, 2);
	bins[next] += 1;
	next = __shfl_up(digit, 3);
	bins[next] += 1;
	next = __shfl_up(digit, 4);
	bins[next] += 1;
	next = __shfl_up(digit, 5);
	bins[next] += 1;
	next = __shfl_up(digit, 6);
	bins[next] += 1;
	next = __shfl_up(digit, 7);
	bins[next] += 1;

	next = __shfl_up(digit, 8);
	bins[next] += 1;
	next = __shfl_up(digit, 9);
	bins[next] += 1;
	next = __shfl_up(digit, 10);
	bins[next] += 1;
	next = __shfl_up(digit, 11);
	bins[next] += 1;
	next = __shfl_up(digit, 12);
	bins[next] += 1;
	next = __shfl_up(digit, 13);
	bins[next] += 1;
	next = __shfl_up(digit, 14);
	bins[next] += 1;
	next = __shfl_up(digit, 15);
	bins[next] += 1;

	next = __shfl_up(digit, 16);
	bins[next] += 1;
	next = __shfl_up(digit, 17);
	bins[next] += 1;
	next = __shfl_up(digit, 18);
	bins[next] += 1;
	next = __shfl_up(digit, 19);
	bins[next] += 1;
	next = __shfl_up(digit, 20);
	bins[next] += 1;
	next = __shfl_up(digit, 21);
	bins[next] += 1;
	next = __shfl_up(digit, 22);
	bins[next] += 1;
	next = __shfl_up(digit, 23);
	bins[next] += 1;

	next = __shfl_up(digit, 24);
	bins[next] += 1;
	next = __shfl_up(digit, 25);
	bins[next] += 1;
	next = __shfl_up(digit, 26);
	bins[next] += 1;
	next = __shfl_up(digit, 27);
	bins[next] += 1;
	next = __shfl_up(digit, 28);
	bins[next] += 1;
	next = __shfl_up(digit, 29);
	bins[next] += 1;
	next = __shfl_up(digit, 30);
	bins[next] += 1;
	next = __shfl_up(digit, 31);
	bins[next] += 1;
}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void radixSort32(INDEX_TYPE *keys, VALUE_TYPE *data, const int lane)
{
	unsigned char bins[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
	bool parity = 0;
	unsigned int cur_bit = 0;
	const unsigned int end_bit = 30;

	while(cur_bit < end_bit)
	{
		//rank keys
		char digit;
		INDEX_TYPE cur = key[lane];
		digit = BFE(cur, cur_bit, RADIX_BITS);
		getDigits(digit, bins);
		
		//get final offsets
		if(lane == WARP_SIZE-1)
		{
			for(int i=1; i<16; i++)
				bins[i] += bins[i-1];
		}

		bins[i] = __shfl(bins[i], WARP_SIZE - 1);

		//shuffle keys and values
		cur = key[lane];
		digit = BFE(cur, cur_bit, RADIX_BITS);
		getDigits(digit, bins);
		keys[bins[digit] + ((parity) ? 64 : 0)] = keys[i];
		data[bins[digit] + ((parity) ? 64 : 0)] = data[i];

		parity ^= 1;
		cur_bit += RADIX_BITS;
	}
}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void radixSort64(INDEX_TYPE *keys, VALUE_TYPE *data, const int lane)
{
	unsigned char bins[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
	bool parity = 0;
	unsigned int cur_bit = 0;
	const unsigned int end_bit = 30;

	while(cur_bit < end_bit)
	{
		//rank keys
		char digit;
		INDEX_TYPE cur = key[lane];
		digit = BFE(cur, cur_bit, RADIX_BITS);
		bins[digit]++;

		cur = key[lane+32];
		getDigits(digit, bins);

		//get final offsets
		if(lane == WARP_SIZE-1)
		{
			for(int i=1; i<16; i++)
				bins[i] += bins[i-1];
		}

		bins[i] = __shfl(bins[i], WARP_SIZE - 1);

		//shuffle keys and values
		cur = key[lane];
		digit = BFE(cur, cur_bit, RADIX_BITS);
		getDigits(digit, bins);
		keys[bins[digit] + ((parity) ? 64 : 0)] = keys[i];
		data[bins[digit] + ((parity) ? 64 : 0)] = data[i];

		cur = key[lane+32];
		digit = BFE(cur, cur_bit, RADIX_BITS);
		getDigits(digit, bins);
		keys[bins[digit] + ((parity) ? 64 : 0)] = keys[i];
		data[bins[digit] + ((parity) ? 64 : 0)] = data[i];

		parity ^= 1;
		cur_bit += RADIX_BITS;
	}
}

// template<typename INDEX_TYPE, typename VALUE_TYPE>
// __device__ __forceinline__ void radixSort(INDEX_TYPE *key, VALUE_TYPE *data, const int lane, const int size)
// {
// 	unsigned char bins[16];

// 	while(cur_bit < end_bit)
// 	{
// 		for(int i=lane; i<size; i+=WARP_SIZE)
// 		{
// 			char digit;
// 			INDEX_TYPE cur = key[i];
			


// 		}


// 		for(int i=lane; i<size; i+=WARP_SIZE)
// 		{

// 		}

// 		bits += RADIX_BITS;
// 	}
// }

#endif