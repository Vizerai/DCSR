#ifndef SORT_INL
#define SORT_INL

/******************************************************************************
* Templated iteration
******************************************************************************/

// General template iteration
template <int SIZE, int MAX>
struct Iterate
{

	template<typename INDEX_TYPE, typename VALUE_TYPE>
	static __device__ __forceinline__ void bitonicSortKey(volatile INDEX_TYPE *key, volatile VALUE_TYPE *data, const int tID)
	{
		INDEX_TYPE kvar;
		VALUE_TYPE dvar;
		int i,j;

		if(tID == 0)
			printf("SIZE: %d  MAX: %d\n", SIZE, MAX);

		if(tID == 0)
			printf("BIR: %d, %d\n", SIZE, tID);
		BITONIC_INDEX_REVERSE(i,j,SIZE,tID);
		if(key[i] > key[j])
		{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
		
		#pragma unroll
		for(int offset=32; offset < MAX>>1; offset+=32)
		{
			if(tID == 0)
				printf("BIR: %d, %d\n", SIZE, tID+offset);
			BITONIC_INDEX_REVERSE(i,j,SIZE,tID+offset);
			if(key[i] > key[j])
			{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
		}

		#pragma unroll
		for(int k=SIZE>>1; k>1; k>>=1)
		{
			if(tID == 0)
				printf("BI: %d, %d\n", k, tID);
			BITONIC_INDEX(i,j,k,tID);
			if(key[i] > key[j])
			{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

			#pragma unroll
			for(int offset=32; offset < MAX>>1; offset+=32)
			{
				if(tID == 0)
					printf("BI: %d, %d\n", k, tID+offset);
				BITONIC_INDEX(i,j,k,tID+offset);
				if(key[i] > key[j])
				{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
			}
		}

		Iterate<SIZE<<1, MAX>::bitonicSortKey(key, data, tID);
	}

};

// Termination
template <int MAX>
struct Iterate<MAX, MAX>
{

	//termination
	template<typename INDEX_TYPE, typename VALUE_TYPE>
	static __device__ __forceinline__ void bitonicSortKey(volatile INDEX_TYPE *key, volatile VALUE_TYPE *data, const int tID)
	{
		INDEX_TYPE kvar;
		VALUE_TYPE dvar;
		int i,j;

		if(tID == 0)
			printf("MAX: %d\n", MAX);

		if(tID == 0)
			printf("BIR: %d, %d\n", MAX, tID);
		BITONIC_INDEX_REVERSE(i,j,MAX,tID);
		if(key[i] > key[j])
		{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

		#pragma unroll
		for(int offset=32; offset < MAX>>1; offset+=32)
		{
			if(tID == 0)
				printf("BIR: %d, %d\n", MAX, tID);
			BITONIC_INDEX_REVERSE(i,j,MAX,tID+offset);
			if(key[i] > key[j])
			{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
		}

		#pragma unroll
		for(int k=MAX>>1; k>1; k>>=1)
		{
			if(tID == 0)
				printf("BI: %d, %d\n", k, tID);
			BITONIC_INDEX(i,j,k,tID);
			if(key[i] > key[j])
			{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

			#pragma unroll
			for(int offset=32; offset < MAX>>1; offset+=32)
			{
				if(tID == 0)
					printf("BI: %d, %d\n", k, tID+offset);
				BITONIC_INDEX(i,j,k,tID+offset);
				if(key[i] > key[j])
				{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
			}
		}
	}

};

template<typename T>
__device__ __forceinline__ void bitonicSort32(T *data, const int tID)
{
	if(tID < 16)
	{
		T dvar;
		int i,j;

		BITONIC_INDEX(i,j,2,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		
		BITONIC_INDEX_REVERSE(i,j,4,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,2,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }

		BITONIC_INDEX_REVERSE(i,j,8,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,4,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,2,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }

		BITONIC_INDEX_REVERSE(i,j,16,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,8,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,4,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,2,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }

		BITONIC_INDEX_REVERSE(i,j,32,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,16,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,8,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,4,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
		BITONIC_INDEX(i,j,2,tID);
		if(data[i] > data[j])
		{	SWAP(data[i], data[j], dvar); }
	}
}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void bitonicSort32_Key(volatile INDEX_TYPE *key, volatile VALUE_TYPE *data, const int tID)//, INDEX_TYPE kvar, VALUE_TYPE dvar)
{
	if(tID < 16)
	{
		INDEX_TYPE kvar;
		VALUE_TYPE dvar;
		int i,j;

		BITONIC_INDEX(i,j,2,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		
		BITONIC_INDEX_REVERSE(i,j,4,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,2,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}

		BITONIC_INDEX_REVERSE(i,j,8,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,4,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,2,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}

		BITONIC_INDEX_REVERSE(i,j,16,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,8,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,4,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,2,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}

		BITONIC_INDEX_REVERSE(i,j,32,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,16,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,8,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,4,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
		BITONIC_INDEX(i,j,2,tID);
		if(key[i] > key[j])
		{	
			SWAP(key[i], key[j], kvar); 
			SWAP(data[i], data[j], dvar);
		}
	}
}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void bitonicSort64_Key(volatile INDEX_TYPE *key, volatile VALUE_TYPE *data, const int tID)
{

	INDEX_TYPE kvar;
	VALUE_TYPE dvar;
	int i,j;

	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void bitonicSort128_Key(volatile INDEX_TYPE *key, volatile VALUE_TYPE *data, const int tID)
{

	INDEX_TYPE kvar;
	VALUE_TYPE dvar;
	int i,j;

	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,128,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

}

#endif