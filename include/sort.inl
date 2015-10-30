#ifndef SORT_INL
#define SORT_INL

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
__device__ __forceinline__ void bitonicSort32_Key(INDEX_TYPE *key, VALUE_TYPE *data, const int tID)
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

template<typename T>
__device__ __forceinline__ void bitonicSort64(T *data, const int tID)
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

	BITONIC_INDEX_REVERSE(i,j,64,tID);
	if(data[i] > data[j])
	{	SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
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

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void bitonicSort64_Key(INDEX_TYPE *key, VALUE_TYPE *data, const int tID)
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
__device__ __forceinline__ void bitonicSort128_Key(INDEX_TYPE *key, VALUE_TYPE *data, const int tID)
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

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void bitonicSort256_Key(INDEX_TYPE *key, VALUE_TYPE *data, const int tID)
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
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,128,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,256,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void bitonicSort512_Key(INDEX_TYPE *key, VALUE_TYPE *data, const int tID)
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
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,128,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,256,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,512,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{       SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__device__ __forceinline__ void bitonicSort1024_Key(INDEX_TYPE *key, VALUE_TYPE *data, const int tID)
{
	INDEX_TYPE kvar;
	VALUE_TYPE dvar;
	int i,j;

	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,16,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,16,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,32,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,32,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,64,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,64,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,128,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,128,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,256,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,256,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,512,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,512,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

	BITONIC_INDEX_REVERSE(i,j,1024,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX_REVERSE(i,j,1024,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,512,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,256,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,128,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,64,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,32,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,16,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,8,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,4,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,tID);
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+32));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+64));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+96));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+128));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+160));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+192));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+224));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+256));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+288));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+320));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+352));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+384));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+416));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+448));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }
	BITONIC_INDEX(i,j,2,(tID+480));
	if(key[i] > key[j])
	{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }

}

#endif