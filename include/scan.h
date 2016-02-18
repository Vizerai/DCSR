#ifndef SCAN_H
#define SCAN_H

#define NUM_BANKS 		32
#define LOG_NUM_BANKS	5
#define CONFLICT_FREE_OFFSET(n)		(n >> LOG_NUM_BANKS)

#define BITONIC_INDEX_REVERSE(i,j,gs,tID)		i = (tID/(gs>>1))*gs + (tID & ((gs>>1)-1));  j = (tID/(gs>>1))*gs + (gs - (tID & ((gs>>1)-1)) - 1)
#define BIR_I											(tID/(gs>>1))*gs + (tID & ((gs>>1)-1))
#define BIR_J											(tID/(gs>>1))*gs + (gs - (tID & ((gs>>1)-1)) - 1)
#define BITONIC_INDEX(i,j,gs,tID)				i = (tID/(gs>>1))*gs + (tID & ((gs>>1)-1));  j = (gs>>1) + i
#define BI_I 											(tID/(gs>>1))*gs + (tID & ((gs>>1)-1))
#define BI_J 											(gs>>1) + (tID/(gs>>1))*gs + (tID & ((gs>>1)-1))
#define SWAP(X,Y,T)								T = X; X = Y; Y = T

//#include <cub/cub/ptx_util.cuh>

__device__ __forceinline__ int shfl_up_add(int x, int offset, int width = WARP_SIZE)
{
	int result = 0;
	int mask = (WARP_SIZE - width) << 8;
	asm(
		"{.reg .s32 r0;"
		".reg .pred p;"
		"shfl.up.b32 r0|p, %1, %2, %3;"
		"@p add.s32 r0, r0, %4;"
		"mov.s32 %0, r0; }"
		: "=r"(result) : "r"(x), "r"(offset), "r"(mask), "r"(x));
	return result;
}

__device__ __forceinline__ int shfl_down_add(int x, int offset, int width = WARP_SIZE)
{
	int result = 0;
	int mask = (WARP_SIZE - width) << 8;
	asm(
		"{.reg .s32 r0;"
		".reg .pred p;"
		"shfl.down.b32 r0|p, %1, %2, %3;"
		"@p add.s32 r0, r0, %4;"
		"mov.s32 %0, r0; }"
		: "=r"(result) : "r"(x), "r"(offset), "r"(mask), "r"(x));
	return result;
}

__device__ __forceinline__ double shfl_down(double val, unsigned int delta, int width = WARP_SIZE)
{
	int lo = __double2loint(val);
	int hi = __double2hiint(val);

	lo = __shfl_down(lo, delta, width);
	hi = __shfl_down(hi, delta, width);

	return __hiloint2double(hi, lo);
}

template<typename T>
__device__ __forceinline__ void warpScanUp32(T &var, const int lane)
{
	var = shfl_up_add(var, 16);
	var = shfl_up_add(var, 8);
	var = shfl_up_add(var, 4);
	var = shfl_up_add(var, 2);
	var = shfl_up_add(var, 1);
}

//only the last lane will have the correct result
template<typename T>
__device__ __forceinline__ void warpScanUp32_last(T &var)
{
	var += __shfl_up(var, 16);
	var += __shfl_up(var, 8);
	var += __shfl_up(var, 4);
	var += __shfl_up(var, 2);
	var += __shfl_up(var, 1);
}

template<typename T>
__device__ __forceinline__ void warpScanDown32(T &var)
{
	var = shfl_down_add(var, 16);
	var = shfl_down_add(var, 8);
	var = shfl_down_add(var, 4);
	var = shfl_down_add(var, 2);
	var = shfl_down_add(var, 1);
}

template<>
__device__ __forceinline__ void warpScanDown32(double &var)
{
	var += shfl_down(var, 16);
	var += shfl_down(var, 8);
	var += shfl_down(var, 4);
	var += shfl_down(var, 2);
	var += shfl_down(var, 1);
}

template<typename T>
__device__ __forceinline__ void prescanE(T *g_odata, T *g_idata, T *temp, const int n)
{
	int tID = threadIdx.x;  
	int offset = 1;
	
	int ai = tID;  							// load input into shared memory  
	int bi = tID + (n/2);  
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
	 	
	for(int d = n>>1; d > 0; d >>= 1)		// build sum in place up the tree  
	{
		__syncthreads();  
		if(tID < d)
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;  
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai]; 
		}
		offset <<= 1;
	}

	if(tID==0)
	{ temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }		// clear the last element
	 	
	for(int d = 1; d < n; d *= 2)			// traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();  

		if(tID < d)                  
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T t = temp[ai];  
			temp[ai] = temp[bi];  
			temp[bi] += t;   
		}
	}
	__syncthreads();

	g_odata[ai] = temp[ai + bankOffsetA];	// write results to device memory  
	g_odata[bi] = temp[bi + bankOffsetB];
}

template<typename T>
__device__ __forceinline__ void prescanE(T *data, const int n)
{
	int tID = threadIdx.x;  
	int offset = 1;
	 	
	for(int d = n>>1; d > 0; d >>= 1)		// build sum in place up the tree  
	{
		__syncthreads();  
		if(tID < d)
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;  
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			data[bi] += data[ai]; 
		}
		offset <<= 1;
	}

	if(tID==0)
	{ data[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }		// clear the last element
	 	
	for(int d = 1; d < n; d *= 2)			// traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();  

		if(tID < d)                  
		{
			int ai = offset*(2*tID+1)-1;  
			int bi = offset*(2*tID+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			T t = data[ai];  
			data[ai] = data[bi];  
			data[bi] += t;   
		}
	}
	__syncthreads();
}

template<typename T>
__device__ __forceinline__ void prescanI(T *data, const int n)
{
	int tID = threadIdx.x;  
	int offset = 1;
	 	
	for(int d = n>>1; d > 0; d >>= 1)		// build sum in place up the tree  
	{
		__syncthreads();
		if(tID < d)
		{
			int ai = offset*(2*tID+1)-1;
			int bi = offset*(2*tID+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			data[bi] += data[ai]; 
		}
		offset <<= 1;
	}

	offset >>= 1;
	for(int d = 2; d < n; d *= 2)			// traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();

		if(tID < d-1)
		{
			int ai = offset*(2*tID+2)-1;  
			int bi = offset*(2*tID+3)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
  
			data[bi] += data[ai];   
		}
	}
	__syncthreads();
}

#include "sort.inl"
#include "reduce.inl"
#include "reduce_global.inl"
//#include "radix_sort.inl"

#endif