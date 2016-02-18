#ifndef REDUCE_INL
#define REDUCE_INL

template<int VT>
__device__ __forceinline__ void DeviceExpandFlagsToRows(int first, int endFlags, int rows[VT + 1]) {

	rows[0] = first;
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		if((1 << i) & endFlags)
			++first;
		rows[i + 1] = first;
	}
}

////////////////////////////////////////////////////////////////////////////////
// DeviceFindSegScanDelta
// Runs an inclusive max-index scan over binary inputs.
////////////////////////////////////////////////////////////////////////////////
template<int NT>
__device__ __forceinline__ int DeviceFindSegScanDelta(int tID, bool flag, int *delta_shared) {
	const int NumWarps = NT / 32;

	int warp = tID / 32;
	int lane = 31 & tID;
	uint warpMask = 0xffffffff >> (31 - lane);		// inclusive search
	uint ctaMask = 0x7fffffff >> (31 - lane);		// exclusive search

	uint warpBits = __ballot(flag);
	delta_shared[warp] = warpBits;
	__syncthreads();

	if(tID < NumWarps) {
		uint ctaBits = __ballot(0 != delta_shared[tID]);
		int warpSegment = 31 - __clz(ctaMask & ctaBits);
		int start = (-1 != warpSegment) ? (31 - __clz(delta_shared[warpSegment]) + 32 * warpSegment) : 0;
		delta_shared[NumWarps + tID] = start;
	}
	__syncthreads();

	// Find the closest flag to the left of this thread within the warp.
	// Include the flag for this thread.
	int start = 31 - __clz(warpMask & warpBits);
	if(-1 != start)
		start += ~31 & tID;
	else 
		start = delta_shared[NumWarps + warp];
	__syncthreads();

	return tID - start;
}

////////////////////////////////////////////////////////////////////////////////
// CTAScan
////////////////////////////////////////////////////////////////////////////////
template<int NT, typename T>
struct CTAScan {
	enum { Size = NT, Capacity = 2 * NT + 1 };

	__device__ __forceinline__ static T Scan(int tID, T x, T *storage, T *total, T identity = T(0)) {

		storage[tID] = x;
		int first = 0;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tID >= offset)
				x = (storage[first + tID - offset] + x);
			first = NT - first;
			storage[first + tID] = x;
			__syncthreads();
		}
		*total = storage[first + NT - 1];

		x = tID ? storage[first + tID - 1] : identity;

		__syncthreads();
		return x;
	}
};

////////////////////////////////////////////////////////////////////////////////
// CTASegScan
////////////////////////////////////////////////////////////////////////////////
template<int NT, typename T>
struct CTASegScan {
	enum { NumWarps = NT / 32, Size = NT, Capacity = 2 * NT };

	// Each thread passes the reduction of the LAST SEGMENT that it covers.
	// flag is set to true if there's at least one segment flag in the thread.
	// SegScan returns the reduction of values for the first segment in this
	// thread over the preceding threads.
	// Return the value init for the first thread.

	// When scanning single elements per thread, interpret the flag as a BEGIN
	// FLAG. If tID's flag is set, its value belongs to thread tID + 1, not 
	// thread tID.

	// The function returns the reduction of the last segment in the CTA.
	static __device__ __forceinline__ T SegScanDelta(int tID, int tIDDelta, T x, 
		T *storage, T *carryOut, T identity = 0) {

		// Run an inclusive scan 
		int first = 0;
		storage[first + tID] = x;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tIDDelta >= offset) 
				x = (storage[first + tID - offset] + x);
			first = NT - first;
			storage[first + tID] = x;
			__syncthreads();
		}

		// Get the exclusive scan.
		x = tID ? storage[first + tID - 1] : identity;
		*carryOut = storage[first + NT - 1];
		__syncthreads();
		return x;
	}

	static __device__ __forceinline__ T SegScan(int tID, T x, bool flag, T *storage,
		T* carryOut, T identity = 0) {

		// Find the left-most thread that covers the first segment of this 
		// thread.
		int tIDDelta = DeviceFindSegScanDelta<NT>(tID, flag, storage);

		return SegScanDelta(tID, tIDDelta, x, storage, carryOut, identity);
	}
};

////////////////////////////////////////////////////////////////////////////////
// KernelReduceByKeyPreprocess
////////////////////////////////////////////////////////////////////////////////
template<typename T, int NT, int VT>
__device__ __forceinline__ int KernelReduceByKeyPreprocess(T *keys, T *keys_out, const int count, int &threadCode) {

	//typedef INDEX_TYPE T;
	const int NV = NT * VT;

	int tID = threadIdx.x;
	//int count2 = count;

	for(int i=tID; i < NV; i+=NT)
		keys_out[i] = keys[i];
	__syncthreads();

	// Compare adjacent keys in each thread and mark discontinuities in 
	// endFlags bits.
	int endFlags = 0;
	if(count > NV) {
		T key = keys_out[VT * tID];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			T next = keys_out[VT * tID + 1 + i];
			if(key != next)
				endFlags |= 1 << i;
			key = next;
		}
	}
	else {
		T key = keys_out[VT * tID];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tID + 1 + i;
			T next = keys_out[index];
			if(index == count || index < count && key != next)
				endFlags |= 1 << i;
			key = next;
		}
	}
	__syncthreads();

	// Count the number of encountered end flags.
	int total;
	int scan = CTAScan<NT, T>::Scan(tID, __popc(endFlags), keys_out, &total);

	if(total) {
		// Find the segmented scan start for this thread.
		int tIDDelta = DeviceFindSegScanDelta<NT>(tID, 0 != endFlags, keys_out);
	
		// threadCodes:
		// 12:0 - end flags for up to 13 values per thread.
		// 19:13 - tID delta for up to 128 threads.
		// 30:20 - tile-local offset for first segment end in thread.
		threadCode = endFlags | (tIDDelta << 13) | (scan << 20);

		// Reconstruct row IDs from thread codes and the starting row offset.
		int rows[VT + 1];
		DeviceExpandFlagsToRows<VT>(scan, threadCode, rows);
		
		// Compact the location of the last element in each segment.
		int index = VT * tID;
		#pragma unroll
		for(int i = 0, idx = 0; i < VT; ++i)
			if(rows[i] != rows[i + 1]) {
				int pos = index + i;
				int seg = scan + idx;
				idx++;
				keys_out[seg] = keys[pos];
			}
		__syncthreads();
	}

	return total;
}

////////////////////////////////////////////////////////////////////////////////
// CTASegReduce
////////////////////////////////////////////////////////////////////////////////
template<int NT, int VT, bool HalfCapacity, typename T>
struct CTASegReduce {
	typedef CTASegScan<NT, T> SegScan;

	enum {
		NV = NT * VT,
		Capacity = HalfCapacity ? (NV / 2) : NV
	};
	
	__device__ __forceinline__ static void ReduceToShared(const int rows[VT + 1], int total, 
		int tIDDelta, int startRow, int tID, T data[VT], T *dest, T *storage, T identity = T(0)) {

		// Run a segmented scan within the thread.
		T x, localScan[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			x = i ? (x + data[i]) : data[i];
			localScan[i] = x;
			if(rows[i] != rows[i + 1])
				x = identity;
		}

		// Run a parallel segmented scan over the carry-out values to compute
		// carry-in.
		T carryOut;
		T carryIn = SegScan::SegScanDelta(tID, tIDDelta, x, storage, &carryOut);
		
		dest += startRow;
		if(HalfCapacity && total > Capacity) {
			// Add carry-in to each thread-local scan value. Store directly
			// to global.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				// Add the carry-in to the local scan.
				T x2 = carryIn + localScan[i];

				// Store on the end flag and clear the carry-in.
				if(rows[i] != rows[i + 1]) {
					carryIn = identity;
					dest[rows[i]] = x2;
				}
			}
			__syncthreads();
		}
		else {
			// All partials fit in shared memory. Add carry-in to each thread-
			// local scan value.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				// Add the carry-in to the local scan.
				T x2 = (carryIn + localScan[i]);

				// Store reduction when the segment changes and clear the 
				// carry-in.
				if(rows[i] != rows[i + 1]) {
					storage[rows[i]] = x2;
					carryIn = identity;
				}
			}
			__syncthreads();

			// Cooperatively store reductions to global memory.
			for(int index = tID; index < total; index += NT)
				dest[index] = storage[index];
			__syncthreads();
		}
	}
};

template<typename T, int NT, int VT>
__device__ __forceinline__ void KernelSegReduceApply(const int threadCodes,
	int count, T *values, T *values_out) {

	//const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int)) ? true : false;

	typedef CTASegReduce<NT, VT, HalfCapacity, T> SegReduce;

	int tID = threadIdx.x;
	// Load the data and transpose into thread order.
	T data[VT];
	#pragma unroll
	for(int i=0; i<VT; ++i)
		data[i] = values[tID*VT + i];

	// Expand the segment indices.
	int segs[VT + 1];
	DeviceExpandFlagsToRows<VT>(threadCodes >> 20, threadCodes, segs);

	// Reduce tile data and store to dest_global.
	int tIDDelta = 0x7f & (threadCodes >> 13);
	SegReduce::ReduceToShared(segs, count, tIDDelta, 0, tID, data, values_out, values);
}

#endif