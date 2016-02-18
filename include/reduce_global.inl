#ifndef REDUCE_GLOBAL_INL
#define REDUCE_GLOBAL_INL

struct SegReduceRange {
	int begin;
	int end;
	int total;
	bool flushLast;
};

////////////////////////////////////////////////////////////////////////////////
// CTAScan
template<int NT, typename T>
struct CTAScan_Global {
	enum { Size = NT, Capacity = 2 * NT + 1 };
	struct Storage { T shared[Capacity]; };

	__device__ __forceinline__ static T Scan(int tID, T x, Storage& storage, T* total, T identity = (T)0) {

		storage.shared[tID] = x;
		int first = 0;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tID >= offset)
				x = (storage.shared[first + tID - offset] + x);
			first = NT - first;
			storage.shared[first + tID] = x;
			__syncthreads();
		}
		*total = storage.shared[first + NT - 1];

		x = tID ? storage.shared[first + tID - 1] : identity;

		__syncthreads();
		return x;
	}

	__device__ __forceinline__ static T Scan(int tID, T x, Storage& storage) {
		T total;
		return Scan(tID, x, storage, &total, T(0));
	}
};

////////////////////////////////////////////////////////////////////////////////
// CTASegScan
template<int NT, typename T>
struct CTASegScan_Global {
	enum { NumWarps = NT / 32, Size = NT, Capacity = 2 * NT };
	union Storage {
		int delta[NumWarps];
		T values[Capacity];
	};

	// Each thread passes the reduction of the LAST SEGMENT that it covers.
	// flag is set to true if there's at least one segment flag in the thread.
	// SegScan returns the reduction of values for the first segment in this
	// thread over the preceding threads.
	// Return the value init for the first thread.

	// When scanning single elements per thread, interpret the flag as a BEGIN
	// FLAG. If tID's flag is set, its value belongs to thread tID + 1, not 
	// thread tID.

	// The function returns the reduction of the last segment in the CTA.
	__device__ __forceinline__ static T SegScanDelta(int tID, int tIDDelta, T x, 
		Storage& storage, T *carryOut, T identity = T(0)) {

		// Run an inclusive scan 
		int first = 0;
		storage.values[first + tID] = x;
		__syncthreads();

		#pragma unroll
		for(int offset = 1; offset < NT; offset += offset) {
			if(tIDDelta >= offset) 
				x = (storage.values[first + tID - offset] + x);
			first = NT - first;
			storage.values[first + tID] = x;
			__syncthreads();
		}

		// Get the exclusive scan.
		x = tID ? storage.values[first + tID - 1] : identity;
		*carryOut = storage.values[first + NT - 1];
		__syncthreads();
		return x;
	}

	__device__ __forceinline__ static T SegScan(int tID, T x, bool flag, Storage& storage,
		T *carryOut, T identity = T(0)) {

		// Find the left-most thread that covers the first segment of this thread.
		int tIDDelta = DeviceFindSegScanDelta<NT>(tID, flag, storage.delta);

		return SegScanDelta(tID, tIDDelta, x, storage, carryOut, identity);
	}
};

template<int NT, int VT, typename T>
__device__ __forceinline__ void DeviceGlobalToThreadDefault(int count, T *data, int tID, T *reg, T init) {

	data += VT * tID;
	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = __ldg(data + i);
	}
	else {
		count -= VT * tID;
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = (i < count) ? __ldg(data + i) : init;
	}
}

template<int NT, int VT, typename T>
__device__ __forceinline__ void DeviceGlobalToRegPred(int count, T *data, int tID, T* reg, bool sync) {

	// TODO: Attempt to issue 4 loads at a time.
	#pragma unroll
	for(int i = 0; i < VT; ++i) {
		int index = NT * i + tID;
		if(index < count)
			reg[i] = data[index];
	}

	if(sync)
		__syncthreads();
}

template<int NT, int VT, typename T>
__device__ __forceinline__ void DeviceGlobalToReg(int count, T *data, int tID, T* reg, bool sync) {

	if(count >= NT * VT) {
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			reg[i] = data[NT * i + tID];
	}
	else
		DeviceGlobalToRegPred<NT, VT>(count, data, tID, reg, false);

	if(sync)
		__syncthreads();
}

template<int NT, int VT, typename T>
__device__ __forceinline__ void DeviceRegToShared(const T *reg, int tID, T *dest, bool sync) {
	
	#pragma unroll
	for(int i = 0; i < VT; ++i)
		dest[NT * i + tID] = reg[i];

	if(sync)
		__syncthreads();
}

template<int NT, int VT0, int VT1, typename T>
__device__ __forceinline__ void DeviceGlobalToReg2(int count, T *data, int tID, T* reg, bool sync) {

	DeviceGlobalToReg<NT, VT0>(count, data, tID, reg, false);
	#pragma unroll
	for(int i = VT0; i < VT1; ++i) {
		int index = NT * i + tID;
		if(index < count) reg[i] = data[index];
	}

	if(sync)
		__syncthreads();
}

template<int NT, int VT0, int VT1, typename InputIt, typename T>
__device__ __forceinline__ void DeviceGlobalToShared2(int count, InputIt source, int tID, T* dest, bool sync = true) {

	T reg[VT1];
	DeviceGlobalToReg2<NT, VT0, VT1>(count, source, tID, reg, false);
	DeviceRegToShared<NT, VT1>(reg, tID, dest, sync);
}

////////////////////////////////////////////////////////////////////////////////
// KernelReduceByKeyPreprocess
// Stream through keys and find discontinuities. Compress these into a bitfield
// for each thread in the reduction CTA. Emit a count of discontinuities.
// These are scanned to provide limits.
template<typename T, int NT, int VT>
__device__ __forceinline__ void KernelReduceByKeyPreprocess_Global(T *keys_global, int count, int* threadCodes_global, int* counts_global) {

	const int NV = NT * VT;

	union Shared {
		T keys[NT * (VT + 1)];
		int indices[NT];
		typename CTAScan_Global<NT, T>::Storage scanStorage;
	};
	__shared__ Shared shared;

	int tID = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV * block;
	int count2 = min(NV + 1, count - gid);

	// Load the keys for this tile with one following element. This allows us 
	// to determine end flags for all segments.
	DeviceGlobalToShared2<NT, VT, VT + 1>(count2, keys_global + gid, tID, shared.keys);

	// Compare adjacent keys in each thread and mark discontinuities in 
	// endFlags bits.
	int endFlags = 0;
	if(count2 > NV) {
		T key = shared.keys[VT * tID];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			T next = shared.keys[VT * tID + 1 + i];
			if(key != next)
				endFlags |= 1 << i;
			key = next;
		}
	}
	else {
		T key = shared.keys[VT * tID];	
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tID + 1 + i;
			T next = shared.keys[index];
			if(index == count2 || (index < count2 && (key != next)))
				endFlags |= 1<< i;
			key = next;
		}
	}
	__syncthreads();
		
	// Count the number of encountered end flags.
	int total;
	int scan = CTAScan_Global<NT, T>::Scan(tID, __popc(endFlags), shared.scanStorage, &total);

	if(!tID)
		counts_global[block] = total;

	if(total) {
		// Find the segmented scan start for this thread.
		int tIDDelta = DeviceFindSegScanDelta<NT>(tID, 0 != endFlags, shared.indices);

		// threadCode:
		// 12:0 - end flags for up to 13 values per thread.
		// 19:13 - tID delta for up to 128 threads.
		// 30:20 - scan offset for streaming partials.
		int threadCode = endFlags | (tIDDelta << 13) | (scan << 20);
		threadCodes_global[NT * block + tID] = threadCode;
	}
}

////////////////////////////////////////////////////////////////////////////////
// CTASegReduce
// Core segmented reduction code. Supports fast-path and slow-path for intra-CTA
// segmented reduction. Stores partials to global memory.
// Callers feed CTASegReduce::ReduceToGlobal values in thread order.
template<int NT, int VT, bool HalfCapacity, typename T>
struct CTASegReduce_Global {
	typedef CTASegScan_Global<NT, T> SegScan;

	enum {
		NV = NT * VT,
		Capacity = HalfCapacity ? (NV / 2) : NV
	};

	union Storage {
		typename SegScan::Storage segScanStorage;
		T values[Capacity];
	};
	
	__device__ __forceinline__ static void ReduceToGlobal(const int rows[VT + 1], int total,
		int tIDDelta, int startRow, int block, int tID, T data[VT], T *dest_global, T *carryOut_global, Storage &storage, T identity = T(0)) {

		// Run a segmented scan within the thread.
		T x, localScan[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			x = i ? (x + data[i]) : data[i];
			localScan[i] = x;
			if(rows[i] != rows[i + 1]) x = identity;
		}

		// Run a parallel segmented scan over the carry-out values to compute
		// carry-in.
		T carryOut;
		T carryIn = SegScan::SegScanDelta(tID, tIDDelta, x, storage.segScanStorage, &carryOut);

		// Store the carry-out for the entire CTA to global memory.
		if(!tID) carryOut_global[block] = carryOut;
		
		dest_global += startRow;
		if(HalfCapacity && total > Capacity) {
			// Add carry-in to each thread-local scan value. Store directly to global.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				// Add the carry-in to the local scan.
				T x2 = (carryIn + localScan[i]);

				// Store on the end flag and clear the carry-in.
				if(rows[i] != rows[i + 1]) {
					carryIn = identity;
					dest_global[rows[i]] = x2;
				}
			}
		}
		else {
			// All partials fit in shared memory. Add carry-in to each thread-local scan value.
			#pragma unroll
			for(int i = 0; i < VT; ++i) {
				// Add the carry-in to the local scan.
				T x2 = (carryIn + localScan[i]);

				// Store reduction when the segment changes and clear the carry-in.
				if(rows[i] != rows[i + 1]) {
					storage.values[rows[i]] = x2;
					carryIn = identity;
				}
			}
			__syncthreads();

			// Cooperatively store reductions to global memory.
			for(int index = tID; index < total; index += NT)
				dest_global[index] = storage.values[index];
			__syncthreads();
		}
	}
};

////////////////////////////////////////////////////////////////////////////////
// CTAReduce
////////////////////////////////////////////////////////////////////////////////
template<int NT>
struct CTAReduce_Global {
	enum { Size = NT, Capacity = WARP_SIZE };
	struct Storage { int shared[Capacity]; };

	__device__ __forceinline__ static int Reduce(int tID, int x, Storage& storage) {

		const int NumSections = WARP_SIZE;
		const int SecSize = NT / NumSections;
		int lane = (SecSize - 1) & tID;
		int sec = tID / SecSize;

		// In the first phase, threads cooperatively find the reduction within
		// their segment. The segments are SecSize threads (NT / WARP_SIZE) 
		// wide.
		#pragma unroll
		for(int offset = 1; offset < SecSize; offset *= 2)
			x = shfl_up_add(x, offset, SecSize);

		// The last thread in each segment stores the local reduction to shared
		// memory.
		if(SecSize - 1 == lane) storage.shared[sec] = x;
		__syncthreads();

		// Reduce the totals of each input segment. The spine is WARP_SIZE 
		// threads wide.
		if(tID < NumSections) {
			x = storage.shared[tID];
			#pragma unroll
			for(int offset = 1; offset < NumSections; offset *= 2)
				x = shfl_up_add(x, offset, NumSections);
			storage.shared[tID] = x;
		}
		__syncthreads();

		int reduction = storage.shared[NumSections - 1];
		__syncthreads();

		return reduction;
	}
};

////////////////////////////////////////////////////////////////////////////////
// CTAIntervalSegReduceGather
// Storage and logic for segmented reduce and interval reduce.
// Pass the Reduce function data in thread order.
template<int NT, int VT, bool HalfCapacity, bool LdgTranspose, typename T>
struct CTASegReduceLoad_Global {
	enum {
		NV = NT * VT,
		Capacity = HalfCapacity ? (NV / 2) : NV
	};
	
	union Storage {
		int sources[NV];
		T data[Capacity];
	};
	
	// Load elements from multiple segments and store in thread order.
	__device__ __forceinline__ static void LoadDirect(int count2, int tID, int gid, T *data_global, T data[VT], Storage &storage) {

		if(LdgTranspose) {
			// Load data in thread order from data_global + gid.
			DeviceGlobalToThreadDefault<NT, VT, T>(count2, data_global + gid, tID, data, T(0));
		}
		// else {
		// 	// Load data in strided order from data_global + gid.
		// 	T stridedData[VT];
		// 	DeviceGlobalToRegDefault<NT, VT>(count2, data_global + gid, tID, stridedData, identity);

		// 	if(HalfCapacity)
		// 		HalfSmemTranspose<NT, VT>(stridedData, tID, storage.data, data);
		// 	else {
		// 		DeviceRegToShared<NT, VT>(stridedData, tID, storage.data);
		// 		DeviceSharedToThread<VT>(storage.data, tID, data);
		// 	}
		// }
	}
};

__device__ __forceinline__ SegReduceRange DeviceShiftRange(int limit0, int limit1) {
	SegReduceRange range;
	range.begin = 0x7fffffff & limit0;
	range.end = 0x7fffffff & limit1; 
	range.total = range.end - range.begin;
	range.flushLast = 0 == (0x80000000 & limit1);
	range.end += !range.flushLast;
	return range;
}

template<typename T, int NT, int VT>
__device__ __forceinline__ void KernelSegReduceApply_Global(const int *threadCodes_global,
	int count, const int *limits_global, T *data_global, T *dest_global, T *carryOut_global) {

	const int NV = NT * VT;
	const bool HalfCapacity = (sizeof(T) > sizeof(int));

	typedef CTAReduce_Global<NT> FastReduce;
	typedef CTASegReduce_Global<NT, VT, HalfCapacity, T> SegReduce;
	typedef CTASegReduceLoad_Global<NT, VT, HalfCapacity, true, T> SegReduceLoad;

	union Shared {
		int csr[NV];
		typename FastReduce::Storage reduceStorage;
		typename SegReduce::Storage segReduceStorage;
		typename SegReduceLoad::Storage loadStorage;
	};
	__shared__ Shared shared;

	int tID = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV * block;
	int count2 = min(NV, count - gid);

	int limit0 = limits_global[block];
	int limit1 = limits_global[block + 1];
	int threadCodes = threadCodes_global[NT * block + tID];

	// Load the data and transpose into thread order.
	T data[VT];
	SegReduceLoad::LoadDirect(count2, tID, gid, data_global, data, shared.loadStorage);

	// Compute the range.
	SegReduceRange range = DeviceShiftRange(limit0, limit1);

	if(range.total) {
		// Expand the segment indices.
		int segs[VT + 1];
		DeviceExpandFlagsToRows<VT>(threadCodes >> 20, threadCodes, segs);

		// Reduce tile data and store to dest_global. Write tile's carry-out
		// term to carryOut_global.
		int tIDDelta = 0x7f & (threadCodes >> 13);
		SegReduce::ReduceToGlobal(segs, range.total, tIDDelta, range.begin,
			block, tID, data, dest_global, carryOut_global, shared.segReduceStorage);
	}
	else {
		// If there are no end flags in this CTA, use a fast reduction.
		T x;
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			x = i ? (x + data[i]) : data[i];
		x = FastReduce::Reduce(tID, x, shared.reduceStorage);
		if(!tID)
			carryOut_global[block] = x;
	}
}

////////////////////////////////////////////////////////////////////////////////
// KernelReduceByKeyEmit
template<typename T, int NT, int VT>
__device__ __forceinline__ void KernelReduceByKeyEmit(T *keys_global,
	int count, const int *threadCodes_global, const int *limits_global,
	T *keysDest_global) {

	const int NV = NT * VT;

	union Shared {
		int indices[NV];
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NV * block;

	int limit0 = limits_global[block];
	int limit1 = limits_global[block + 1];
	int threadCodes = threadCodes_global[NT * block + tid];

	int total = limit1 - limit0;
	if(total) {
		// Reconstruct row IDs from thread codes and the starting row offset.
		int rows[VT + 1];
		DeviceExpandFlagsToRows<VT>(threadCodes>> 20, threadCodes, rows);
		
		// Compact the location of the last element in each segment.
		int index = gid + VT * tid;
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			if(rows[i] != rows[i + 1])
				shared.indices[rows[i]] = index + i;
		__syncthreads();

		// Copy a key from the start of each segment.
		for(int i = tid; i < total; i += NT) {
			int pos = shared.indices[i] + 1;
			int seg = limit0 + i + 1;
			if(pos >= count) {
				pos = 0;
				seg = 0;
			}
			keysDest_global[seg] = keys_global[pos];
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// SegReduceSpine
// Compute the carry-in in-place. Return the carry-out for the entire tile.
// A final spine-reducer scans the tile carry-outs and adds into individual
// results.
template<typename T, int NT>
__global__ void KernelSegReduceSpine1(const int *limits_global, int count,
	T *dest_global, const T *carryIn_global, T identity, T *carryOut_global) {

	typedef CTASegScan_Global<NT, T> SegScan;
	union Shared {
		typename SegScan::Storage segScanStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	int block = blockIdx.x;
	int gid = NT * block + tid;

	// Load the current carry-in and the current and next row indices.
	int row = (gid < count) ? 
		(0x7fffffff & limits_global[gid]) :
		INT_MAX;
	int row2 = (gid + 1 < count) ? 
		(0x7fffffff & limits_global[gid + 1]) :
		INT_MAX;
	
	T carryIn2 = (gid < count) ? carryIn_global[gid] : identity;
	T dest = (gid < count) ? dest_global[row] : identity;

	// Run a segmented scan of the carry-in values.
	bool endFlag = row != row2;

	T carryOut;
	T x = SegScan::SegScan(tid, carryIn2, endFlag, shared.segScanStorage, &carryOut, identity);
			
	// Store the reduction at the end of a segment to dest_global.
	if(endFlag)
		dest_global[row] = (x + dest);
	
	// Store the CTA carry-out.
	if(!tid)
		carryOut_global[block] = carryOut;
}

template<typename T, int NT>
__global__ void KernelSegReduceSpine2(const int* limits_global, int numBlocks,
	int count, int nv, T *dest_global, const T *carryIn_global, T identity) {

	typedef CTASegScan_Global<NT, T> SegScan;
	struct Shared {
		typename SegScan::Storage segScanStorage;
		int carryInRow;
		T carryIn;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;
	
	for(int i = 0; i < numBlocks; i += NT) {
		int gid = (i + tid) * nv;

		// Load the current carry-in and the current and next row indices.
		int row = (gid < count) ? 
			(0x7fffffff & limits_global[gid]) : INT_MAX;
		int row2 = (gid + nv < count) ? 
			(0x7fffffff & limits_global[gid + nv]) : INT_MAX;
		T carryIn2 = (i + tid < numBlocks) ? carryIn_global[i + tid] : identity;
		T dest = (gid < count) ? dest_global[row] : identity;

		// Run a segmented scan of the carry-in values.
		bool endFlag = row != row2;
		
		T carryOut;
		T x = SegScan::SegScan(tid, carryIn2, endFlag, shared.segScanStorage, &carryOut, identity);

		// Add the carry-in to the reductions when we get to the end of a segment.
		if(endFlag) {
			// Add the carry-in from the last loop iteration to the carry-in
			// from this loop iteration.
			if(i && row == shared.carryInRow) 
				x = (shared.carryIn + x);
			dest_global[row] = (x + dest);
		}

		// Set the carry-in for the next loop iteration.
		if(i + NT < numBlocks) {
			__syncthreads();
			if(i > 0) {
				// Add in the previous carry-in.
				if(NT - 1 == tid) {
					shared.carryIn = (shared.carryInRow == row2) ? (shared.carryIn + carryOut) : carryOut;
					shared.carryInRow = row2;
				}
			}
			else {
				if(NT - 1 == tid) {
					shared.carryIn = carryOut;
					shared.carryInRow = row2;
				}
			}
			__syncthreads();
		}
	}
}

#endif