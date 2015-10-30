#include "Matrix_Test.h"
#include "load_matrix.h"

void Matrix_Test(const std::string filename);
void createStreams();
void createStreams(const int ID);
void destroyStreams();
void destroyStreams(const int ID);

//#include "Fill_Tests.h"
#include "SPMM_Tests.h"

void createStreams(const int ID)
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamCreate(&__multiStreams[ID][i]);
}

void destroyStreams(const int ID)
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamDestroy(__multiStreams[ID][i]);
}

void createStreams()
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamCreate(&__streams[i]);
}

void destroyStreams()
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamDestroy(__streams[i]);
}

void Matrix_Test(const std::string filename)
{
	#if(MULTI_GPU == 1)
		//SPMMTests_Multi(filename);
	#else
		SPMMTests(filename);
	#endif
}
