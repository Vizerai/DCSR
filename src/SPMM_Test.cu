#include "Matrix_Test.h"
#include "load_matrix.h"

void FillTests(const std::string &filename);
void SPMM_Test(const std::string filenameA, const std::string filenameB, const std::string filenameC);
void Matrix_Test(const std::string filename);
void createStreams();
void createStreams(const int ID);
void destroyStreams();
void destroyStreams(const int ID);

#include "SPMM_Test.inl"

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

void SPMM_Test(const std::string filenameA, const std::string filenameB, const std::string filenameC)
{
	#if(MULTI_GPU == 1)
		SPMMTests(filenameA, filenameB, filenameC);
	#else
		SPMMTests(filenameA, filenameB, filenameC);
	#endif
}

////////////////////////////////////////////////////////////////////////////////
//	Parse input file and run test
////////////////////////////////////////////////////////////////////////////////

void runTest(int argc, char** argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "Invalid input...\n");
		fprintf(stderr, "Usage: CFA <testfile>\n");
		exit(1);
	}

	std::string filenameA(argv[1]);
	std::string filenameB(argv[1]);
	std::string filenameC(argv[1]);
	SPMM_Test(filenameA, filenameB, filenameC);
}

int main(int argc, char **argv)
{
	fprintf(stderr, "TEST START\n");
	runTest(argc, argv);
	fprintf(stderr, "TEST COMPLETE\n");
	return 0;
}