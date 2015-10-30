#include "utils.h"

// forward declarations
void runTest(int argc, char **argv);
void ParseOptions(int argc, char **argv);

int main(int argc, char **argv)
{
	fprintf(stderr, "TEST START\n");
	//parse command line arguments
	ParseOptions(argc, argv);

	runTest(argc, argv);

	fprintf(stderr, "TEST COMPLETE\n");
	return 0;
}

void ParseOptions(int argc, char **argv)
{
	for(int i=1; i<argc; ++i)
	{
		char *arg = argv[i];

		//parse option
	}
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

	std::string filename(argv[1]);
	//Test(filename);
	Matrix_Test(filename);
}