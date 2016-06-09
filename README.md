# DCSR
Dynamic CSR matrix

This is still a work in progress...

To build just type make, but Makefile will need to be edited to match the specs/local paths on your machine.

There are a few matrix files in .mtx format in the matrices folder.

Usage:
./DCSR matrixfile


//comments on usage
The matrix format is compatible with existing formats.  Typical usage will involve conversion from another format such as CSR, COO, HYB, etc. followed by updates to the matrix and then conversion back to the previous format.  The code could also be written to use DCSR entirely.