#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <cstdio>
#include <string>

//extern void Test(const std::string &filename);
extern void Matrix_Test(const std::string filename);
extern void SPMM_Test(const std::string filenameA, const std::string filenameB, const std::string filenameC);

#endif