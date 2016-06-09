CC=gcc
CPP=g++
NVCC=/usr/local/cuda/bin/nvcc

OMP=-fopenmp
OPT=-O3
DEBUG=-g -Wall -Wno-reorder
CUDA_DEBUG=-g -G
CUDA_CC_FLAGS=--generate-code arch=compute_30,code=sm_30 -m64 -Xcompiler -rdynamic -Xcompiler -fopenmp
LD_LIBS=-lcuda -lcudart -lcublas -lrt -llapack -lgomp 
#LIB_INCLUDE=-L /usr/local/cuda/lib64
CUDA_INCLUDE=-I../ -I../include -I../cub -I/usr/local/cuda/include -I/usr/local/cuda/NVIDIA_CUDA-7.0_Samples/common/inc 
#-I/usr/local/cuda-5.0/samples/common/inc/
PROJECT=DCSR

#default to optimized build for now
OPT_FLAGS=$(OPT)
NVCC_OPT_FLAGS=$(OPT)

all: $(PROJECT)

profile: 
	$(MAKE) DEFINES=-DPROFILE

ptx:
	$(MAKE) CUDA_CC_FLAGS="$(CUDA_CC_FLAGS) -ptx"

verbose: 
	$(MAKE) DEFINES=-DDEBUG \
	CUDA_CC_FLAGS=#(CUDA_CC_FLAGS) -Xptxas-v --ptxas-options=-v

opt:
	$(MAKE) OPT_FLAGS="$(OPT)" NVCC_OPT_FLAGS="$(OPT)"

debug:
	$(MAKE) OPT_FLAGS="$(DEBUG)" NVCC_OPT_FLAGS="$(CUDA_DEBUG)"

SRCS=../src/Fill_Test.cu $(wildcard ../include/*.cu)
OBJS=$(SRCS:.cu=.o)
INCLUDE_FILES=$(wildcard ../include/*.h) $(wildcard ../include/*.inl)
CUDA_INCLUDE_FILES=$(wildcard ../include/*.cuh) $(wildcard ../include/*.cinl)
CPP_FILES=
CPP_OBJS=$(CPP_FILES:.cpp=.o)

%.o : %.cu $(INCLUDE_FILES) $(CUDA_INCLUDE_FILES)
	$(NVCC) ${CUDA_INCLUDE} ${NVCC_OPT_FLAGS} ${CXX_FLAGS} ${CUDA_CC_FLAGS} -c $< -o $@ 

%.o : %.cpp $(INCLUDE_FILES)
	$(CPP) ${CUDA_INCLUDE} ${OPT_FLAGS} $(OMP) -lpthread -c $< -o $@

.PHONY clean:
	rm -f ../src/*.o
	rm -f $(PROJECT)

$(PROJECT): $(CPP_FILES) $(CPP_OBJS) $(OBJS)
	$(NVCC) -o $(PROJECT) ${NVCC_OPT_FLAGS} ${CUDA_CC_FLAGS} ${LD_LIBS} $(CPP_OBJS) $(OBJS)
#	$(CPP) -o $(PROJECT) ${OPT_FLAGS} $(OMP) $(LIB_INCLUDE) ${LD_LIBS} $(CPP_OBJS)
