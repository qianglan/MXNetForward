CC = arm-linux-androideabi-g++
CFLAGS += -g -std=c++11  -fPIC -D_NDK_MATH_NO_SOFTFP=1
#-Wno-unknown-pragmas -Wall
export OPENBLAS_ROOT=`pwd`/OpenBLAS

INC +=

LIB += -lOpenCL
#LIB += -lblas -lrt

CFLAGS += -I${OPENBLAS_ROOT} -I./
LIB += -L${OPENBLAS_ROOT} -lopenblas -L./lib



SOURCES = classify.cc mxnet_predict-withopeblas.cc
OBJS = $(SOURCES:.cc=.o)
EXECUTABLE = classify

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJS)
	$(CC) $(INC) $(CFLAGS) $(OBJS) $(LIB) -o $@

.cc.o:
	$(CC) $(INC) $(CFLAGS) $(LIB) -c $^ -o $@

clean:
	rm -rf $(OBJS) $(EXECUTABLE)
