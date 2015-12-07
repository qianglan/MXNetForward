CC = arm-linux-androideabi-g++
CFLAGS += -g -std=c++11  -fPIC -D_NDK_MATH_NO_SOFTFP=1
#-Wno-unknown-pragmas -Wall
export OPENBLAS_ROOT=`pwd`/OpenBLAS

INC +=

LIB += -lOpenCL
#LIB += -lblas -lrt

CFLAGS += -I${OPENBLAS_ROOT} -I./
LIB1 = -L./lib  -lOpenCL
#LIB2 = -static -L${OPENBLAS_ROOT} -lopenblas 
LIB2 = ${OPENBLAS_ROOT}/libopenblas.a


SOURCES = classify.cc mxnet_predict-withopeblas.cc
OBJS = $(SOURCES:.cc=.o)
EXECUTABLE = classify

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJS)
	$(CC) $(INC) $(OBJS) $(LIB1) $(LIB2) -o $@

.cc.o:
	$(CC) $(INC) $(CFLAGS)  -c $^ -o $@

clean:
	rm -rf $(OBJS) $(EXECUTABLE)
