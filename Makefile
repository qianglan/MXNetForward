CC = arm-linux-androideabi-g++
#CC = g++
CFLAGS += -O3 -std=c++11  #-D_NDK_MATH_NO_SOFTFP=1 #-fPIC -D_NDK_MATH_NO_SOFTFP=1
#-Wno-unknown-pragmas -Wall
export OPENBLAS_ROOT=`pwd`/OpenBLAS

INC +=

#LIB += -lOpenCL
LIB += #-lblas -lrt

CFLAGS += -I${OPENBLAS_ROOT}  #-mhard-float #-I./ 
LIB1 += #-L./lib  -lOpenCL
LIB2 += -L${OPENBLAS_ROOT} -lopenblas #-Wl,--no-warn-mismatch -lm_hard
LIB2 += #${OPENBLAS_ROOT}/libopenblas.a


SOURCES = classify.cc mxnet_predict-all.cc
OBJS = $(SOURCES:.cc=.o)
EXECUTABLE = classify

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJS)
	$(CC) $(INC) $(CFLAGS) $(OBJS) -o $@ $(LIB1) $(LIB2)

.cc.o:
	$(CC) $(INC) $(CFLAGS)   -c $^ -o $@ $(LIB1) $(LIB2)

clean:
	rm -rf $(OBJS) $(EXECUTABLE)
