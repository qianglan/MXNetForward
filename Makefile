CC = g++
CFLAGS += -g -std=c++11 -Wno-unknown-pragmas -Wall -DMSHADOW_STAND_ALONE 

INC += 

LIB += -lblas -lrt -lOpenCL

SOURCES = classify.cc mxnet_predict-all.cc
OBJS = $(SOURCES:.cc=.o)
EXECUTABLE = classify

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJS)
	$(CC) $(INC) $(CFLAGS) $(OBJS) $(LIB) -o $@

.cc.o:
	$(CC) $(INC) $(CFLAGS) $(LIB) -c $^ -o $@

clean:
	rm -rf $(OBJS) $(EXECUTABLE)
