CC = g++
CFLAGS += -g -std=c++11 -Wno-unknown-pragmas -Wall

INC +=

LIB += -lrt -lOpenCL

SOURCES = main.cc
OBJS = $(SOURCES:.cc=.o)
EXECUTABLE = mul

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJS)
	$(CC) $(INC) $(CFLAGS) $(OBJS) $(LIB) -o $@

.cc.o:
	$(CC) $(INC) $(CFLAGS) $(LIB) -c $^ -o $@

clean:
	rm -rf $(OBJS) $(EXECUTABLE)
