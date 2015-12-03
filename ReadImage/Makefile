CC = g++
CFLAGS += -g -std=c++11 

INC += `pkg-config --cflags opencv`

LIB += `pkg-config --libs opencv` 

SOURCES = read_image.cc 
OBJS = $(SOURCES:.cc=.o)
EXECUTABLE = read_image

all: $(EXECUTABLE)
$(EXECUTABLE): $(OBJS)
	$(CC) $(INC) $(CFLAGS) $(OBJS) $(LIB) -o $@

.cc.o:
	$(CC) $(INC) $(CFLAGS) $(LIB) -c $^ -o $@

clean:
	rm -rf $(OBJS) $(EXECUTABLE) rgb.txt
