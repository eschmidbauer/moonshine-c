CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic
LDLIBS  := -lm -lpthread

.PHONY: all clean test

all: test_process test_streaming

moonshine.o: moonshine.c moonshine.h
	$(CC) $(CFLAGS) -c -o $@ moonshine.c

test_process: test_process.c moonshine.o moonshine.h
	$(CC) $(CFLAGS) -o $@ test_process.c moonshine.o $(LDLIBS)

test: test_process
	./test_process tiny input.wav test.wav input.wav test.wav

test_streaming: test_streaming.c moonshine.o moonshine.h
	$(CC) $(CFLAGS) -o $@ test_streaming.c moonshine.o $(LDLIBS)

clean:
	$(RM) moonshine.o test_process test_streaming
