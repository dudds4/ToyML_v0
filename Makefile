RM = rm -f

CC = g++
CFLAGS = -O2 -std=c++14

HDRS = main.h graph.h nodetypes.h
SRCS = main.cpp graph.cpp nodetypes.cpp

all: app

app: bin bin/toyml

bin:
	mkdir bin

bin/toyml: ${SRCS}
	@printf "Compiling toy ml\n"
	$(CC) ${SRCS} $(CFLAGS) -o bin/toyml

clean:
	$(RM) -r bin
