RM = rm -f

CC = g++
CFLAGS = -O2 -std=c++17

LIBHDRS = inc/graph.h inc/nodetypes.h inc/layers.h inc/nodeset.h
LIBSRCS = src/graph.cpp src/nodetypes.cpp src/layers.cpp

INCLUDES = inc

MSRCS = main.cpp
TSRCS = tests.cpp

all: bin bin/toyml bin/tests

bin:
	mkdir bin

bin/toyml: ${MSRCS} ${LIBHDRS} ${LIBSRCS}
	@printf "Compiling toyml\n"
	$(CC) -I${INCLUDES} ${MSRCS} ${LIBSRCS} $(CFLAGS) -o bin/toyml

bin/tests: ${TSRCS} ${LIBHDRS} ${LIBSRCS}
	@printf "Compiling tests\n"
	$(CC) -I${INCLUDES} ${TSRCS} ${LIBSRCS} $(CFLAGS) -o bin/tests

clean:
	$(RM) -r bin
