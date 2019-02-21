RM = rm -f

CC = g++
CFLAGS = -O2 -std=c++17

LIBHDRS = graph.h nodetypes.h layers.h
LIBSRCS = graph.cpp nodetypes.cpp

MSRCS = main.cpp
TSRCS = tests.cpp

all: bin bin/toyml bin/tests

bin:
	mkdir bin

bin/toyml: ${MSRCS} ${LIBHDRS} ${LIBSRCS}
	@printf "Compiling toyml\n"
	$(CC) ${MSRCS} ${LIBSRCS} $(CFLAGS) -o bin/toyml

bin/tests: ${TSRCS} ${LIBHDRS} ${LIBSRCS}
	@printf "Compiling tests\n"
	$(CC) ${TSRCS} ${LIBSRCS} $(CFLAGS) -o bin/tests

clean:
	$(RM) -r bin
