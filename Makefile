CXXFLAGS += -g -std=c++11 -Wno-deprecated-declarations -pthread $(shell pkg-config --cflags opencv)
LDFLAGS += -pthread
LDLIBS += -lboost_program_options $(shell pkg-config --libs opencv) -lglog -lgflags

.PHONY:	all 

all:	stat-ab test-ab

test-ab:	test-ab.cpp colorize.cpp

clean:
	rm -rf stat-ab *.o
