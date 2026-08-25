LLAMA ?= $(HOME)/projects/llama.cpp
BUILD ?= $(LLAMA)/build-hip

CXX      ?= g++
CXXFLAGS  = -O2 -std=c++17 -Wall \
            -I$(LLAMA)/include -I$(LLAMA)/common -I$(LLAMA)/src \
            -I$(LLAMA)/ggml/include -I$(BUILD)/common
LIBS      = -L$(BUILD)/bin -lllama-common -lllama -lggml -lggml-base \
            -Wl,-rpath,$(BUILD)/bin

all: bin/tree-dump

bin/tree-dump: src/tree_dump.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf bin

.PHONY: all clean
