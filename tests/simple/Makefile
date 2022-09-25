NAME := simple

CXX := /opt/wasi-sdk/bin/clang
CXXFLAGS := -I../../include -std=c++17 -O2 -g

.PHONY: all
all: $(NAME).wasm

$(NAME).wasm: $(NAME).o
	$(CXX) $(CXXFLAGS) -o $@ $^

$(NAME).o: $(NAME).cpp ../../include/weval.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(NAME).wasm *.o
