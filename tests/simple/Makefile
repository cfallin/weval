NAME := simple

CC := /opt/wasi-sdk/bin/clang
CXX := /opt/wasi-sdk/bin/clang++
CFLAGS := -I../../include -O2 -g
CXXFLAGS := $(CFLAGS) -std=c++17
WIZER := wizer

.PHONY: all
all: $(NAME).wasm $(NAME)-wizened.wasm

$(NAME).wasm: $(NAME).o weval-rt.o
	$(CXX) $(CXXFLAGS) -o $@ $^

$(NAME)-wizened.wasm: $(NAME).wasm
	$(WIZER) -r _start=wizer.resume --wasm-bulk-memory true --allow-wasi $^ -o $@

$(NAME).o: $(NAME).cpp ../../include/weval.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

weval-rt.o: ../../lib/weval-rt.c
	$(CC) $(CFLAGS) -c -o $@ $^

.PHONY: clean
clean:
	rm -f $(NAME).wasm $(NAME)-wizened.wasm *.o
