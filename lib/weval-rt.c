#include <weval.h>

weval_req_t* weval_req_pending_head;
weval_req_t* weval_req_freelist_head;

static int __hook = 0;
static int __loop_end = 0;

__attribute__((export_name("weval.hook")))
void set_hook() {
    __hook = 1 + __loop_end;
}


__attribute__((export_name("weval.assume.const")))
uint64_t weval_assume_const(uint64_t value) {
    if (__hook) {
        // Return a different constant than below so that whole-wasm opts don't
        // merge this function body with others.
        return 1;
    } else {
        return value;
    }
}

__attribute__((export_name("weval.assume.const.memory")))
const void* weval_assume_const_memory(const void* value) {
    if (__hook) {
        return 0;
    } else {
        return value;
    }
}

__attribute__((export_name("weval.loop.pc32")))
uint32_t weval_loop_pc32(uint32_t pc) {
    if (__hook) {
        return 2;
    } else {
        return pc;
    }
}

__attribute__((export_name("weval.loop.pc64")))
uint64_t weval_loop_pc64(uint64_t pc) {
    if (__hook) {
        return 3;
    } else {
        return pc;
    }
}

__attribute__((export_name("weval.loop.end")))
void weval_loop_end() {
    __loop_end++;
}

__attribute__((export_name("weval.pending.head")))
weval_req_t** __weval_pending_head() {
    return &weval_req_pending_head;
}

__attribute__((export_name("weval.freelist.head")))
weval_req_t** __weval_freelist_head() {
    return &weval_req_freelist_head;
}
