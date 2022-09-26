#include <weval.h>

weval_req_t* weval_req_pending_head;
weval_req_t* weval_req_freelist_head;

__attribute__((noinline))
__attribute__((export_name("weval.assume.const.memory")))
const void* weval_assume_const_memory(const void* p) {
    return p;
}

__attribute__((noinline))
__attribute__((export_name("weval.assume.const")))
uint64_t weval_assume_const(uint64_t x) {
    return x;
}

__attribute__((export_name("weval.pending.head")))
weval_req_t** __weval_pending_head() {
    return &weval_req_pending_head;
}

__attribute__((export_name("weval.freelist.head")))
weval_req_t** __weval_freelist_head() {
    return &weval_req_freelist_head;
}
