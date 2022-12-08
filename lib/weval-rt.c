#include <weval.h>

weval_req_t* weval_req_pending_head;
weval_req_t* weval_req_freelist_head;

static int __hook = 0;
static int __accum = 0;

__attribute__((export_name("weval.hook")))
void set_hook() {
    __hook = 1;
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

__attribute__((export_name("weval.push.context")))
void weval_push_context(uint32_t pc) {
    __accum += pc + 1;
}

__attribute__((export_name("weval.update.context")))
void weval_update_context(uint32_t pc) {
    __accum += pc + 2;
}

__attribute__((export_name("weval.pop.context")))
void weval_pop_context() {
    __accum += 3;
}

__attribute__((export_name("weval.pending.head")))
weval_req_t** __weval_pending_head() {
    return &weval_req_pending_head;
}

__attribute__((export_name("weval.freelist.head")))
weval_req_t** __weval_freelist_head() {
    return &weval_req_freelist_head;
}
