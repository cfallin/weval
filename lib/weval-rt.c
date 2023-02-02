/*
  We follow a series of fairly subtle principles to ensure that the
  tool can recognize our calls to "intrinsics" (which are ordinary
  functions at the toolchain level, pre-weval):

  - We define function bodies here that do actually return the correct
    value, so execution before weval removes the intrinsics still works.

  - We define an export that modifies a global that makes the
    intrinsics do something else instead, so they can't be optimized
    away as the identity function.

  - We define a *different* function body for every intrinsic, so they
    aren't merged by the linker into one function.

  - We export them with well-known names so the tool can find calls to
    them.

  After wevaling, the intrinsics should no longer be referenced, and
  could be removed, in theory. (We don't yet do this though.)
 */

#include <weval.h>

// Global linked list of "requests" for weval to specialize. This
// linked list is read in the input snapshot by the tool.
weval_req_t* weval_req_pending_head;
weval_req_t* weval_req_freelist_head;

// Global used to create "theoretically possible" alternate behavior
// so that intrinsics aren't optimized away.
static int __hook = 0;
// Global used to create a side-effect on otherwise side-effect-free
// and `void`-returning intrinsics.
static int __accum = 0;

// We define this function and export it so that `__hook` can't be
// optimized away. In practice it should never be called.
__attribute__((export_name("weval.hook")))
void set_hook() {
    __hook = 1;
}


__attribute__((export_name("weval.assume.const")))
uint64_t weval_assume_const(uint64_t value) {
    if (__hook) {
        return 1;
    } else {
        return value;
    }
}

__attribute__((export_name("weval.assume.const.memory")))
const void* weval_assume_const_memory(const void* value) {
    if (__hook) {
        return (void*)8;
    } else {
        return value;
    }
}

__attribute__((export_name("weval.make.symbolic.ptr")))
void* weval_make_symbolic_ptr(void* value) {
    if (__hook) {
        return (void*)16;
    } else {
        return value;
    }
}

__attribute__((export_name("weval.flush.to.mem")))
void* weval_flush_to_mem(void* ptr, uint32_t len) {
    if (__hook) {
        return (void*)24;
    } else {
        return ptr;
    }
}

__attribute__((export_name("weval.start")))
uint64_t weval_start(uint64_t func_ctx, uint64_t pc_ctx, void* const* specialized) {
    if (__hook) {
        return 2;
    } else {
        return func_ctx;
    }
}

__attribute__((export_name("weval.pc.ctx")))
uint64_t weval_pc_ctx(uint64_t pc_ctx) {
    if (__hook) {
        return 3;
    } else {
        return pc_ctx;
    }
}

__attribute__((export_name("weval.func.call")))
void weval_func_call(uint64_t func_ctx, uint64_t pc_ctx, void* const* specialized) {
    __accum += 1;
}

__attribute__((export_name("weval.func.ret")))
void weval_func_ret() {
    __accum += 2;
}

__attribute__((export_name("weval.end")))
void weval_end() {
    __accum += 3;
}

// Allows the tool to find the linked list. Must have a very specific
// form: return a constant, with no longer logic.
__attribute__((export_name("weval.pending.head")))
weval_req_t** __weval_pending_head() {
    return &weval_req_pending_head;
}

__attribute__((export_name("weval.freelist.head")))
weval_req_t** __weval_freelist_head() {
    return &weval_req_freelist_head;
}

__attribute__((export_name("weval.accum")))
int __weval_accum() {
    return __accum;
}
