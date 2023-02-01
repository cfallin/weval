#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* partial-evaluation async requests and queues                              */
/* ------------------------------------------------------------------------- */

typedef void (*weval_func_t)();

typedef struct weval_req_t weval_req_t;

struct weval_req_t {
    weval_req_t* next;
    weval_func_t func;
    uint64_t func_ctx;
    uint64_t pc_ctx;
    void** specialized;
};

extern weval_req_t* weval_req_pending_head;
extern weval_req_t* weval_req_freelist_head;

static void weval_request(weval_req_t* req) {
    req->next = weval_req_pending_head;
    weval_req_pending_head = req;
}

static void weval_free() {
    weval_req_t* next = NULL;
    for (; weval_req_freelist_head; weval_req_freelist_head = next) {
        next = weval_req_freelist_head->next;
        free(weval_req_freelist_head);
    }
    weval_req_freelist_head = NULL;
}

/* ------------------------------------------------------------------------- */
/* intrinsics                                                                */
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/* "bless" a pointer so that all loads from it, directly and
 * indirectly, are "const" and allowed to see values during partial
 * evaluation. */
__attribute__((noinline))
const void* weval_assume_const_memory(const void* p);

/* Start a specialized region. Should come before any unrolled
 * loop. Returns the function context. */
__attribute__((noinline))
uint64_t weval_start(uint64_t func_ctx, uint64_t pc_ctx, void* const* specialized);
/* Within a specialized region, update the PC ctx. Value returned
 * should be used for all accesses throughout the specialized
 * region. If updated, next basic block edge goes to block of new
 * context. */
__attribute__((noinline))
uint64_t weval_pc_ctx(uint64_t pc_ctx);
/* End a specialized region. Should come after any loop. */
__attribute__((noinline))
void weval_end();

/* "bless" a pointer for memory renaming. */
__attribute__((noinline))
void* weval_make_symbolic_ptr(void* p);
/* flush a region of renamed memory back to memory, returning a pointer. */
__attribute__((noinline))
void* weval_flush_to_mem(void* p, uint32_t len);

#ifdef __cplusplus
}  // extern "C"
#endif

static int weval_enroll(weval_func_t func, uint64_t func_ctx, uint64_t pc_ctx, void** specialized) {
    weval_req_t* req = (weval_req_t*)malloc(sizeof(weval_req_t));
    if (!req) {
        return 0;
    }

    req->func = func;
    req->func_ctx = func_ctx;
    req->pc_ctx = pc_ctx;
    req->specialized = specialized;

    weval_request(req);

    return 1;
}

/* ------------------------------------------------------------------------- */
/*            C++ wrapper (`weval` namespace)                                */
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus

namespace weval {
    
template<typename T>
T* make_symbolic_ptr(T* ptr) {
    return reinterpret_cast<T*>(weval_make_symbolic_ptr(reinterpret_cast<void*>(ptr)));
}

template<typename T>
T* assume_const_memory(T* ptr) {
    return reinterpret_cast<T*>(weval_make_symbolic_ptr(reinterpret_cast<void*>(ptr)));
}

template<typename T>
const T* assume_const_memory(const T* ptr) {
    return reinterpret_cast<const T*>(weval_make_symbolic_ptr(reinterpret_cast<void*>(const_cast<T*>(ptr))));
}

template<typename T>
void flush_to_mem(T* ptr, size_t count) {
    weval_flush_to_mem(reinterpret_cast<void*>(ptr), sizeof(T) * count);
}

template<typename T>
void flush_to_mem(const T* ptr, size_t count) {
    weval_flush_to_mem(reinterpret_cast<void*>(ptr), sizeof(T) * count);
}
    
template<typename Func>
Func* start(Func* func_ctx, uint64_t pc_ctx, void* const* specialized) {
    return reinterpret_cast<Func*>(weval_start(reinterpret_cast<uint64_t>(func_ctx), pc_ctx, specialized));
}

template<typename Func>
const Func* start(const Func* func_ctx, uint64_t pc_ctx, void* const* specialized) {
    return reinterpret_cast<const Func*>(weval_start(reinterpret_cast<uint64_t>(func_ctx), pc_ctx, specialized));
}

static uint64_t pc_ctx(uint64_t pc) {
    return weval_pc_ctx(pc);
}

static void end() {
    weval_end();
}

template<typename InterpretFunc, typename Func>
void enroll(InterpretFunc interpret, const Func* func, uint64_t pc_ctx, void** specialized) {
    weval_enroll(reinterpret_cast<weval_func_t>(interpret), reinterpret_cast<uint64_t>(func), pc_ctx, specialized);
}

}  // namespace weval
#endif  // __cplusplus
