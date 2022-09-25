#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* partial-evaluation async requests and queues                              */
/* ------------------------------------------------------------------------- */

typedef void (*weval_func_t)();

typedef struct weval_req_t weval_req_t;
typedef struct weval_req_arg_t weval_req_arg_t;

struct weval_req_t {
    weval_req_t* next;
    weval_func_t func;
    weval_req_arg_t* args;
    uint32_t nargs;
    weval_func_t* specialized;
};

struct weval_req_arg_t {
    uint64_t specialize;
    union {
        uint32_t i32;
        uint64_t i64;
        float f32;
        double f64;
    } u;
};

extern weval_req_t* weval_req_pending_head;
extern weval_req_t* weval_req_freelist_head;

#define WEVAL_GLOBALS()                                 \
    weval_req_t* weval_req_pending_head;                \
    weval_req_t* weval_req_freelist_head;               \
                                                        \
    __attribute__((export_name("weval.pending.head")))  \
   weval_req_t** __weval_pending_head() {               \
       return &weval_req_pending_head;                  \
   }                                                    \
                                                        \
    __attribute__((export_name("weval.freelist.head"))) \
   weval_req_t** __weval_freelist_head() {              \
       return &weval_req_freelist_head;                 \
   }

static void weval_request(weval_req_t* req) {
    req->next = weval_req_pending_head;
    weval_req_pending_head = req;
}

static void weval_free() {
    weval_req_t* next = NULL;
    for (; weval_req_freelist_head; weval_req_freelist_head = next) {
        next = weval_req_freelist_head->next;
        if (weval_req_freelist_head->args) {
            free(weval_req_freelist_head->args);
        }
        free(weval_req_freelist_head);
    }
    weval_req_freelist_head = NULL;
}

/* ------------------------------------------------------------------------- */
/* intrinsics                                                                */
/* ------------------------------------------------------------------------- */

const void* weval_assume_const_memory(const void* p);
uint64_t weval_assume_const(uint64_t p);
void weval_unroll_loop();

#define WEVAL_INTRINSIC_BODIES()                                              \
    __attribute__((noinline))                                                 \
    __attribute__((export_name("weval.assume.const.memory")))                 \
    const void* weval_assume_const_memory(const void* p) {                    \
        return p;                                                             \
    }                                                                         \
    __attribute__((noinline))                                                 \
    __attribute__((export_name("weval.assume.const")))                        \
    uint64_t weval_assume_const(uint64_t x) {                                 \
        return x;                                                             \
    }                                                                         \
    __attribute__((noinline))                                                 \
    __attribute__((export_name("weval.unroll.loop")))                         \
    void weval_unroll_loop() {                                                \
    }

#ifdef __cplusplus
namespace weval {
template<typename T>
const T* assume_const_memory(const T* t) {
    return (const T*)weval_assume_const_memory((const void*)t);
}

template<typename T>
T assume_const(T t);

template<>
uint32_t assume_const(uint32_t x) {
    return (uint32_t)weval_assume_const((uint64_t)x);
}
template<>
uint64_t assume_const(uint64_t x) {
    return weval_assume_const(x);
}
template<typename T>
T* assume_const(T* t) {
    return (T*)weval_assume_const((uint64_t)t);
}
template<typename T>
const T* assume_const(const T* t) {
    return (T*)weval_assume_const((uint64_t)t);
}

static void unroll_loop() {
    weval_unroll_loop();
}

}  // namespace weval
#endif  // __cplusplus

/* ------------------------------------------------------------------------- */
/* C++ type-safe wrapper for partial evaluation of functions                 */
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
namespace weval {

template<typename Ret, typename... Args>
struct FuncPtr {
    Ret (*func)(Args...);
};

namespace impl {
template<typename... Args>
constexpr int arg_len();

template<typename Arg1, typename... Args>
constexpr int arg_len() {
    return arg_len<Args...>() + 1;
}
template<>
constexpr int arg_len() {
    return 0;
}

template<typename... Args>
void store_args(weval_req_arg_t* storage, Args... args);

template<typename Arg1, typename... Args>
void store_args(weval_req_arg_t* args, uint32_t arg1, Args... rest) {
    args[0].specialize = 1;
    args[0].u.i32 = arg1;
    store_args(args + 1, rest...);
}

template<typename Arg1, typename... Args>
void store_args(weval_req_arg_t* args, uint64_t arg1, Args... rest) {
    args[0].specialize = 1;
    args[0].u.i64 = arg1;
    store_args(args + 1, rest...);
}

template<typename Arg1, typename... Args>
void store_args(weval_req_arg_t* args, float arg1, Args... rest) {
    args[0].specialize = 1;
    args[0].u.f32 = arg1;
    store_args(args + 1, rest...);
}

template<typename Arg1, typename... Args>
void store_args(weval_req_arg_t* args, double arg1, Args... rest) {
    args[0].specialize = 1;
    args[0].u.f64 = arg1;
    store_args(args + 1, rest...);
}

template<typename Arg1, typename... Args>
void store_args(weval_req_arg_t* args, Arg1* arg1, Args... rest) {
    static_assert(sizeof(Arg1*) == 4, "Only 32-bit Wasm supported");
    args[0].specialize = 1;
    args[0].u.i32 = reinterpret_cast<uint32_t>(arg1);
    store_args(args + 1, rest...);
}

template<>
void store_args(weval_req_arg_t* args) {
    (void)args;
}
}  // impl

template<typename Ret, typename ...Args>
bool weval(FuncPtr<Ret, Args...>* dest, FuncPtr<Ret, Args...> generic, Args... args) {
    weval_req_t* req = (weval_req_t*)malloc(sizeof(weval_req_t));
    if (!req) {
        return false;
    }
    uint32_t nargs = impl::arg_len<Args...>();
    weval_req_arg_t* arg_storage = (weval_req_arg_t*)malloc(sizeof(weval_req_arg_t) * nargs);
    if (!arg_storage) {
        return false;
    }
    impl::store_args(arg_storage, args...);

    req->func = (weval_func_t)generic.func;
    req->args = arg_storage;
    req->nargs = nargs;
    req->specialized = (weval_func_t*)(&dest->func);

    weval_request(req);

    return true;
}

}  // namespace weval

#endif // __cplusplus
