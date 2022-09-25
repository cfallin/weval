#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

#ifdef __cplusplus
template<typename Ret, typename... Args>
struct FuncPtr {
    Ret (*func)(Args...);
};

namespace weval_impl {
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
}  // weval_impl

template<typename Ret, typename ...Args>
bool weval(FuncPtr<Ret, Args...>* dest, FuncPtr<Ret, Args...> generic, Args... args) {
    weval_req_t* req = (weval_req_t*)malloc(sizeof(weval_req_t));
    if (!req) {
        return false;
    }
    uint32_t nargs = weval_impl::arg_len<Args...>();
    weval_req_arg_t* arg_storage = (weval_req_arg_t*)malloc(sizeof(weval_req_arg_t) * nargs);
    if (!arg_storage) {
        return false;
    }
    store_args(arg_storage, args...);

    req->func = (weval_func_t)generic.func;
    req->args = arg_storage;
    req->nargs = nargs;
    req->specialized = (weval_func_t*)(&dest->func);

    req->next = weval_req_pending_head;
    weval_req_pending_head = req;

    return true;
}

#endif // __cplusplus

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
