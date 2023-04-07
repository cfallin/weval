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

typedef enum {
  weval_req_arg_i32 = 0,
  weval_req_arg_i64 = 1,
  weval_req_arg_f32 = 2,
  weval_req_arg_f64 = 3,
} weval_req_arg_type;

struct weval_req_arg_t {
  uint32_t specialize;
  uint32_t ty;
  union {
    uint32_t i32;
    uint64_t i64;
    float f32;
    double f64;
  } u;
};

extern weval_req_t* weval_req_pending_head;
extern weval_req_t* weval_req_freelist_head;

static inline void weval_request(weval_req_t* req) {
  req->next = weval_req_pending_head;
  weval_req_pending_head = req;
}

static inline void weval_free() {
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

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noinline)) const void* weval_assume_const_memory(const void* p);
__attribute__((noinline)) const void* weval_assume_const_memory_transitive(const void* p);
__attribute__((noinline)) void weval_push_context(uint32_t pc);
__attribute__((noinline)) void weval_pop_context();
__attribute__((noinline)) void weval_update_context(uint32_t pc);
__attribute__((noinline)) void* weval_make_symbolic_ptr(void* p);
__attribute__((noinline)) void* weval_alias_with_symbolic_ptr(void* p, void* symbolic);
__attribute__((noinline)) void weval_flush_to_mem();
__attribute__((noinline)) void weval_trace_line(uint32_t line_number);
__attribute__((noinline)) void weval_abort_specialization(uint32_t line_number,
                                                          uint32_t fatal);
__attribute__((noinline)) void weval_assert_const32(uint32_t value, uint32_t line_no);
__attribute__((noinline)) void weval_assert_const_memory(void* p, uint32_t line_no);
__attribute__((noinline)) uint32_t weval_specialize_value(uint32_t value, uint32_t lo, uint32_t hi);
__attribute__((noinline)) void weval_print(const char* message, uint32_t line, uint32_t val);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
namespace weval {
template <typename T>
const T* assume_const_memory(const T* t) {
  return (const T*)weval_assume_const_memory((const void*)t);
}
template <typename T>
T* assume_const_memory(T* t) {
  return (T*)weval_assume_const_memory((void*)t);
}
template <typename T>
const T* assume_const_memory_transitive(const T* t) {
  return (const T*)weval_assume_const_memory_transitive((const void*)t);
}
template <typename T>
T* assume_const_memory_transitive(T* t) {
  return (T*)weval_assume_const_memory_transitive((void*)t);
}

static inline void push_context(uint32_t pc) { weval_push_context(pc); }

static inline void pop_context() { weval_pop_context(); }

static inline void update_context(uint32_t pc) { weval_update_context(pc); }
template <typename T>
static T* make_symbolic_ptr(T* t) {
  return (T*)weval_make_symbolic_ptr((void*)t);
}
template <typename T>
void flush_to_mem() {
    weval_flush_to_mem();
}

}  // namespace weval
#endif  // __cplusplus

/* ------------------------------------------------------------------------- */
/* C++ type-safe wrapper for partial evaluation of functions                 */
/* ------------------------------------------------------------------------- */

#ifdef __cplusplus
namespace weval {

template <typename T>
struct ArgSpec {};

template <typename T>
struct RuntimeArg : ArgSpec<T> {};

template <typename T>
RuntimeArg<T> Runtime() {
  return RuntimeArg<T>{};
}

template <typename T>
struct Specialize : ArgSpec<T> {
  T value;
  explicit Specialize(T value_) : value(value_) {}
};

namespace impl {
template <typename Ret, typename... Args>
using FuncPtr = Ret (*)(Args...);

template <typename T>
struct StoreArg;

template <>
struct StoreArg<uint32_t> {
  void operator()(weval_req_arg_t* arg, uint32_t value) {
    arg->specialize = 1;
    arg->ty = weval_req_arg_i32;
    arg->u.i32 = value;
  }
};
template <>
struct StoreArg<bool> {
  void operator()(weval_req_arg_t* arg, bool value) {
    arg->specialize = 1;
    arg->ty = weval_req_arg_i32;
    arg->u.i32 = value ? 1 : 0;
  }
};
template <>
struct StoreArg<uint64_t> {
  void operator()(weval_req_arg_t* arg, uint64_t value) {
    arg->specialize = 1;
    arg->ty = weval_req_arg_i64;
    arg->u.i64 = value;
  }
};
template <>
struct StoreArg<float> {
  void operator()(weval_req_arg_t* arg, float value) {
    arg->specialize = 1;
    arg->ty = weval_req_arg_f32;
    arg->u.f32 = value;
  }
};
template <>
struct StoreArg<double> {
  void operator()(weval_req_arg_t* arg, double value) {
    arg->specialize = 1;
    arg->ty = weval_req_arg_f64;
    arg->u.f64 = value;
  }
};
template <typename T>
struct StoreArg<T*> {
  void operator()(weval_req_arg_t* arg, T* value) {
    static_assert(sizeof(T*) == 4, "Only 32-bit Wasm supported");
    arg->specialize = 1;
    arg->ty = weval_req_arg_i32;
    arg->u.i32 = reinterpret_cast<uint32_t>(value);
  }
};
template <typename T>
struct StoreArg<T&> {
  void operator()(weval_req_arg_t* arg, T& value) { StoreArg<T*>(arg, &value); }
};
template <typename T>
struct StoreArg<const T*> {
  void operator()(weval_req_arg_t* arg, const T* value) {
    static_assert(sizeof(const T*) == 4, "Only 32-bit Wasm supported");
    arg->specialize = 1;
    arg->ty = weval_req_arg_i32;
    arg->u.i32 = reinterpret_cast<uint32_t>(value);
  }
};

template <typename... Args>
struct StoreArgs {};

template <>
struct StoreArgs<> {
  void operator()(weval_req_arg_t* args) {}
};

template <typename T, typename... Rest>
struct StoreArgs<Specialize<T>, Rest...> {
  void operator()(weval_req_arg_t* args, Specialize<T> arg0, Rest... rest) {
    StoreArg<T>()(args, arg0.value);
    StoreArgs<Rest...>()(args + 1, rest...);
  }
};

template <typename T, typename... Rest>
struct StoreArgs<RuntimeArg<T>, Rest...> {
  void operator()(weval_req_arg_t* args, RuntimeArg<T> arg0, Rest... rest) {
    args[0].specialize = 0;
    StoreArgs<Rest...>()(args + 1, rest...);
  }
};

}  // namespace impl

template <typename Ret, typename... Args, typename... WrappedArgs>
bool weval(impl::FuncPtr<Ret, Args...>* dest,
           impl::FuncPtr<Ret, Args...> generic, WrappedArgs... args) {
  weval_free();

  weval_req_t* req = (weval_req_t*)malloc(sizeof(weval_req_t));
  if (!req) {
    return false;
  }
  uint32_t nargs = sizeof...(Args);
  weval_req_arg_t* arg_storage =
      (weval_req_arg_t*)malloc(sizeof(weval_req_arg_t) * nargs);
  if (!arg_storage) {
    return false;
  }
  impl::StoreArgs<WrappedArgs...>()(arg_storage, args...);

  req->func = (weval_func_t)generic;
  req->args = arg_storage;
  req->nargs = nargs;
  req->specialized = (weval_func_t*)dest;

  weval_request(req);

  return true;
}

}  // namespace weval

#endif  // __cplusplus
