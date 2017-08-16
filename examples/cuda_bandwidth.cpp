//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <utility>
#include <type_traits>
#include <chrono>

#include <vector>
#include <map>
#include <unordered_map>
#include <string>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

#include <cub/cub.cuh>


// CubSum functor
struct CubSum
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};


template <typename data_type>
struct data_type_traits {
  static const char* name()
  {
    using my_type = typename std::decay<data_type>::type;
    if (std::is_integral<my_type>::value) {
      switch (sizeof(my_type)) {
        case 1: return std::is_signed<my_type>::value ? "int8" : "uint8" ;
        case 2: return std::is_signed<my_type>::value ? "int16" : "uint16" ;
        case 4: return std::is_signed<my_type>::value ? "int32" : "uint32" ;
        case 8: return std::is_signed<my_type>::value ? "int64" : "uint64" ;
        case 16: return std::is_signed<my_type>::value ? "int128" : "uint128" ;
      }
    } else if (std::is_floating_point<my_type>::value) {
      switch (sizeof(my_type)) {
        case 1: return "fp8";
        case 2: return "fp16";
        case 4: return "fp32";
        case 8: return "fp64";
        case 16: return "fp128";
      }
    }
    return "n/a";
  }
};
template <>
struct data_type_traits <void> {
  static const char* name()
  {
    return "n/a";
  }
};

template <typename exec_policy>
struct exec_policy_traits {
  static const char* name()
  {
    static_assert(std::is_void<exec_policy>::value || !std::is_void<exec_policy>::value, "");
    return "unknown_exec_policy";
  }
};
template <size_t block_size, bool Async>
struct exec_policy_traits< RAJA::cuda_exec<block_size, Async> > {
  static const char* name()
  {
    static char name[1024] = "";
    if (name[0] == '\0') {
      snprintf(name, 1024, "cuda_exec<%zu,%s>", block_size, Async ? "true" : "false");
    }
    return name;
  }
};

template <typename exec_policy>
struct reduce_policy_traits {
  static const char* name()
  {
    static_assert(std::is_void<exec_policy>::value || !std::is_void<exec_policy>::value, "");
    return "unknown_reduce_policy";
  }
};
template <size_t block_size, bool Async>
struct reduce_policy_traits< RAJA::cuda_reduce<block_size, Async> > {
  static const char* name()
  {
    return Async ? "cuda_reduce<true>" : "cuda_reduce<false>";
  }
};
template <size_t block_size, bool Async>
struct reduce_policy_traits< RAJA::cuda_reduce_atomic<block_size, Async> > {
  static const char* name()
  {
    return Async ? "cuda_reduce_atomic<true>" : "cuda_reduce_atomic<false>";
  }
};


enum struct SynchronizationType : int {
  none = 0x0,
  async_full = 0x1,
  async_empty = 0x2,
  sync = 0x4
};

using SynchronizationTypeEnumType = typename std::underlying_type<SynchronizationType>::type;

inline
SynchronizationType operator| (SynchronizationType lhs, SynchronizationType rhs)
{
  return static_cast<SynchronizationType>(static_cast<SynchronizationTypeEnumType>(lhs) | static_cast<SynchronizationTypeEnumType>(rhs));
}
inline
SynchronizationType operator& (SynchronizationType lhs, SynchronizationType rhs)
{
  return static_cast<SynchronizationType>(static_cast<SynchronizationTypeEnumType>(lhs) & static_cast<SynchronizationTypeEnumType>(rhs));
}

inline
const char* SyncType_name(SynchronizationType type)
{
  switch (type) {
    case SynchronizationType::none: return "none";
    case SynchronizationType::async_full: return "async_full";
    case SynchronizationType::async_empty: return "async_empty";
    case SynchronizationType::sync: return "sync";
  }
  return "n/a";
}



bool do_print = false;
size_t num_test_repeats = 1;

const double factor = 2.0;

const char* memory_type = nullptr;


struct cudaDeviceProp devProp;

static RAJA::Index_type global_max_len = 0;



cudaEvent_t start_event;
cudaEvent_t end_event;

template <typename my_type>
struct Pair {
  my_type a[2];
  Pair() = delete;
  Pair(my_type arg) : a{arg,arg} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
};


struct PointerHolder {
  void* ptr = nullptr;
  size_t size = 0;
};

PointerHolder device_data;
PointerHolder device_other;

PointerHolder pinned_data;


const size_t num_streams = 16;

cudaStream_t streams[num_streams];
cudaEvent_t  events[num_streams];

double estimated_launch_penalty_s = 1.0e-5; // 10 us

size_t max_memory_bandwidth_Bps = 1;


template < typename T >
__global__ void device_copy(T* dst, const T* src, long long int len)
{
  long long int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < len) {
    dst[tid] = src[tid];
  }
}

__global__ void device_dummy()
{
}

RAJA::Index_type next_size(RAJA::Index_type size, RAJA::Index_type max_size)
{
  size = RAJA::Index_type(std::ceil(size * factor));

  return size;
}



void pretest_async_full(size_t num_overhead, size_t num_test_repeats, cudaStream_t stream) {
  using data_type = long;
  size_t copy_len = num_overhead*device_other.size/(2*num_streams)/sizeof(data_type);
  size_t gridSize = (copy_len + 128-1)/128;
  for(size_t i = 0; i < num_test_repeats; ++i) {
    device_copy<<<gridSize,128,0,stream>>>(static_cast<data_type*>(device_other.ptr),
                                      static_cast<data_type*>(device_other.ptr)+copy_len,
                                      copy_len);
  }
}

void pretest_async_empty(cudaStream_t stream) {
  device_dummy<<<1,1,0,stream>>>();
  device_dummy<<<1,1,0,stream>>>();
}

void pretest_sync(cudaStream_t stream) {
  cudaErrchk(cudaStreamSynchronize(stream));
}


template <typename my_type, size_t size>
class Array;
template <typename my_type>
class Array<my_type, 1ul> {
  static constexpr const size_t size = 1ul;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{my_type{val}} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(RAJA::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 2ul> {
  static constexpr const size_t size = 2ul;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{my_type{val},my_type{val}} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(RAJA::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 4ul> {
  static constexpr const size_t size = 4ul;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{my_type{val},my_type{val},my_type{val},my_type{val}} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(RAJA::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 8ul> {
  static constexpr const size_t size = 8ul;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val}} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(RAJA::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 16ul> {
  static constexpr const size_t size = 16ul;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},
                          my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val}} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(RAJA::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 32ul> {
  static constexpr const size_t size = 32ul;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},
                          my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},
                          my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},
                          my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val},my_type{val}} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(RAJA::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};

template <typename data_type, size_t size>
bool set_unique_ptrs(Array<data_type*, size>& a, size_t len, PointerHolder& p) {
  size_t bytes = len*sizeof(data_type);
  void* ptr = p.ptr;
  size_t sz = p.size;
  for(size_t i = 0; i < size; ++i) {
    if (RAJA::align(128, bytes, ptr, sz)) {
      a[i] = static_cast<data_type*>(ptr);
      ptr  = static_cast<void*>(static_cast<char*>(ptr) + bytes);
      sz  -= bytes;
    } else {
      return false;
    }
  }
  return true;
}

static const size_t text_len = 1024;


const char* make_test_name_format()
{
  return "test_name-repeats<data_type[,exec_policy[,reduce_policy]]>";
}

template <size_t repeats, typename data_type>
const char* make_cuda_test_name(const char* name)
{
  static char text[text_len];
  snprintf(text, text_len, "cuda_%s-%zu<%s>",
    name, repeats, data_type_traits<data_type>::name());
  return text;
}

template <size_t repeats, typename data_type, typename exec_policy>
const char* make_raja_test_name(const char* name)
{
  static char text[text_len];
  snprintf(text, text_len, "raja_%s-%zu<%s,%s>",
    name, repeats, data_type_traits<data_type>::name(), exec_policy_traits<exec_policy>::name());
  return text;
}

template <size_t repeats, typename data_type, typename exec_policy, typename reduce_policy>
const char* make_raja_reduce_test_name(const char* name)
{
  static char text[text_len];
  snprintf(text, text_len, "raja_%s_reduce-%zu<%s,%s,%s>",
    name, repeats, data_type_traits<data_type>::name(), exec_policy_traits<exec_policy>::name(), reduce_policy_traits<reduce_policy>::name());
  return text;
}

template <size_t repeats, typename data_type>
const char* make_cub_reduce_test_name(const char* name)
{
  static char text[text_len];
  snprintf(text, text_len, "cub_%s_reduce-%zu<%s>",
    name, repeats, data_type_traits<data_type>::name());
  return text;
}


// host time, device time
using time_return_type = std::tuple<double, double>;
// test ran successful, {host time, device time}
using test_return_type = std::tuple<bool, time_return_type>;


template <typename Test>
time_return_type time_test_helper(Test const& test, SynchronizationType sync_type, size_t num_overhead, size_t num_test_repeats)
{
  double host_time = 0.0, dev_time = 0.0;


  for(size_t i = 0; i < num_test_repeats; ++i)
  {

    switch (sync_type) {
      case SynchronizationType::none: break;
      case SynchronizationType::async_full: pretest_async_full(num_overhead, num_test_repeats, cudaStream_t{0}); break;
      case SynchronizationType::async_empty: pretest_async_empty(cudaStream_t{0}); break;
      case SynchronizationType::sync: pretest_sync(cudaStream_t{0}); break;
      default: assert(0); break;
    }

    const size_t num_extra_streams = 4;
    const size_t num_extra_tests = 6;

    cudaErrchk(cudaEventRecord(start_event, cudaStream_t{0}));
    auto start = std::chrono::high_resolution_clock::now();

    for(size_t s = 0; s < num_extra_streams; ++s) {
      cudaErrchk(cudaStreamWaitEvent(streams[s], start_event, 0));
    }

    for(size_t s = 0; s < num_extra_streams; ++s) {
      pretest_async_full(num_overhead, 1, streams[s]);
    }
    for(size_t s = 0; s < num_extra_tests; ++s) {
      test();
    }
    for(size_t s = 0; s < num_extra_streams; ++s) {
      pretest_async_full(num_overhead, 1, streams[s]);
    }

    for(size_t s = 0; s < num_extra_streams; ++s) {
      cudaErrchk(cudaEventRecord(events[s], streams[s]));
      cudaErrchk(cudaStreamWaitEvent(cudaStream_t{0}, events[s], 0));
    }

    auto stop = std::chrono::high_resolution_clock::now();
    cudaErrchk(cudaEventRecord(end_event));
    std::chrono::duration<double> diff = stop-start;
    host_time += (diff.count()*1e6) / double(num_extra_tests);
    float fms;
    cudaErrchk(cudaEventSynchronize(end_event));
    cudaErrchk(cudaEventElapsedTime(&fms, start_event, end_event));
    dev_time += (fms*1e3) / double(num_extra_tests);

  }

  return time_return_type{host_time / num_test_repeats,
                          dev_time  / num_test_repeats};
}

template <typename Test>
time_return_type time_test(Test const& test, SynchronizationType sync_type, size_t num_overhead, size_t num_test_repeats)
{
  time_return_type presync_times = time_test_helper([&](){
    // do nothing
  }, sync_type, num_overhead, num_test_repeats);
  time_return_type test_times = time_test_helper([&](){
    test();
  }, sync_type, num_overhead, num_test_repeats);
  return time_return_type{std::get<0>(test_times) - std::get<0>(presync_times),
                          std::get<1>(test_times) - std::get<1>(presync_times)};
}


template <size_t repeats, typename data_type>
struct cudaMemset_test {
  const char* name()
  {
    return make_cuda_test_name<repeats, data_type>("Memset");
  }
  size_t num_overhead()
  {
    return repeats;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      dp.for_each([=](data_type* ptr, size_t) {
        cudaErrchk(cudaMemsetAsync(ptr, 0, len*sizeof(*ptr), cudaStream_t{0}));
      });
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
};

template <size_t repeats, typename data_type>
struct cudaStreamsMemset_test {
  const char* name()
  {
    return make_cuda_test_name<repeats, data_type>("StreamsMemset");
  }
  size_t num_overhead()
  {
    return repeats;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      for (size_t i = 0; i < repeats; ++i) {
        cudaErrchk(cudaStreamWaitEvent(streams[i], start_event, 0));
        cudaErrchk(cudaMemsetAsync(dp[0], 0, len*sizeof(*dp[0]), streams[i]));
        cudaErrchk(cudaEventRecord(events[i], streams[i]));
        cudaErrchk(cudaStreamWaitEvent(cudaStream_t{0}, events[i], 0));
      }
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
};

template <size_t repeats, typename data_type>
struct cudaMemcpy_test {
  const char* name()
  {
    return make_cuda_test_name<repeats, data_type>("Memcpy");
  }
  size_t num_overhead()
  {
    return repeats;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, 2ul*repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      Array<Pair<data_type*>, repeats> dp_pair{nullptr};
      for (size_t i = 0; i < repeats; ++i) {
        dp_pair[i][0] = dp[2*i];
        dp_pair[i][1] = dp[2*i+1];
      }
      dp_pair.for_each([=](Pair<data_type*> ptr_pair, size_t) {
        cudaErrchk(cudaMemcpyAsync(ptr_pair[0], ptr_pair[1], len*sizeof(*ptr_pair[1]), cudaMemcpyDefault, cudaStream_t{0}));
      });
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
};

template <size_t repeats, typename data_type>
struct cudaStreamsMemcpy_test {
  const char* name()
  {
    return make_cuda_test_name<repeats, data_type>("StreamsMemcpy");
  }
  size_t num_overhead()
  {
    return repeats;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, 2ul*repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      for (size_t i = 0; i < repeats; ++i) {
        data_type* dp0 = dp[2*i];
        data_type* dp1 = dp[2*i+1];
        cudaErrchk(cudaStreamWaitEvent(streams[i], start_event, 0));
        cudaErrchk(cudaMemcpyAsync(dp0, dp1, len*sizeof(*dp1), cudaMemcpyDefault, streams[i]));
        cudaErrchk(cudaEventRecord(events[i], streams[i]));
        cudaErrchk(cudaStreamWaitEvent(cudaStream_t{0}, events[i], 0));
      }
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
};

template <size_t repeats, typename data_type, typename exec_policy>
struct rajaMemset_test {
  const char* name()
  {
    return make_raja_test_name<repeats, data_type, exec_policy>("Memset");
  }
  size_t num_overhead()
  {
    return 1;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      my_test(len, dp);
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
// private:
  void my_test(RAJA::Index_type& len, Array<data_type*, repeats>& dp)
  {
    dp.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(data_type* ptr, size_t, RAJA::Index_type idx) {
      ptr[idx] = data_type{0};
    });
  }
};

template <size_t repeats, typename data_type, typename exec_policy>
struct rajaStreamsMemset_test {
  const char* name()
  {
    return make_raja_test_name<repeats, data_type, exec_policy>("StreamsMemset");
  }
  size_t num_overhead()
  {
    return repeats;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      for (size_t i = 0; i < repeats; ++i) {
        data_type* ptr = dp[i];
        cudaErrchk(cudaStreamWaitEvent(streams[i], start_event, 0));
        RAJA::cuda::stream() = streams[i];
        my_test(len, ptr);
        cudaErrchk(cudaEventRecord(events[i], streams[i]));
        cudaErrchk(cudaStreamWaitEvent(cudaStream_t{0}, events[i], 0));
      }
      RAJA::cuda::stream() = cudaStream_t{0};
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
// private:
  void my_test(RAJA::Index_type& len, data_type*& ptr)
  {
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      ptr[idx] = data_type{0};
    });
  }
};

template <size_t repeats, typename data_type, typename exec_policy>
struct rajaMemcpy_test {
  const char* name()
  {
    return make_raja_test_name<repeats, data_type, exec_policy>("Memcpy");
  }
  size_t num_overhead()
  {
    return 1;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, 2ul*repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      Array<Pair<data_type*>, repeats> dp_pair{nullptr};
      for (size_t i = 0; i < repeats; ++i) {
        dp_pair[i][0] = dp[2*i];
        dp_pair[i][1] = dp[2*i+1];
      }
      my_test(len, dp_pair);
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
// private:
  void my_test(RAJA::Index_type& len, Array<Pair<data_type*>, repeats>& dp_pair)
  {
    dp_pair.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Pair<data_type*> ptr_pair, size_t, RAJA::Index_type idx) {
      ptr_pair[0][idx] = ptr_pair[1][idx];
    });
  }
};

template <size_t repeats, typename data_type, typename exec_policy>
struct rajaStreamsMemcpy_test {
  const char* name()
  {
    return make_raja_test_name<repeats, data_type, exec_policy>("StreamsMemcpy");
  }
  size_t num_overhead()
  {
    return repeats;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, 2ul*repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      for (size_t i = 0; i < repeats; ++i) {
        data_type* dp0 = dp[2*i];
        data_type* dp1 = dp[2*i+1];
        cudaErrchk(cudaStreamWaitEvent(streams[i], start_event, 0));
        RAJA::cuda::stream() = streams[i];
        my_test(len, dp0, dp1);
        cudaErrchk(cudaEventRecord(events[i], streams[i]));
        cudaErrchk(cudaStreamWaitEvent(cudaStream_t{0}, events[i], 0));
      }
      RAJA::cuda::stream() = cudaStream_t{0};
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
// private:
  void my_test(RAJA::Index_type& len, data_type*& dp0, data_type*& dp1)
  {
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(RAJA::Index_type idx) {
      dp0[idx] = dp1[idx];
    });
  }
};

#if 0
template <size_t repeats, typename data_type, typename exec_policy, typename reduce_policy>
struct rajasumpi_test {
  const char* name()
  {
    return make_raja_reduce_test_name<repeats, data_type, exec_policy, reduce_policy>("pi_sum");
  }
  size_t num_overhead()
  {
    return 1;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    using Reducer = RAJA::ReduceSum<reduce_policy, data_type>;
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      Array<Reducer, repeats> rd{static_cast<data_type>(0)};
      my_test(len, rd)
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
// private:
  template <typename Reducer>
  void my_test(RAJA::Index_type& len, Array<Reducer, repeats>& rd)
  {
    rd.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t, RAJA::Index_type idx) {
      data_type x = (data_type(idx) + data_type(0.5)) / data_type(len);
      r.reduce(4.0 / (1.0 + x * x));
    });
  }
};
#endif

template <size_t repeats, typename data_type, typename exec_policy, typename reduce_policy>
struct rajasumindex_test {
  const char* name()
  {
    return make_raja_reduce_test_name<repeats, data_type, exec_policy, reduce_policy>("idx_sum");
  }
  size_t num_overhead()
  {
    return 1;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    using Reducer = RAJA::ReduceSum<reduce_policy, data_type>;
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      Array<Reducer, repeats> rd{static_cast<data_type>(0)};
      my_test(len, rd);
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
// private:
  template <typename Reducer>
  void my_test(RAJA::Index_type& len, Array<Reducer, repeats>& rd)
  {
    rd.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t, RAJA::Index_type idx) {
      r.reduce(data_type(idx));
    });
  }
};

template <size_t repeats, typename data_type, typename exec_policy, typename reduce_policy>
struct rajasumreduce_test {
  const char* name()
  {
    return make_raja_reduce_test_name<repeats, data_type, exec_policy, reduce_policy>("array_sum");
  }
  size_t num_overhead()
  {
    return 1;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    using Reducer = RAJA::ReduceSum<reduce_policy, data_type>;
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      Array<Reducer, repeats> rd{static_cast<data_type>(0)};
      my_test(len, rd, dp);
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
// private:
  template <typename Reducer>
  void my_test(RAJA::Index_type& len, Array<Reducer, repeats>& rd, Array<data_type*, repeats>& dp)
  {
    rd.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t i, RAJA::Index_type idx) {
      r.reduce(dp[i][idx]);
    });
  }
};

template <typename Reducer, typename data_type>
inline void do_cub_reduce(data_type* ptr_in, data_type* ptr_out, size_t len)
{
  char     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, len, Reducer{}, data_type{0}, cudaStream_t{0});
  d_temp_storage = RAJA::cuda::device_mempool_type::getInstance().malloc<char>(temp_storage_bytes);
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, len, Reducer{}, data_type{0}, cudaStream_t{0});
  RAJA::cuda::device_mempool_type::getInstance().free(d_temp_storage);
}


template <size_t repeats, typename data_type>
struct cubsumreduce_test {
  const char* name()
  {
    return make_cub_reduce_test_name<repeats, data_type>("array_sum");
  }
  size_t num_overhead()
  {
    return repeats;
  }
  test_return_type operator()(RAJA::Index_type len, SynchronizationType sync_type, size_t num_test_repeats) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);

    auto single_test = [&](){
      data_type* pinned_buf = static_cast<data_type*>(pinned_data.ptr);
      dp.for_each([=](data_type* ptr, size_t i) {
        do_cub_reduce<CubSum>(ptr, pinned_buf+i, len);
      });
    };

    if (can_test) {
      return test_return_type {true, time_test(single_test, sync_type, num_overhead(), num_test_repeats)};
    } else {
      return test_return_type {false, time_return_type{0.0, 0.0}};
    }
  }
};

static bool first_print = true;

template <typename Test>
void run_single_test_pass(SynchronizationType sync_type, const char* gs_mode, RAJA::Index_type max_len)
{
  Test atest;

  if (do_print) {
    if (first_print) {
      printf("%s{gridstride}(memory)[presynchronization]", make_test_name_format());

      for (RAJA::Index_type size = 32; size <= global_max_len; size = next_size(size, global_max_len)) {
        printf("\t%li", size);
      }
      for (RAJA::Index_type size = 32; size <= global_max_len; size = next_size(size, global_max_len)) {
        printf("\t%li", size);
      }
      printf("\n"); fflush(stdout);
      first_print = false;
    }
    printf("%s{%s}(%s)[%s]",
            atest.name(), gs_mode, memory_type, SyncType_name(sync_type));
  }

  std::map<RAJA::Index_type, test_return_type> size_time_map;

  for (RAJA::Index_type size = 32; size <= global_max_len; size = next_size(size, global_max_len)) {

    bool emplaced = size_time_map.emplace( size, test_return_type{false, time_return_type{0.0, 0.0}} ).second;
    assert(emplaced);

    // run each size of the test once to warm up
    atest(size, sync_type, 1);

  }

  for (RAJA::Index_type size = 32; size <= global_max_len; size = next_size(size, global_max_len)) {

    auto iter = size_time_map.find(size);

    // run smaller tests more times to reduce noise
    size_t min_bytes_moved = sizeof(double) * size;
    double min_runtime = min_bytes_moved / double(max_memory_bandwidth_Bps);

    double repeat_ratio = estimated_launch_penalty_s * 8.0 / min_runtime;
    // clamp repeat_ratio between 1 and 2
    repeat_ratio = std::min(repeat_ratio, 2.0);
    repeat_ratio = std::max(repeat_ratio, 1.0);

    size_t test_repeats = num_test_repeats * repeat_ratio;

    // run test
    iter->second = atest(size, sync_type, test_repeats);

  }

  if (do_print) {

    for(auto iter = size_time_map.begin(); iter != size_time_map.end(); ++iter) {
      printf("\t");
      if (std::get<0>(iter->second)) {
        printf("%e", std::get<0>(std::get<1>(iter->second)) );
      }
    }
    for(auto iter = size_time_map.begin(); iter != size_time_map.end(); ++iter) {
      printf("\t");
      if (std::get<0>(iter->second)) {
        printf("%e", std::get<1>(std::get<1>(iter->second)) );
      }
    }

    printf("\n"); fflush(stdout);
  }

}

template <typename Test>
void run_single_raja_test_pass(SynchronizationType sync_type, RAJA::Index_type max_len)
{

  for (int i = 0; i < 2; ++i) {

    const char* gs_mode = nullptr;

    if (i == 1) {
    //   RAJA::cuda::getGridStrideMode() = RAJA::cuda::GridStrideMode::static_size;
    //   gs_mode = "static";
    // } else if (i == 2) {
      RAJA::cuda::getGridStrideMode() = RAJA::cuda::GridStrideMode::occupancy_size;
      gs_mode = "occupancy";
    } else {
      RAJA::cuda::getGridStrideMode() = RAJA::cuda::GridStrideMode::disabled;
      gs_mode = "disabled";
    }

    run_single_test_pass<Test>(sync_type, gs_mode, max_len);

  }

}

template <typename data_type, size_t repeats>
void run_memset_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
}
template <typename data_type, size_t repeats, size_t block_size, size_t... block_sizes>
void run_memset_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
  using exec_policy = RAJA::cuda_exec<block_size, true>;
  run_single_raja_test_pass< rajaMemset_test<repeats, data_type, exec_policy> >(sync_type, max_len);
  // run_single_raja_test_pass< rajaStreamsMemset_test<repeats, data_type, exec_policy> >(sync_type, max_len);

  run_memset_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}
template <typename data_type, size_t repeats, size_t... block_sizes>
void run_memset_test_pass(SynchronizationType sync_type)
{
  RAJA::Index_type max_len = device_data.size/sizeof(data_type);

  run_single_test_pass< cudaMemset_test<repeats, data_type> >(sync_type, "n/a", max_len);
  // run_single_test_pass< cudaStreamsMemset_test<repeats, data_type> >(sync_type, "n/a", max_len);

  run_memset_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}


template <typename data_type, size_t repeats>
void run_memcpy_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
}
template <typename data_type, size_t repeats, size_t block_size, size_t... block_sizes>
void run_memcpy_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
  using exec_policy = RAJA::cuda_exec<block_size, true>;
  run_single_raja_test_pass< rajaMemcpy_test<repeats, data_type, exec_policy> >(sync_type, max_len);
  // run_single_raja_test_pass< rajaStreamsMemcpy_test<repeats, data_type, exec_policy> >(sync_type, max_len);

  run_memcpy_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}
template <typename data_type, size_t repeats, size_t... block_sizes>
void run_memcpy_test_pass(SynchronizationType sync_type)
{
  RAJA::Index_type max_len = device_data.size/sizeof(data_type);

  run_single_test_pass< cudaMemcpy_test<repeats, data_type> >(sync_type, "n/a", max_len);
  // run_single_test_pass< cudaStreamsMemcpy_test<repeats, data_type> >(sync_type, "n/a", max_len);

  run_memcpy_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}


template <typename data_type, size_t repeats>
void run_sumarray_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
}
template <typename data_type, size_t repeats, size_t block_size, size_t... block_sizes>
void run_sumarray_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
  using exec_policy          = RAJA::cuda_exec         <block_size, true>;
  using reduce_policy        = RAJA::cuda_reduce       <block_size, true>;
  using reduce_atomic_policy = RAJA::cuda_reduce_atomic<block_size, true>;
  run_single_raja_test_pass< rajasumreduce_test<repeats, data_type, exec_policy, reduce_policy       > >(sync_type, max_len);
  run_single_raja_test_pass< rajasumreduce_test<repeats, data_type, exec_policy, reduce_atomic_policy> >(sync_type, max_len);

  run_sumarray_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}
template <typename data_type, size_t repeats, size_t... block_sizes>
void run_sumarray_test_pass(SynchronizationType sync_type)
{
  RAJA::Index_type max_len = device_data.size/sizeof(data_type);

  run_single_test_pass< cubsumreduce_test<repeats, data_type> >(sync_type, "n/a", max_len);

  run_sumarray_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}


template <typename data_type, size_t repeats>
void run_sumindex_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
}
template <typename data_type, size_t repeats, size_t block_size, size_t... block_sizes>
void run_sumindex_test_pass_helper(SynchronizationType sync_type, RAJA::Index_type max_len)
{
  using exec_policy          = RAJA::cuda_exec         <block_size, true>;
  using reduce_policy        = RAJA::cuda_reduce       <block_size, true>;
  using reduce_atomic_policy = RAJA::cuda_reduce_atomic<block_size, true>;
  run_single_raja_test_pass< rajasumindex_test<repeats, data_type, exec_policy, reduce_policy       > >(sync_type, max_len);
  run_single_raja_test_pass< rajasumindex_test<repeats, data_type, exec_policy, reduce_atomic_policy> >(sync_type, max_len);

  run_sumindex_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}
template <typename data_type, size_t repeats, size_t... block_sizes>
void run_sumindex_test_pass(SynchronizationType sync_type)
{
  RAJA::Index_type max_len = device_data.size/sizeof(data_type);

  run_sumindex_test_pass_helper<data_type, repeats, block_sizes...>(sync_type, max_len);
}


template <typename data_type>
void run_repeated_memset_tests(SynchronizationType sync_type)
{
}
template <typename data_type, size_t repeat, size_t... repeats>
void run_repeated_memset_tests(SynchronizationType sync_type)
{
  if (SynchronizationType::async_full == (sync_type & SynchronizationType::async_full)) {
    run_memset_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_full);
  }
  if (SynchronizationType::async_empty == (sync_type & SynchronizationType::async_empty)) {
    run_memset_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_empty);
  }
  if (SynchronizationType::sync == (sync_type & SynchronizationType::sync)) {
    run_memset_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::sync);
  }

  run_repeated_memset_tests<data_type, repeats...>(sync_type);
}

template <typename data_type>
void run_repeated_memcpy_tests(SynchronizationType sync_type)
{
}
template <typename data_type, size_t repeat, size_t... repeats>
void run_repeated_memcpy_tests(SynchronizationType sync_type)
{
  if (SynchronizationType::async_full == (sync_type & SynchronizationType::async_full)) {
    run_memcpy_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_full);
  }
  if (SynchronizationType::async_empty == (sync_type & SynchronizationType::async_empty)) {
    run_memcpy_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_empty);
  }
  if (SynchronizationType::sync == (sync_type & SynchronizationType::sync)) {
    run_memcpy_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::sync);
  }

  run_repeated_memcpy_tests<data_type, repeats...>(sync_type);
}

template <typename data_type>
void run_repeated_sumarray_tests(SynchronizationType sync_type)
{
}
template <typename data_type, size_t repeat, size_t... repeats>
void run_repeated_sumarray_tests(SynchronizationType sync_type)
{
  if (SynchronizationType::async_full == (sync_type & SynchronizationType::async_full)) {
    run_sumarray_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_full);
  }
  if (SynchronizationType::async_empty == (sync_type & SynchronizationType::async_empty)) {
    run_sumarray_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_empty);
  }
  if (SynchronizationType::sync == (sync_type & SynchronizationType::sync)) {
    run_sumarray_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::sync);
  }

  run_repeated_sumarray_tests<data_type, repeats...>(sync_type);
}

template <typename data_type>
void run_repeated_sumindex_tests(SynchronizationType sync_type)
{
}
template <typename data_type, size_t repeat, size_t... repeats>
void run_repeated_sumindex_tests(SynchronizationType sync_type)
{
  if (SynchronizationType::async_full == (sync_type & SynchronizationType::async_full)) {
    run_sumindex_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_full);
  }
  if (SynchronizationType::async_empty == (sync_type & SynchronizationType::async_empty)) {
    run_sumindex_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::async_empty);
  }
  if (SynchronizationType::sync == (sync_type & SynchronizationType::sync)) {
    run_sumindex_test_pass<data_type, repeat, 128, 256, 512>(SynchronizationType::sync);
  }

  run_repeated_sumindex_tests<data_type, repeats...>(sync_type);
}

void run_all_tests()
{
  // run real tests
  do_print = true;
  num_test_repeats = 10;

  SynchronizationType sync_type = SynchronizationType::async_full;
  run_repeated_memset_tests<char,  1, 2, 4, 8, 16>(sync_type | SynchronizationType::async_empty | SynchronizationType::sync);
  run_repeated_memset_tests<short, 1, 2, 4, 8, 16>(sync_type);
  run_repeated_memset_tests<int,   1, 2, 4, 8, 16>(sync_type);
  run_repeated_memset_tests<long,  1, 2, 4, 8, 16>(sync_type);

  run_repeated_memcpy_tests<char,  1, 2, 4, 8, 16>(sync_type);
  run_repeated_memcpy_tests<short, 1, 2, 4, 8, 16>(sync_type);
  run_repeated_memcpy_tests<int,   1, 2, 4, 8, 16>(sync_type);
  run_repeated_memcpy_tests<long,  1, 2, 4, 8, 16>(sync_type);

  run_repeated_sumarray_tests<int,    1, 2, 4, 8, 16>(sync_type);
  run_repeated_sumarray_tests<double, 1, 2, 4, 8, 16>(sync_type);

  run_repeated_sumindex_tests<int,    1, 2, 4, 8, 16>(sync_type);
  run_repeated_sumindex_tests<double, 1, 2, 4, 8, 16>(sync_type);
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{

  unsigned compiler_v = __CUDACC_VER__;
  int runtime_v;
  int driver_v;
  cudaErrchk(cudaRuntimeGetVersion(&runtime_v));
  cudaErrchk(cudaDriverGetVersion(&driver_v));

  int device;
  cudaErrchk(cudaGetDevice(&device));

  cudaErrchk(cudaGetDeviceProperties(&devProp, device));

  estimated_launch_penalty_s = 1.0e-5; // 10 us

  size_t memory_clock_hz = size_t(1000)*size_t(devProp.memoryClockRate);
  size_t memory_bus_width_B = size_t(devProp.memoryBusWidth)/size_t(8);

  max_memory_bandwidth_Bps = 2*memory_clock_hz*memory_bus_width_B;

  size_t data_size = size_t(devProp.totalGlobalMem)/size_t(2);
  global_max_len = data_size;

  size_t max_concurrent_threads = size_t(devProp.maxThreadsPerMultiProcessor) * size_t(devProp.multiProcessorCount);

  size_t other_size = static_cast<size_t>(std::ceil(4*estimated_launch_penalty_s * double(max_memory_bandwidth_Bps) ));
  other_size = std::max(other_size, 4*size_t(devProp.l2CacheSize));
  if (other_size % (2*8*max_concurrent_threads) != 0) other_size += (2*8*max_concurrent_threads) - other_size % (2*8*max_concurrent_threads);
  other_size *= num_streams;

  size_t pinned_size = 16*sizeof(double);

  printf("CUDA version: compiler %u, runtime %d, driver %d\n", compiler_v, runtime_v, driver_v);
  printf("GPU: %s clock %f GHz, Memory Bandwidth %f GiBps\n", devProp.name, devProp.clockRate/(1000.0*1000.0), max_memory_bandwidth_Bps/(1024.0*1024.0*1024.0));
  printf("Test allocations: data %f MiB, other %f MiB\n", data_size/(1024.0*1024.0), other_size/(1024.0*1024.0));
  fflush(stdout);

  {
    memory_type = "dev";

    // allocate resources
    cudaErrchk(cudaMalloc(&device_data.ptr, data_size)); device_data.size = data_size;
    cudaErrchk(cudaMalloc(&device_other.ptr, other_size)); device_other.size = other_size;
    cudaErrchk(cudaHostAlloc(&pinned_data.ptr, pinned_size, cudaHostAllocDefault)); pinned_data.size = pinned_size;

    cudaErrchk(cudaMemset(device_data.ptr, 0, data_size));
    cudaErrchk(cudaMemset(device_other.ptr, 0, other_size));

    cudaErrchk(cudaEventCreateWithFlags(&start_event, cudaEventDefault));
    cudaErrchk(cudaEventCreateWithFlags(&end_event, cudaEventDefault));

    for (size_t i = 0; i < num_streams; ++i) {
      cudaErrchk(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
      cudaErrchk(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    }

    run_all_tests();

    for (size_t i = 0; i < num_streams; ++i) {
      cudaErrchk(cudaEventDestroy(events[i]));
      cudaErrchk(cudaStreamDestroy(streams[i]));
    }

    // free resources
    cudaErrchk(cudaEventDestroy(end_event));
    cudaErrchk(cudaEventDestroy(start_event));

    cudaErrchk(cudaFreeHost(pinned_data.ptr)); pinned_data.ptr = nullptr; pinned_data.size = 0;

    cudaErrchk(cudaFree(device_data.ptr));  device_data.ptr = nullptr;  device_data.size = 0;
    cudaErrchk(cudaFree(device_other.ptr)); device_other.ptr = nullptr; device_other.size = 0;

    memory_type = nullptr;
  }

  // {
  //   memory_type = "um";

  //   // allocate resources (um)
  //   cudaErrchk(cudaMallocManaged(&device_data.ptr, data_size)); device_data.size = data_size;
  //   cudaErrchk(cudaMalloc(&device_other.ptr, other_size)); device_other.size = other_size;
  //   cudaErrchk(cudaHostAlloc(&pinned_data.ptr, pinned_size, cudaHostAllocDefault)); pinned_data.size = pinned_size;

  //   cudaErrchk(cudaMemset(device_data.ptr, 0, data_size));
  //   cudaErrchk(cudaMemset(device_other.ptr, 0, other_size));

  //   cudaErrchk(cudaEventCreateWithFlags(&start_event, cudaEventDefault));
  //   cudaErrchk(cudaEventCreateWithFlags(&end_event, cudaEventDefault));

  //   for (size_t i = 0; i < num_streams; ++i) {
  //     cudaErrchk(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
  //     cudaErrchk(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
  //   }

  //   run_all_tests();

  //   for (size_t i = 0; i < num_streams; ++i) {
  //     cudaErrchk(cudaEventDestroy(events[i]));
  //     cudaErrchk(cudaStreamDestroy(streams[i]));
  //   }

  //   // free resources
  //   cudaErrchk(cudaEventDestroy(end_event));
  //   cudaErrchk(cudaEventDestroy(start_event));

  //   cudaErrchk(cudaFreeHost(pinned_data.ptr)); pinned_data.ptr = nullptr; pinned_data.size = 0;

  //   cudaErrchk(cudaFree(device_data.ptr));  device_data.ptr = nullptr;  device_data.size = 0;
  //   cudaErrchk(cudaFree(device_other.ptr)); device_other.ptr = nullptr; device_other.size = 0;

  //   memory_type = nullptr;
  // }

  return 0;
}
