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
  async_full,
  async_empty,
  sync
};

inline
const char* SyncType_name(SynchronizationType type)
{
  switch (type) {
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

SynchronizationType sync_type = SynchronizationType::async_empty;


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



__global__ void device_copy(char* dst, const char* src, long long int bytes)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < bytes) {
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


template <size_t repeats, typename data_type>
struct cudaMemset_test {
  const char* name()
  {
    return make_cuda_test_name<repeats, data_type>("Memset");
  }
  bool operator()(RAJA::Index_type len) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      dp.for_each([=](data_type* ptr, size_t) {
        cudaErrchk(cudaMemsetAsync(ptr, 0, len*sizeof(*ptr), 0));
      });
    }
    return can_test;
  }
};

template <size_t repeats, typename data_type>
struct cudaMemcpy_test {
  const char* name()
  {
    return make_cuda_test_name<repeats, data_type>("Memcpy");
  }
  bool operator()(RAJA::Index_type len) {
    Array<data_type*, 2ul*repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      Array<Pair<data_type*>, repeats> dp_pair{nullptr};
      for (size_t i = 0; i < repeats; ++i) {
        dp_pair[i][0] = dp[2*i];
        dp_pair[i][1] = dp[2*i+1];
      }
      dp_pair.for_each([=](Pair<data_type*> ptr_pair, size_t) {
        cudaErrchk(cudaMemcpyAsync(ptr_pair[0], ptr_pair[1], len*sizeof(*ptr_pair[1]), cudaMemcpyDefault, 0));
      });
    }
    return can_test;
  }
};

template <size_t repeats, typename data_type, typename exec_policy>
struct rajaMemset_test {
  const char* name()
  {
    return make_raja_test_name<repeats, data_type, exec_policy>("Memset");
  }
  bool operator()(RAJA::Index_type len) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      dp.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(data_type* ptr, size_t, RAJA::Index_type idx) {
        ptr[idx] = 0;
      });
    }
    return can_test;
  }
};

template <size_t repeats, typename data_type, typename exec_policy>
struct rajaMemcpy_test {
  const char* name()
  {
    return make_raja_test_name<repeats, data_type, exec_policy>("Memcpy");
  }
  bool operator()(RAJA::Index_type len) {
    Array<data_type*, 2ul*repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      Array<Pair<data_type*>, repeats> dp_pair{nullptr};
      for (size_t i = 0; i < repeats; ++i) {
        dp_pair[i][0] = dp[2*i];
        dp_pair[i][1] = dp[2*i+1];
      }
      dp_pair.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Pair<data_type*> ptr_pair, size_t, RAJA::Index_type idx) {
        ptr_pair[0][idx] = ptr_pair[1][idx];
      });
    }
    return can_test;
  }
};

#if 0
template <size_t repeats, typename data_type, typename exec_policy, typename reduce_policy>
struct rajasumpi_test {
  const char* name()
  {
    return make_raja_reduce_test_name<repeats, data_type, exec_policy, reduce_policy>("pi_sum");
  }
  bool operator()(RAJA::Index_type len) {
    using Reducer = RAJA::ReduceSum<reduce_policy, data_type>;
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      Array<Reducer, repeats> rd{static_cast<data_type>(0)};
      rd.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t, RAJA::Index_type idx) {
        data_type x = (data_type(idx) + data_type(0.5)) / data_type(len);
        r.reduce(4.0 / (1.0 + x * x));
      });
    }
    return can_test;
  }
};
#endif

template <size_t repeats, typename data_type, typename exec_policy, typename reduce_policy>
struct rajasumindex_test {
  const char* name()
  {
    return make_raja_reduce_test_name<repeats, data_type, exec_policy, reduce_policy>("idx_sum");
  }
  bool operator()(RAJA::Index_type len) {
    using Reducer = RAJA::ReduceSum<reduce_policy, data_type>;
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      Array<Reducer, repeats> rd{static_cast<data_type>(0)};
      rd.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t, RAJA::Index_type idx) {
        r.reduce(data_type(idx));
      });
    }
    return can_test;
  }
};

template <size_t repeats, typename data_type, typename exec_policy, typename reduce_policy>
struct rajasumreduce_test {
  const char* name()
  {
    return make_raja_reduce_test_name<repeats, data_type, exec_policy, reduce_policy>("array_sum");
  }
  bool operator()(RAJA::Index_type len) {
    using Reducer = RAJA::ReduceSum<reduce_policy, data_type>;
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      Array<Reducer, repeats> rd{static_cast<data_type>(0)};
      rd.template forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t i, RAJA::Index_type idx) {
        r.reduce(dp[i][idx]);
      });
    }
    return can_test;
  }
};

template <typename Reducer, typename data_type>
inline void do_cub_reduce(data_type* ptr_in, data_type* ptr_out, size_t len)
{
  char     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, len, Reducer{}, 0);
  d_temp_storage = RAJA::cuda::device_mempool_type::getInstance().malloc<char>(temp_storage_bytes);
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, len, Reducer{}, 0);
  RAJA::cuda::device_mempool_type::getInstance().free(d_temp_storage);
}

template <size_t repeats, typename data_type>
struct cubsumreduce_test {
  const char* name()
  {
    return make_cub_reduce_test_name<repeats, data_type>("array_sum");
  }
  bool operator()(RAJA::Index_type len) {
    Array<data_type*, repeats> dp{nullptr};
    bool can_test = set_unique_ptrs(dp, len, device_data);
    if (can_test) {
      data_type* pinned_buf = static_cast<data_type*>(pinned_data.ptr);
      dp.for_each([=](data_type* ptr, size_t i) {
        do_cub_reduce<CubSum>(ptr, pinned_buf+i, len);
      });
    }
    return can_test;
  }
};

static bool first_print = true;


void pretest_async_full() {
  size_t copy_size = device_other.size/2;
  size_t gridSize = (copy_size + 128-1)/128;
  device_copy<<<gridSize,128,0,0>>>(static_cast<char*>(device_other.ptr),
                                    static_cast<char*>(device_other.ptr)+copy_size,
                                    copy_size);
}

void pretest_async_empty() {
  device_dummy<<<1,1,0,0>>>();
  device_dummy<<<1,1,0,0>>>();
}

void pretest_sync() {
  cudaErrchk(cudaStreamSynchronize(0));
}

template <typename Test>
void run_single_test_pass(const char* block_size, const char* gs_mode, RAJA::Index_type max_len)
{
  Test atest;

  if (do_print) {
    if (first_print) {
      printf("%s{gridstride}(memory)[presynchronization]", make_test_name_format());

      RAJA::Index_type size = 32;
      while ( size <= global_max_len ) {
        printf("\t%li", size);
        size = next_size(size, global_max_len);
      }
      printf("\n"); fflush(stdout);
      first_print = false;
    }
    printf("%s{%s}(%s)[%s]",
            atest.name(), gs_mode, memory_type, SyncType_name(sync_type));
  }

  std::map<RAJA::Index_type, Pair<double>> size_time_map;

  {
    RAJA::Index_type size = 32;
    while ( size <= global_max_len ) {
      bool emplaced = size_time_map.emplace( size, 0.0 ).second;
      assert(emplaced);
      size = next_size(size, global_max_len);
    }
  }

  for (size_t repeat = 0; repeat < num_test_repeats; ++repeat) {

    RAJA::Index_type size = 32;
    while ( size <= max_len ) {

      auto iter = size_time_map.find(size);

      switch (sync_type) {
        case SynchronizationType::async_full: pretest_async_full(); break;
        case SynchronizationType::async_empty: pretest_async_empty(); break;
        case SynchronizationType::sync: pretest_sync(); break;
      }

      cudaErrchk(cudaEventRecord(start_event));
      bool successful_run = atest(size);
      cudaErrchk(cudaEventRecord(end_event));
      cudaErrchk(cudaEventSynchronize(end_event));
      float fms;
      cudaErrchk(cudaEventElapsedTime(&fms, start_event, end_event));

      if (successful_run) {
        iter->second[0] += fms*1e3;
        iter->second[1] += 1.0;
      }

      size = next_size(size, max_len);

    }

  }

  if (do_print) {

    for(auto iter = size_time_map.begin(); iter != size_time_map.end(); ++iter) {

      if (iter->second[1] > 0.0) {
        printf("\t%e", iter->second[0] / iter->second[1]);
      } else {
        printf("\t%s", "n/a");
      }
    }

    printf("\n"); fflush(stdout);
  }

}

template <typename data_type>
void run_cuda_test_pass(RAJA::Index_type max_len)
{
  run_single_test_pass< cudaMemset_test<1,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemset_test<2,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemset_test<4,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemset_test<8,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemset_test<16, data_type> >("n/a", "n/a", max_len);

  run_single_test_pass< cudaMemcpy_test<1,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemcpy_test<2,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemcpy_test<4,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemcpy_test<8,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cudaMemcpy_test<16, data_type> >("n/a", "n/a", max_len);
}

template <typename data_type>
void run_cub_test_pass(RAJA::Index_type max_len)
{
  run_single_test_pass< cubsumreduce_test<1,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cubsumreduce_test<2,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cubsumreduce_test<4,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cubsumreduce_test<8,  data_type> >("n/a", "n/a", max_len);
  run_single_test_pass< cubsumreduce_test<16, data_type> >("n/a", "n/a", max_len);
}

template <typename data_type, size_t block_size>
void run_raja_test_pass(const char* gs_mode, RAJA::Index_type max_len)
{
  char blksz[128];
  snprintf(blksz, 128, "%zu", block_size);

  typedef RAJA::cuda_reduce<block_size, true> reduce_policy;
  typedef RAJA::cuda_reduce_atomic<block_size, true> reduce_atomic_policy;
  typedef RAJA::cuda_exec<block_size, true> execute_policy;

  run_single_test_pass< rajaMemset_test<1,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemset_test<2,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemset_test<4,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemset_test<8,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemset_test<16, data_type, execute_policy> >(blksz, gs_mode, max_len);

  run_single_test_pass< rajaMemcpy_test<1,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemcpy_test<2,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemcpy_test<4,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemcpy_test<8,  data_type, execute_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajaMemcpy_test<16, data_type, execute_policy> >(blksz, gs_mode, max_len);

  run_single_test_pass< rajasumindex_test<1,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<2,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<4,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<8,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<16, data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);

  run_single_test_pass< rajasumindex_test<1,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<2,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<4,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<8,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumindex_test<16, data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);

  run_single_test_pass< rajasumreduce_test<1,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<2,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<4,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<8,  data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<16, data_type, execute_policy, reduce_policy> >(blksz, gs_mode, max_len);

  run_single_test_pass< rajasumreduce_test<1,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<2,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<4,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<8,  data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);
  run_single_test_pass< rajasumreduce_test<16, data_type, execute_policy, reduce_atomic_policy> >(blksz, gs_mode, max_len);

}

template <typename data_type, size_t block_size>
void run_raja_tests(RAJA::Index_type max_len)
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

    run_raja_test_pass<data_type, block_size>(gs_mode, max_len);

  }

}

template <typename data_type>
void run_typed_tests()
{
  RAJA::Index_type max_len = device_data.size/sizeof(data_type);

  run_cuda_test_pass<data_type>(max_len);

  run_cub_test_pass<data_type>(max_len);

  // run_raja_tests<data_type, 64  >(max_len);
  run_raja_tests<data_type, 128 >(max_len);
  // run_raja_tests<data_type, 192 >(max_len);
  run_raja_tests<data_type, 256 >(max_len);
  // run_raja_tests<data_type, 512 >(max_len);
  // run_raja_tests<data_type, 1024>(max_len);
}

void run_setup_tests()
{
  using data_type = int;

  RAJA::Index_type max_len = device_data.size/sizeof(data_type);

  run_cub_test_pass<data_type>(max_len);

  run_raja_tests<data_type, 64 >(max_len);
}

void run_all_tests()
{
  do_print = false;
  num_test_repeats = 2;


  sync_type = SynchronizationType::async_empty;

    // run some tests a few times to eat initialization overheads
    run_setup_tests();


  // run real tests
  do_print = true;
  num_test_repeats = 14;


  sync_type = SynchronizationType::async_full;

    run_typed_tests<int>();
    // run_typed_tests<float>();
    // run_typed_tests<unsigned long long int>();
    run_typed_tests<double>();

  sync_type = SynchronizationType::async_empty;

    run_typed_tests<double>();

  sync_type = SynchronizationType::sync;

    run_typed_tests<double>();

  sync_type = SynchronizationType::async_empty;

}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  cudaErrchk(cudaGetDeviceProperties(&devProp, 0));

  double estimated_launch_penalty_s = 1.0e-5; // 10 us

  size_t memory_clock_hz = size_t(1000)*size_t(devProp.memoryClockRate);
  size_t memory_bus_width_B = size_t(devProp.memoryBusWidth)/size_t(8);

  size_t max_memory_bandwidth_Bps = 2*memory_clock_hz*memory_bus_width_B;

  size_t data_size = size_t(devProp.totalGlobalMem)/size_t(2);
  global_max_len = data_size;

  size_t other_size = static_cast<size_t>(std::ceil(4*estimated_launch_penalty_s * double(max_memory_bandwidth_Bps) ));

  size_t pinned_size = 16*sizeof(double);

  printf("Memory Bandwidth %f GiBps data %f MiB other %f MiB\n", max_memory_bandwidth_Bps/(1024.0*1024.0*1024.0), data_size/(1024.0*1024.0), other_size/(1024.0*1024.0)); fflush(stdout);

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

    run_all_tests();

    // free resources
    cudaErrchk(cudaEventDestroy(end_event));
    cudaErrchk(cudaEventDestroy(start_event));

    cudaErrchk(cudaFreeHost(pinned_data.ptr)); pinned_data.ptr = nullptr; pinned_data.size = 0;

    cudaErrchk(cudaFree(device_data.ptr));  device_data.ptr = nullptr;  device_data.size = 0;
    cudaErrchk(cudaFree(device_other.ptr)); device_other.ptr = nullptr; device_other.size = 0;

    memory_type = nullptr;
  }

  {
    memory_type = "um";

    // allocate resources (um)
    cudaErrchk(cudaMallocManaged(&device_data.ptr, data_size)); device_data.size = data_size;
    cudaErrchk(cudaMalloc(&device_other.ptr, other_size)); device_other.size = other_size;
    cudaErrchk(cudaHostAlloc(&pinned_data.ptr, pinned_size, cudaHostAllocDefault)); pinned_data.size = pinned_size;

    cudaErrchk(cudaMemset(device_data.ptr, 0, data_size));
    cudaErrchk(cudaMemset(device_other.ptr, 0, other_size));

    cudaErrchk(cudaEventCreateWithFlags(&start_event, cudaEventDefault));
    cudaErrchk(cudaEventCreateWithFlags(&end_event, cudaEventDefault));

    run_all_tests();

    // free resources
    cudaErrchk(cudaEventDestroy(end_event));
    cudaErrchk(cudaEventDestroy(start_event));

    cudaErrchk(cudaFreeHost(pinned_data.ptr)); pinned_data.ptr = nullptr; pinned_data.size = 0;

    cudaErrchk(cudaFree(device_data.ptr));  device_data.ptr = nullptr;  device_data.size = 0;
    cudaErrchk(cudaFree(device_other.ptr)); device_other.ptr = nullptr; device_other.size = 0;

    memory_type = nullptr;
  }

  return 0;
}
