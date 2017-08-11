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
#include <iostream>
#include <utility>

#include <vector>
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

// cudaStream_t stream1 = 0;

const size_t gigabyte = 1024ul*1024ul*1024ul;


bool do_print = false;
int num_test_repeats = 1;

const double factor = 2.0;


struct cudaDeviceProp devProp;

std::unordered_map<std::string, std::vector<double>> test_map;



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

using data_type     = double ;
template <my_type>
using Ptr = my_type* ;


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

RAJA::Index_type next_size(RAJA::Index_type size, RAJA::Index_type max_size)
{
  size = RAJA::Index_type(std::ceil(size * factor));

  return size;
}

template <typename my_type, size_t size>
class Array;
template <typename my_type>
class Array<my_type, 1> {
  static constexpr const size_t size = 1;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{val} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(Raja::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(Raja::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 2> {
  static constexpr const size_t size = 2;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{val,val} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(Raja::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(Raja::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 4> {
  static constexpr const size_t size = 4;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{val,val,val,val} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(Raja::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(Raja::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 8> {
  static constexpr const size_t size = 8;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{val,val,val,val,val,val,val,val} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(Raja::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(Raja::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};
template <typename my_type>
class Array<my_type, 16> {
  static constexpr const size_t size = 16;
public:
  Array() = delete;
  template <typename T>
  Array(T const& val) : a{val,val,val,val,val,val,val,val,
                          val,val,val,val,val,val,val,val} {}
  RAJA_HOST_DEVICE
  my_type & operator[](size_t i){return a[i];}
  RAJA_HOST_DEVICE
  my_type const& operator[](size_t i)const{return a[i];}
  template <typename Func>
  void for_each(Func&& func) {for(size_t i = 0; i < size; ++i) func(a[i],i);}
  template <typename exec_policy, typename Func>
  void forall_for_each(Raja::Index_type len, Func&& func)
  {
    Array<my_type, size>& arr = *this;
    RAJA::forall<exec_policy>(0, len, [=] RAJA_DEVICE(Raja::Index_type idx) {
      for(size_t i = 0; i < size; ++i) func(arr[i],i,idx);
    });
  }
private:
  my_type a[size];
};

template <typename my_type, size_t size>
bool set_unique_ptrs(Array<my_type*, size>& a, size_t len, PointerHolder& p) {
  size_t bytes = len*sizeof(my_type);
  void* ptr = p.ptr;
  size_t sz = p.size;
  for(size_t i = 0; i < size; ++i) {
    if (RAJA::align(128, bytes, ptr, sz)) {
      a[i] = ptr;
      ptr  = static_cast<void*>(static_cast<char*>(ptr) + bytes);
      sz  -= bytes;
    } else {
      return false;
    }
  }
  return true;
}


struct cudaMemset_test {
  static constexpr const char[] name = "cudaMemset";
  template <size_t repeats, typename data_type, typename exec_policy, typename Reducer>
  bool do_test(RAJA::Index_type len) {
    Array<Ptr<data_type>, repeats> dp(nullptr);
    bool do_test = set_unique_ptrs(dp, len, device_data);
    if (do_test) {
      dp.for_each([=](Ptr<data_type> ptr, size_t) {
        cudaErrchk(cudaMemsetAsync(ptr, 0, len*sizeof(*ptr), 0));
      });
    }
    return do_test;
  }
};

struct cudaMemcpy_test {
  static constexpr const char[] name = "cudaMemcpy";
  template <size_t repeats, typename data_type, typename exec_policy, typename Reducer>
  bool do_test(RAJA::Index_type len) {
    Array<Ptr<data_type>, 2ul*repeats> dp(nullptr);
    bool do_test = set_unique_ptrs(dp, len, device_data);
    if (do_test) {
      Array<Pair<Ptr<data_type>>, repeats> dp_pair(nullptr);
      for (size_t i = 0; i < repeats; ++i) {
        dp_pair[i][0] = dp[2*i];
        dp_pair[i][1] = dp[2*i+1];
      }
      dp_pair.for_each([=](Pair<Ptr<data_type>> ptr_pair, size_t) {
        cudaErrchk(cudaMemcpyAsync(ptr_pair[0], ptr_pair[1], len*sizeof(*ptr_pair[1]), cudaMemcpyDefault, 0));
      });
    }
    return do_test;
  }
};

struct rajaMemset_test {
  static constexpr const char[] name = "rajaMemset";
  template <size_t repeats, typename data_type, typename exec_policy, typename Reducer>
  bool do_test(RAJA::Index_type len) {
    Array<Ptr<data_type>, repeats> dp(nullptr);
    bool do_test = set_unique_ptrs(dp, len, device_data);
    if (do_test) {
      dp.forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Ptr<data_type> ptr, size_t, RAJA::Index_type idx) {
        ptr[idx] = 0;
      });
    }
    return do_test;
  }
};

struct rajaMemcpy_test {
  static constexpr const char[] name = "rajaMemcpy";
  template <size_t repeats, typename data_type, typename exec_policy, typename Reducer>
  bool do_test(RAJA::Index_type len) {
    Array<Ptr<data_type>, 2ul*repeats> dp(nullptr);
    bool do_test = set_unique_ptrs(dp, len, device_data);
    if (do_test) {
      Array<Pair<Ptr<data_type>>, repeats> dp_pair(nullptr);
      for (size_t i = 0; i < repeats; ++i) {
        dp_pair[i][0] = dp[2*i];
        dp_pair[i][1] = dp[2*i+1];
      }
      dp_pair.forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Pair<Ptr<data_type>> ptr_pair, size_t, RAJA::Index_type idx) {
        ptr_pair[0][idx] = ptr_pair[1][idx];
      });
    }
    return do_test;
  }
};

struct rajapi_test {
  static constexpr const char[] name = "raja_pi_reduce";
  template <size_t repeats, typename data_type, typename exec_policy, typename Reducer>
  bool do_test(RAJA::Index_type len) {
    bool do_test = true;
    if (do_test) {
      Array<Reducer, repeats> rd(0);
      rd.forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t, RAJA::Index_type idx) {
        double x = (double(idx) + 0.5) / len;
        r.reduce(4.0 / (1.0 + x * x));
      });
    }
    return do_test;
  }
};

struct rajareduce_test {
  static constexpr const char[] name = "raja_array_reduce";
  template <size_t repeats, typename data_type, typename exec_policy, typename Reducer>
  bool do_test(RAJA::Index_type len) {
    Array<Ptr<data_type>, repeats> dp(nullptr);
    bool do_test = set_unique_ptrs(dp, len, device_data);
    if (do_test) {
      Array<Reducer, repeats> rd(0);
      rd.forall_for_each<exec_policy>(len, [=] RAJA_DEVICE(Reducer const& r, size_t i, RAJA::Index_type idx) {
        r.reduce(dp[i][idx]);
      });
    }
    return do_test;
  }
};

template <typename Reducer, typename data_type>
inline void do_cub_reduce(Ptr<data_type> ptr_in, Ptr<data_type> ptr_out, size_t len)
{
  char     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, len, Reducer{}, 0);
  d_temp_storage = RAJA::cuda::device_mempool_type::getInstance().malloc<char>(temp_storage_bytes);
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, len, Reducer{}, 0);
  RAJA::cuda::device_mempool_type::getInstance().free(d_temp_storage);
}

struct cubreduce_test {
  static constexpr const char[] name = "cub_array_reduce";
  template <size_t repeats, typename data_type, typename exec_policy, typename Reducer>
  bool do_test(RAJA::Index_type len) {
    Array<Ptr<data_type>, repeats> dp(nullptr);
    bool do_test = set_unique_ptrs(dp, len, device_data);
    if (do_test) {
      data_type* pinned_buf = static_cast<data_type*>(pinned_data);
      dp.for_each([=](Ptr<data_type> ptr, size_t i) {
        do_cub_reduce<CubSum>(ptr, pinned_buf+i, len);
      });
    }
    return do_test;
  }
};

static bool first_print = true;

template <typename Test, size_t repeats, typename data_type, typename exec_policy, typename Reducer>
void run_test_pass(const char* block_size, const char* gs_mode, RAJA::Index_type max_size)
{
  if (do_print) {
    if (first_print) {
      printf("test_name");
      RAJA::Index_type size = 32;
      while ( size <= max_size ) {
        printf("\t%li", size);
        size = next_size(size, max_size);
      }
      printf("\n"); fflush(stdout);
      first_print = false;
    }
    printf("%s(%zu)<%s>{%s}", Test::name, repeats, block_size, gs_mode);
  }

  RAJA::Index_type size = 32;
  while ( size <= max_size ) {

    double test_time = 0;

    for (int repeat = 0; repeat < num_test_repeats; ++repeat) {

      size_t copy_size = device_other.size/2;
      size_t gridSize = (copy_size + 128-1)/128;
      device_copy<<<gridSize,128,0,0>>>(static_cast<char*>(device_other.ptr),
                                        static_cast<char*>(device_other.ptr)+copy_size,
                                        copy_size);

      cudaErrchk(cudaEventRecord(start_event));
      Test{}.do_test<repeats, data_type, exec_policy, Reducer>(size);
      cudaErrchk(cudaEventRecord(end_event));
      cudaErrchk(cudaEventSynchronize(end_event));
      float fms;
      cudaErrchk(cudaEventElapsedTime(&fms, start_event, end_event));

      test_time += fms*1e3;

    }

    test_time /= (double)num_test_repeats;

    if (do_print) {
      printf("\t%e", test_time);
    }

    size = next_size(size, max_size);

  }

  if (do_print) {
    printf("\n"); fflush(stdout);
  }

}

template <typename data_type>
void run_cuda_test_pass(RAJA::Index_type max_size)
{
  const int block_size = 256;
  typedef RAJA::cuda_reduce<block_size, true> reduce_policy;
  typedef RAJA::cuda_exec<block_size, true> execute_policy;

  run_test_pass<cudaMemset_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemset_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemset_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemset_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemset_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >("n/a", "n/a", max_size);

  run_test_pass<cudaMemcpy_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemcpy_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemcpy_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemcpy_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cudaMemcpy_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);

  run_test_pass<cubreduce_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cubreduce_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cubreduce_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cubreduce_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
  run_test_pass<cubreduce_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("n/a", "n/a", max_size);
}

template <typename data_type, size_t block_size>
void run_block_size_test_pass(const char* gs_mode, RAJA::Index_type max_size)
{
  char blksz[128];
  snprintf(blksz, 128, "%zu", block_size);

  typedef RAJA::cuda_reduce<block_size, true> reduce_policy;
  typedef RAJA::cuda_reduce_atomic<block_size, true> reduce_atomic_policy;
  typedef RAJA::cuda_exec<block_size, true> execute_policy;

  run_test_pass<rajaMemset_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemset_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemset_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemset_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemset_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);

  run_test_pass<rajaMemcpy_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);

  run_test_pass<rajapi_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_policy, double> >(blksz, gs_mode, max_size);

  run_test_pass<rajapi_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >(blksz, gs_mode, max_size);
  run_test_pass<rajapi_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >(blksz, gs_mode, max_size);

  run_test_pass<rajareduce_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_policy, data_type> >(blksz, gs_mode, max_size);

  run_test_pass<rajareduce_test, 1,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 2,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 4,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 8,  data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, data_type> >(blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test, 16, data_type, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, data_type> >(blksz, gs_mode, max_size);

}

template <typename data_type, size_t... block_size>
void run_tests(RAJA::Index_type max_size)
{

  run_cuda_test_pass<data_type>(max_size);

  for (int i = 0; i < 3; ++i) {

    const char* gs_mode = nullptr;

    if (i == 1) {
      RAJA::cuda::getGridStrideMode() = RAJA::cuda::GridStrideMode::static_size;
      gs_mode = "static";
    } else if (i == 2) {
      RAJA::cuda::getGridStrideMode() = RAJA::cuda::GridStrideMode::occupancy_size;
      gs_mode = "occupancy";
    } else {
      RAJA::cuda::getGridStrideMode() = RAJA::cuda::GridStrideMode::disabled;
      gs_mode = "disabled";
    }

    run_block_size_test_pass<data_type, block_size>(gs_mode, max_size)...;

  }

}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  cudaErrchk(cudaGetDeviceProperties(&devProp, 0));

  RAJA::Index_type max_size = 1l;
  while(1) {
    if (2l*max_size >= devProp.totalGlobalMem) {
      break;
    }
    max_size *= 2l;
  }
  while(1) {
    if (gigabyte + max_size >= devProp.totalGlobalMem) {
      break;
    }
    max_size += gigabyte;
  }
  cudaErrchk(cudaMalloc(&device_data.ptr, max_size));
  cudaErrchk(cudaMemset(device_data.ptr, 0, max_size));
  device_data.size = max_size;

  size_t other_size = gigabyte/3;
  cudaErrchk(cudaMalloc(&device_other.ptr, other_size));
  cudaErrchk(cudaMemset(device_other.ptr, 0, other_size));
  device_other.size = other_size;

  size_t pinned_size = 16*sizeof(double);
  cudaErrchk(cudaHostAlloc(&pinned_data.ptr, pinned_size, cudaHostAllocDefault));
  pinned_data.size = pinned_size;

  cudaErrchk(cudaEventCreateWithFlags(&start_event, cudaEventDefault));
  cudaErrchk(cudaEventCreateWithFlags(&end_event, cudaEventDefault));

  // cudaErrchk(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

  {
    // run some tests a few times to eat initialization overheads
    do_print = false;
    num_test_repeats = 3;

    using data_type = double;

    run_tests<data_type, 64, 128, 256>(max_size);

  }

  {
    // run real tests
    do_print = true;
    num_test_repeats = 25;

    using data_type = double;

    run_tests<data_type, 64, 128, 192, 256, 512, 1024>(max_size);

  }

  // cudaErrchk(cudaStreamDestroy(stream1));

  cudaErrchk(cudaEventDestroy(end_event));
  cudaErrchk(cudaEventDestroy(start_event));

  cudaErrchk(cudaFreeHost(pinned_data.ptr)); pinned_data.ptr = nullptr; pinned_data.size = 0;

  cudaErrchk(cudaFree(device_data.ptr));  device_data.ptr = nullptr;  device_data.size = 0;
  cudaErrchk(cudaFree(device_other.ptr)); device_other.ptr = nullptr; device_other.size = 0;

  return 0;
}
