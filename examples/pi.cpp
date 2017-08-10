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
#include <vector>
#include <utility>

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

cudaEvent_t start_event;
cudaEvent_t end_event;

bool do_print = false;
int num_test_repeats = 1;

const double factor = 2.0;

double* device0 = nullptr;
double* device1 = nullptr;
double* device2 = nullptr;
double* device3 = nullptr;
double* device4 = nullptr;
double* device5 = nullptr;
double* device6 = nullptr;
double* device7 = nullptr;

double* pinned0 = nullptr;

using double_ptr = double*;


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

struct cudaMemcpy_test1 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1);
    cudaErrchk(cudaMemcpyAsync(d0, d1, size*sizeof(*d1), cudaMemcpyDefault, 0));
  }
};
struct cudaMemcpy_test2 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3);
    cudaErrchk(cudaMemcpyAsync(d0, d1, size*sizeof(*d1), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d2, d3, size*sizeof(*d3), cudaMemcpyDefault, 0));
  }
};
struct cudaMemcpy_test4 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3),
               d4(device4), d5(device5),
               d6(device6), d7(device7);
    cudaErrchk(cudaMemcpyAsync(d0, d1, size*sizeof(*d1), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d2, d3, size*sizeof(*d3), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d4, d5, size*sizeof(*d5), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d6, d7, size*sizeof(*d7), cudaMemcpyDefault, 0));
  }
};
struct cudaMemcpy_test8 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3),
               d4(device4), d5(device5),
               d6(device6), d7(device7),
               d8(device1), d9(device0),
               d10(device3), d11(device2),
               d12(device5), d13(device4),
               d14(device7), d15(device6);
    cudaErrchk(cudaMemcpyAsync(d0,  d1,  size*sizeof(*d1), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d2,  d3,  size*sizeof(*d3), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d4,  d5,  size*sizeof(*d5), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d6,  d7,  size*sizeof(*d7), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d8,  d9,  size*sizeof(*d9), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d10, d11, size*sizeof(*d11), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d12, d13, size*sizeof(*d13), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d14, d15, size*sizeof(*d15), cudaMemcpyDefault, 0));
  }
};
struct cudaMemcpy_test16 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3),
               d4(device4), d5(device5),
               d6(device6), d7(device7),
               d8(device1), d9(device0),
               d10(device3), d11(device2),
               d12(device5), d13(device4),
               d14(device7), d15(device6),
               d16(device0), d17(device3),
               d18(device2), d19(device1),
               d20(device4), d21(device7),
               d22(device6), d23(device5),
               d24(device1), d25(device2),
               d26(device3), d27(device0),
               d28(device5), d29(device6),
               d30(device7), d31(device4);
    cudaErrchk(cudaMemcpyAsync(d0,  d1,  size*sizeof(*d1), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d2,  d3,  size*sizeof(*d3), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d4,  d5,  size*sizeof(*d5), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d6,  d7,  size*sizeof(*d7), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d8,  d9,  size*sizeof(*d9), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d10, d11, size*sizeof(*d11), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d12, d13, size*sizeof(*d13), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d14, d15, size*sizeof(*d15), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d16, d17, size*sizeof(*d17), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d18, d19, size*sizeof(*d19), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d20, d21, size*sizeof(*d21), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d22, d23, size*sizeof(*d23), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d24, d25, size*sizeof(*d25), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d26, d27, size*sizeof(*d27), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d28, d29, size*sizeof(*d29), cudaMemcpyDefault, 0));
    cudaErrchk(cudaMemcpyAsync(d30, d31, size*sizeof(*d31), cudaMemcpyDefault, 0));
  }
};


struct rajaMemcpy_test1 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      d0[i] = d1[i];
    });
  }
};
struct rajaMemcpy_test2 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      d0[i] = d1[i];
      d2[i] = d3[i];
    });
  }
};
struct rajaMemcpy_test4 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3),
               d4(device4), d5(device5),
               d6(device6), d7(device7);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      d0[i] = d1[i];
      d2[i] = d3[i];
      d4[i] = d5[i];
      d6[i] = d7[i];
    });
  }
};
struct rajaMemcpy_test8 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3),
               d4(device4), d5(device5),
               d6(device6), d7(device7),
               d8(device1), d9(device0),
               d10(device3), d11(device2),
               d12(device5), d13(device4),
               d14(device7), d15(device6);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      d0[i] = d1[i];
      d2[i] = d3[i];
      d4[i] = d5[i];
      d6[i] = d7[i];
      d8[i] = d9[i];
      d10[i] = d11[i];
      d12[i] = d13[i];
      d14[i] = d15[i];
    });
  }
};
struct rajaMemcpy_test16 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1),
               d2(device2), d3(device3),
               d4(device4), d5(device5),
               d6(device6), d7(device7),
               d8(device1), d9(device0),
               d10(device3), d11(device2),
               d12(device5), d13(device4),
               d14(device7), d15(device6),
               d16(device0), d17(device3),
               d18(device2), d19(device1),
               d20(device4), d21(device7),
               d22(device6), d23(device5),
               d24(device1), d25(device2),
               d26(device3), d27(device0),
               d28(device5), d29(device6),
               d30(device7), d31(device4);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      d0[i] = d1[i];
      d2[i] = d3[i];
      d4[i] = d5[i];
      d6[i] = d7[i];
      d8[i] = d9[i];
      d10[i] = d11[i];
      d12[i] = d13[i];
      d14[i] = d15[i];
      d16[i] = d17[i];
      d18[i] = d19[i];
      d20[i] = d21[i];
      d22[i] = d23[i];
      d24[i] = d25[i];
      d26[i] = d27[i];
      d28[i] = d29[i];
      d30[i] = d31[i];
    });
  }
};


struct rajapi_test1 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      double x = (double(i) + 0.5) / size;
      r0.reduce(4.0 / (1.0 + x * x));
    });
  }
};
struct rajapi_test2 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      double x = (double(i) + 0.5) / size;
      r0.reduce(4.0 / (1.0 + x * x));
      r1.reduce(4.0 / (1.0 + x * x));
    });
  }
};
struct rajapi_test4 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0), r2(0.0), r3(0.0);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      double x = (double(i) + 0.5) / size;
      r0.reduce(4.0 / (1.0 + x * x));
      r1.reduce(4.0 / (1.0 + x * x));
      r2.reduce(4.0 / (1.0 + x * x));
      r3.reduce(4.0 / (1.0 + x * x));
    });
  }
};
struct rajapi_test8 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0), r2(0.0), r3(0.0),
            r4(0.0), r5(0.0), r6(0.0), r7(0.0);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      double x = (double(i) + 0.5) / size;
      r0.reduce(4.0 / (1.0 + x * x));
      r1.reduce(4.0 / (1.0 + x * x));
      r2.reduce(4.0 / (1.0 + x * x));
      r3.reduce(4.0 / (1.0 + x * x));
      r4.reduce(4.0 / (1.0 + x * x));
      r5.reduce(4.0 / (1.0 + x * x));
      r6.reduce(4.0 / (1.0 + x * x));
      r7.reduce(4.0 / (1.0 + x * x));
    });
  }
};
struct rajapi_test16 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0), r2(0.0), r3(0.0),
            r4(0.0), r5(0.0), r6(0.0), r7(0.0),
            r8(0.0), r9(0.0), r10(0.0), r11(0.0),
            r12(0.0), r13(0.0), r14(0.0), r15(0.0);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      double x = (double(i) + 0.5) / size;
      r0.reduce(4.0 / (1.0 + x * x));
      r1.reduce(4.0 / (1.0 + x * x));
      r2.reduce(4.0 / (1.0 + x * x));
      r3.reduce(4.0 / (1.0 + x * x));
      r4.reduce(4.0 / (1.0 + x * x));
      r5.reduce(4.0 / (1.0 + x * x));
      r6.reduce(4.0 / (1.0 + x * x));
      r7.reduce(4.0 / (1.0 + x * x));
      r8.reduce(4.0 / (1.0 + x * x));
      r9.reduce(4.0 / (1.0 + x * x));
      r10.reduce(4.0 / (1.0 + x * x));
      r11.reduce(4.0 / (1.0 + x * x));
      r12.reduce(4.0 / (1.0 + x * x));
      r13.reduce(4.0 / (1.0 + x * x));
      r14.reduce(4.0 / (1.0 + x * x));
      r15.reduce(4.0 / (1.0 + x * x));
    });
  }
};


struct rajareduce_test1 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0);
    double_ptr d0(device0);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      r0.reduce(d0[i]);
    });
  }
};
struct rajareduce_test2 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0);
    double_ptr d0(device0), d1(device1);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      r0.reduce(d0[i]);
      r1.reduce(d1[i]);
    });
  }
};
struct rajareduce_test4 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0), r2(0.0), r3(0.0);
    double_ptr d0(device0), d1(device1), d2(device2), d3(device3);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      r0.reduce(d0[i]);
      r1.reduce(d1[i]);
      r2.reduce(d2[i]);
      r3.reduce(d3[i]);
    });
  }
};
struct rajareduce_test8 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0), r2(0.0), r3(0.0),
            r4(0.0), r5(0.0), r6(0.0), r7(0.0);
    double_ptr d0(device0), d1(device1), d2(device2), d3(device3),
            d4(device4), d5(device5), d6(device6), d7(device7);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      r0.reduce(d0[i]);
      r1.reduce(d1[i]);
      r2.reduce(d2[i]);
      r3.reduce(d3[i]);
      r4.reduce(d4[i]);
      r5.reduce(d5[i]);
      r6.reduce(d6[i]);
      r7.reduce(d7[i]);
    });
  }
};
struct rajareduce_test16 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    Reducer r0(0.0), r1(0.0), r2(0.0), r3(0.0),
            r4(0.0), r5(0.0), r6(0.0), r7(0.0),
            r8(0.0), r9(0.0), r10(0.0), r11(0.0),
            r12(0.0), r13(0.0), r14(0.0), r15(0.0);
    double_ptr d0(device0), d1(device1), d2(device2), d3(device3),
            d4(device4), d5(device5), d6(device6), d7(device7),
            d8(device0), d9(device1), d10(device2), d11(device3),
            d12(device4), d13(device5), d14(device6), d15(device7);
    RAJA::forall<exec_policy>(0, size, [=] RAJA_DEVICE(int i) {
      r0.reduce(d0[i]);
      r1.reduce(d1[i]);
      r2.reduce(d2[i]);
      r3.reduce(d3[i]);
      r4.reduce(d4[i]);
      r5.reduce(d5[i]);
      r6.reduce(d6[i]);
      r7.reduce(d7[i]);
      r8.reduce(d8[i]);
      r9.reduce(d9[i]);
      r10.reduce(d10[i]);
      r11.reduce(d11[i]);
      r12.reduce(d12[i]);
      r13.reduce(d13[i]);
      r14.reduce(d14[i]);
      r15.reduce(d15[i]);
    });
  }
};

template <typename Reducer>
inline void do_cub_reduce(double_ptr ptr_in, double_ptr ptr_out, size_t size)
{
  char     *d_temp_storage = NULL;
  size_t   temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, size, Reducer{}, 0.0);
  d_temp_storage = RAJA::cuda::device_mempool_type::getInstance().malloc<char>(temp_storage_bytes);
  cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, ptr_in, ptr_out, size, Reducer{}, 0.0);
  RAJA::cuda::device_mempool_type::getInstance().free(d_temp_storage);
}

struct cubreduce_test1 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0);
    do_cub_reduce<CubSum>(d0, pinned0+0,size);
  }
};
struct cubreduce_test2 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1);
    do_cub_reduce<CubSum>(d0, pinned0+0,size);
    do_cub_reduce<CubSum>(d1, pinned0+1,size);
  }
};
struct cubreduce_test4 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1), d2(device2), d3(device3);
    do_cub_reduce<CubSum>(d0, pinned0+0,size);
    do_cub_reduce<CubSum>(d1, pinned0+1,size);
    do_cub_reduce<CubSum>(d2, pinned0+2,size);
    do_cub_reduce<CubSum>(d3, pinned0+3,size);
  }
};
struct cubreduce_test8 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1), d2(device2), d3(device3),
            d4(device4), d5(device5), d6(device6), d7(device7);
    do_cub_reduce<CubSum>(d0, pinned0+0,size);
    do_cub_reduce<CubSum>(d1, pinned0+1,size);
    do_cub_reduce<CubSum>(d2, pinned0+2,size);
    do_cub_reduce<CubSum>(d3, pinned0+3,size);
    do_cub_reduce<CubSum>(d4, pinned0+4,size);
    do_cub_reduce<CubSum>(d5, pinned0+5,size);
    do_cub_reduce<CubSum>(d6, pinned0+6,size);
    do_cub_reduce<CubSum>(d7, pinned0+7,size);
  }
};
struct cubreduce_test16 {
  template <typename exec_policy, typename Reducer>
  void do_test(exec_policy const&, Reducer const&, RAJA::Index_type size) {
    double_ptr d0(device0), d1(device1), d2(device2), d3(device3),
            d4(device4), d5(device5), d6(device6), d7(device7),
            d8(device0), d9(device1), d10(device2), d11(device3),
            d12(device4), d13(device5), d14(device6), d15(device7);
    do_cub_reduce<CubSum>(d0, pinned0+0,size);
    do_cub_reduce<CubSum>(d1, pinned0+1,size);
    do_cub_reduce<CubSum>(d2, pinned0+2,size);
    do_cub_reduce<CubSum>(d3, pinned0+3,size);
    do_cub_reduce<CubSum>(d4, pinned0+4,size);
    do_cub_reduce<CubSum>(d5, pinned0+5,size);
    do_cub_reduce<CubSum>(d6, pinned0+6,size);
    do_cub_reduce<CubSum>(d7, pinned0+7,size);
    do_cub_reduce<CubSum>(d8, pinned0+8,size);
    do_cub_reduce<CubSum>(d9, pinned0+9,size);
    do_cub_reduce<CubSum>(d10, pinned0+10,size);
    do_cub_reduce<CubSum>(d11, pinned0+11,size);
    do_cub_reduce<CubSum>(d12, pinned0+12,size);
    do_cub_reduce<CubSum>(d13, pinned0+13,size);
    do_cub_reduce<CubSum>(d14, pinned0+14,size);
    do_cub_reduce<CubSum>(d15, pinned0+15,size);
  }
};

static bool first_print = true;

template <typename Test, typename exec_policy, typename Reducer>
void run_test_pass(const char* test_name, const char* block_size, const char* gs_mode, RAJA::Index_type max_size)
{
  using value_type = std::pair<RAJA::Index_type, double>;

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
    printf("%s<%s>{%s}", test_name, block_size, gs_mode);
  }

  RAJA::Index_type size = 32;
  while ( size <= max_size ) {

    double test_time = 0.0;

    for (int repeat = 0; repeat < num_test_repeats; ++repeat) {

      int copy_size = 1024*1024;
      int gridSize = (copy_size + 128-1)/128;
      device_copy<<<gridSize,128,0,0>>>((char*)device7, (char*)device6, copy_size);

      cudaErrchk(cudaEventRecord(start_event));
      Test{}.do_test(exec_policy{}, Reducer{0.0}, size);
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


template <int block_size>
void run_block_size_test_pass(const char* gs_mode, RAJA::Index_type max_size)
{
  char blksz[128];
  snprintf(blksz, 128, "%i", block_size);

  typedef RAJA::cuda_reduce<block_size, true> reduce_policy;
  typedef RAJA::cuda_reduce_atomic<block_size, true> reduce_atomic_policy;
  typedef RAJA::cuda_exec<block_size, true> execute_policy;

  run_test_pass<rajaMemcpy_test1, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_memcpy(1)", blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test2, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_memcpy(2)", blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_memcpy(4)", blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test8, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_memcpy(8)", blksz, gs_mode, max_size);
  run_test_pass<rajaMemcpy_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_memcpy(16)", blksz, gs_mode, max_size);

  run_test_pass<rajapi_test1, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_pi_sum(1)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test2, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_pi_sum(2)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_pi_sum(4)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test8, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_pi_sum(8)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_pi_sum(16)", blksz, gs_mode, max_size);

  run_test_pass<rajapi_test1, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_pi_sum_atomic(1)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test2, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_pi_sum_atomic(2)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test4, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_pi_sum_atomic(4)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test8, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_pi_sum_atomic(8)", blksz, gs_mode, max_size);
  run_test_pass<rajapi_test16, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_pi_sum_atomic(16)", blksz, gs_mode, max_size);

  run_test_pass<rajareduce_test1, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_array_sum(1)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test2, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_array_sum(2)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_array_sum(4)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test8, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_array_sum(8)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_array_sum(16)", blksz, gs_mode, max_size);

  run_test_pass<rajareduce_test1, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_array_sum_atomic(1)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test2, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_array_sum_atomic(2)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test4, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_array_sum_atomic(4)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test8, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_array_sum_atomic(8)", blksz, gs_mode, max_size);
  run_test_pass<rajareduce_test16, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_array_sum_atomic(16)", blksz, gs_mode, max_size);

}



int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
  const RAJA::Index_type max_size = 128l*1024l*1024l;

  cudaErrchk(cudaMalloc(&device0, max_size*sizeof(*device0)));
  cudaErrchk(cudaMemset(device0, 0, max_size*sizeof(*device0)));
  cudaErrchk(cudaMalloc(&device1, max_size*sizeof(*device1)));
  cudaErrchk(cudaMemset(device1, 0, max_size*sizeof(*device1)));
  cudaErrchk(cudaMalloc(&device2, max_size*sizeof(*device2)));
  cudaErrchk(cudaMemset(device2, 0, max_size*sizeof(*device2)));
  cudaErrchk(cudaMalloc(&device3, max_size*sizeof(*device3)));
  cudaErrchk(cudaMemset(device3, 0, max_size*sizeof(*device3)));
  cudaErrchk(cudaMalloc(&device4, max_size*sizeof(*device4)));
  cudaErrchk(cudaMemset(device4, 0, max_size*sizeof(*device4)));
  cudaErrchk(cudaMalloc(&device5, max_size*sizeof(*device5)));
  cudaErrchk(cudaMemset(device5, 0, max_size*sizeof(*device5)));
  cudaErrchk(cudaMalloc(&device6, max_size*sizeof(*device6)));
  cudaErrchk(cudaMemset(device6, 0, max_size*sizeof(*device6)));
  cudaErrchk(cudaMalloc(&device7, max_size*sizeof(*device7)));
  cudaErrchk(cudaMemset(device7, 0, max_size*sizeof(*device7)));

  cudaErrchk(cudaHostAlloc(&pinned0, 16*sizeof(*pinned0), cudaHostAllocDefault));

  cudaErrchk(cudaEventCreateWithFlags(&start_event, cudaEventDefault));
  cudaErrchk(cudaEventCreateWithFlags(&end_event, cudaEventDefault));

  // cudaErrchk(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));

  {
    // run all tests once to eat initialization overheads
    do_print = false;
    num_test_repeats = 8;

    const int block_size = 256;
    typedef RAJA::cuda_reduce<block_size, true> reduce_policy;
    typedef RAJA::cuda_reduce_atomic<block_size, true> reduce_atomic_policy;
    typedef RAJA::cuda_exec<block_size, true> execute_policy;

    run_test_pass<cudaMemcpy_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cuda_memcpy(4)", "n/a", "n/a", max_size);
    run_test_pass<cudaMemcpy_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cuda_memcpy(16)", "n/a", "n/a", max_size);

    run_test_pass<cubreduce_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cub_array_sum(4)", "n/a", "n/a", max_size);
    run_test_pass<cubreduce_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cub_array_sum(16)", "n/a", "n/a", max_size);

    run_test_pass<rajaMemcpy_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_memcpy(4)", "256", "disabled", max_size);
    run_test_pass<rajaMemcpy_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_memcpy(16)", "256", "disabled", max_size);

    run_test_pass<rajapi_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_pi_sum(4)", "256", "disabled", max_size);
    run_test_pass<rajapi_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_pi_sum(16)", "256", "disabled", max_size);

    run_test_pass<rajapi_test4, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_pi_sum_atomic(4)", "256", "disabled", max_size);
    run_test_pass<rajapi_test16, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_pi_sum_atomic(16)", "256", "disabled", max_size);

    run_test_pass<rajareduce_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_array_sum(4)", "256", "disabled", max_size);
    run_test_pass<rajareduce_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("raja_array_sum(16)", "256", "disabled", max_size);

    run_test_pass<rajareduce_test4, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_array_sum_atomic(4)", "256", "disabled", max_size);
    run_test_pass<rajareduce_test16, execute_policy, RAJA::ReduceSum<reduce_atomic_policy, double> >("raja_array_sum_atomic(16)", "256", "disabled", max_size);

  }

  {
    // run real tests
    do_print = true;
    num_test_repeats = 25;

    {
      const int block_size = 256;
      typedef RAJA::cuda_reduce<block_size, true> reduce_policy;
      typedef RAJA::cuda_exec<block_size, true> execute_policy;

      run_test_pass<cudaMemcpy_test1, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cuda_memcpy(1)", "n/a", "n/a", max_size);
      run_test_pass<cudaMemcpy_test2, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cuda_memcpy(2)", "n/a", "n/a", max_size);
      run_test_pass<cudaMemcpy_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cuda_memcpy(4)", "n/a", "n/a", max_size);
      run_test_pass<cudaMemcpy_test8, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cuda_memcpy(8)", "n/a", "n/a", max_size);
      run_test_pass<cudaMemcpy_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cuda_memcpy(16)", "n/a", "n/a", max_size);

      run_test_pass<cubreduce_test1, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cub_array_sum(1)", "n/a", "n/a", max_size);
      run_test_pass<cubreduce_test2, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cub_array_sum(2)", "n/a", "n/a", max_size);
      run_test_pass<cubreduce_test4, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cub_array_sum(4)", "n/a", "n/a", max_size);
      run_test_pass<cubreduce_test8, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cub_array_sum(8)", "n/a", "n/a", max_size);
      run_test_pass<cubreduce_test16, execute_policy, RAJA::ReduceSum<reduce_policy, double> >("cub_array_sum(16)", "n/a", "n/a", max_size);
    }

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

      run_block_size_test_pass<64>(gs_mode, max_size);
      run_block_size_test_pass<128>(gs_mode, max_size);
      run_block_size_test_pass<192>(gs_mode, max_size);
      run_block_size_test_pass<256>(gs_mode, max_size);
      run_block_size_test_pass<512>(gs_mode, max_size);
      run_block_size_test_pass<1024>(gs_mode, max_size);

    }

  }

  // cudaErrchk(cudaStreamDestroy(stream1));

  cudaErrchk(cudaEventDestroy(end_event));
  cudaErrchk(cudaEventDestroy(start_event));

  cudaErrchk(cudaFreeHost(pinned0)); pinned0 = nullptr;

  cudaErrchk(cudaFree(device7)); device7 = nullptr;
  cudaErrchk(cudaFree(device6)); device6 = nullptr;
  cudaErrchk(cudaFree(device5)); device5 = nullptr;
  cudaErrchk(cudaFree(device4)); device4 = nullptr;
  cudaErrchk(cudaFree(device3)); device3 = nullptr;
  cudaErrchk(cudaFree(device2)); device2 = nullptr;
  cudaErrchk(cudaFree(device1)); device1 = nullptr;
  cudaErrchk(cudaFree(device0)); device0 = nullptr;

  return 0;
}
