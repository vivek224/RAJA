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

#include <cstdlib>
#include <iostream>
#define RAJA_USE_NESTED

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
#include "RAJA/util/defines.hpp"


using SeqPolicy = RAJA::NestedPolicy<
                    RAJA::ExecList<RAJA::seq_exec,
                                   RAJA::seq_exec,
                                   RAJA::simd_exec>>;

using CudaPolicy = RAJA::NestedPolicy<
                    RAJA::ExecList<RAJA::cuda_threadblock_z_exec<4>,
                                   RAJA::cuda_threadblock_y_exec<4>,
                                   RAJA::cuda_threadblock_x_exec<8>>>;


int main()
{ 

  const int niter = 10;
  RAJA::RangeSegment seg_x(0, 200);
  RAJA::RangeSegment seg_y(0, 200);
  RAJA::RangeSegment seg_z(0, 200);
  RAJA::Index_type array_size = seg_x.size() * seg_y.size() * seg_z.size();
  
  // allocate host and device arrays
  double *h_ptr = new double[array_size];
  double *d_ptr = nullptr;
  cudaMalloc(&d_ptr, sizeof(double) * array_size);
  
  // Create host and device views
  RAJA::View<double, RAJA::Layout<3>> h_view(h_ptr, seg_x.size(), seg_y.size(), seg_z.size());
  RAJA::View<double, RAJA::Layout<3>> d_view(d_ptr, seg_x.size(), seg_y.size(), seg_z.size());

  // Create host, host-device and device lambdas
  auto h_lambda = [=](int x, int y, int z){
    h_view(x,y,z) = x*y+z;
  };  
  
  auto d_lambda = [=] __device__ (int x, int y, int z){
    d_view(x,y,z) = x*y+z;
  };
  
  auto h_hd_lambda = [=] __host__ __device__ (int x, int y, int z){
    h_view(x,y,z) = x*y+z;
  };
  
  auto d_hd_lambda = [=] __host__ __device__ (int x, int y, int z){
    d_view(x,y,z) = x*y+z;
  };
  
   
  

  double times[4] = {0.0, 0.0, 0.0, 0.0};
  
  // do a bunch of iterations to get better statistics
  for(int iter = 0;iter < niter;++ iter){     
  
    // run through each loop
    for(int loop = 0;loop < 4;++ loop){
    
      printf("Iter %d, loop %d: ", iter, loop);
    
      RAJA::Timer loop_timer;
      loop_timer.start("loop");
    
      switch(loop){
        case 0:
          RAJA::forallN<SeqPolicy>(seg_x, seg_y, seg_z, h_lambda);
          break;
        case 1:
          RAJA::forallN<CudaPolicy>(seg_x, seg_y, seg_z, d_lambda);  
          break;
        case 2:
          RAJA::forallN<SeqPolicy>(seg_x, seg_y, seg_z, h_hd_lambda);
          break;
        case 3:
          RAJA::forallN<CudaPolicy>(seg_x, seg_y, seg_z, d_hd_lambda);  
          break;
      }
      
      loop_timer.stop("loop");
      times[loop] += loop_timer.elapsed(); 
      printf("%lf seconds\n", loop_timer.elapsed());
    } 
    
    
  }  
    

  delete[] h_ptr;
  cudaFree(d_ptr);
 
  printf("\n");
  printf("Average execution times in seconds:\n");
  printf("host lambda        on host:    %4.8lf\n", times[0]/(double)niter);
  printf("device lambda      on device:  %4.8lf\n", times[1]/(double)niter);
  printf("host-device lambda on host:    %4.8lf, slowdown %.3lf\n", times[2]/(double)niter, times[2]/times[0]);
  printf("host-device lambda on device:  %4.8lf, slowdown %.3lf\n", times[3]/(double)niter, times[3]/times[1]);
 
  return 0;
}
