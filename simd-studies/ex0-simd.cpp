//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
/*
 *  Simd test 1 - adds vectors
 */

#define ADD_ALIGN_HINT  


#if defined(ADD_ALIGN_HINT)
#define VEC_ADD_BODY \
  z[i] = x[i] + y[i];
#else
#define VEC_ADD_BODY \
  c[i] = a[i] + b[i];
#endif


using realType = double;
using TFloat = realType * const RAJA_RESTRICT;

template<typename T>
void vecAdd_noVec(T a, T b, T c, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

  RAJA_NO_SIMD
  for(int i=0; i<N; ++i){    
    //VEC_ADD_BODY
    std::cout<<"data "<<std::endl;
  }
}

template<typename T>
void vecAdd_native(T a, T b, T c, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

  for(int i=0; i<N; ++i){
    VEC_ADD_BODY;
  }
}

template<typename T>
void vecAdd_simd(T a, T b, T c, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

  RAJA_SIMD
  for(int i=0; i<N; ++i){
    VEC_ADD_BODY;
  }


}




int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition benchmark...\n";

//
// Define vector length
//
  const int N = 10000;
  const RAJA::Index_type Niter = 20;

  realType *a = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  realType *b = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  realType *c = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  
  //Intialize data 
  for(int i=0; i<N; ++i)
    {
      a[i] = i*.10;
      b[i] = i*.10;
    }

  auto timer = RAJA::Timer();
  double minRun;
  minRun = std::numeric_limits<double>::max();  
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    vecAdd_noVec(a, b, c, N);
    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    std::cout<<"elapsed = "<<timer.elapsed()<<std::endl;
    std::cout<<"tMin = "<<tMin<<std::endl;
    if (tMin < minRun) minRun = tMin;
    ///timer.reset();
  }
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;




  
  std::cout << "\n DONE!...\n";
  return 0;
}

//
// Function to check result and report P/F.
//
template<typename T>
void checkResult(T* res, int len) 
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( res[i] != 0 ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}
