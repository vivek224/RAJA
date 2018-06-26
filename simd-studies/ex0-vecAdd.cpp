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
 *  Simd benchmark 1 - adds vectors
 */

//#define ADD_ALIGN_HINT  


#if defined(ADD_ALIGN_HINT)

#define VEC_ADD_BODY \
  z[i] = x[i] + y[i];
#else
#define VEC_ADD_BODY \
  c[i] = a[i] + b[i];
#endif


using realType = double;
using TFloat = realType * const RAJA_RESTRICT;
RAJA_INLINE
void vecAdd_noVec(TFloat a, TFloat b, TFloat c, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

  RAJA_NO_SIMD
  for(RAJA::Index_type i=0; i<N; ++i){
    VEC_ADD_BODY
  }
}

RAJA_INLINE
void vecAdd_native(TFloat a, TFloat b, TFloat c, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

  for(RAJA::Index_type i=0; i<N; ++i){
    VEC_ADD_BODY;
  }

}

RAJA_INLINE
void vecAdd_simd(TFloat a, TFloat b, TFloat c, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

  RAJA_SIMD
  for(RAJA::Index_type i=0; i<N; ++i){
    VEC_ADD_BODY;
  }
  
}

template<typename POL>
RAJA_INLINE
void vecAdd_RAJA(TFloat a, TFloat b, TFloat c, RAJA::Index_type N)
{

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

  RAJA::forall<POL>(RAJA::RangeSegment(0, N), [=] (RAJA::Index_type i) {
      VEC_ADD_BODY;
    });

}


template<typename T>
void checkResult(T res, RAJA::Index_type len);

int main(int argc, char *argv[])
{

  if(argc !=2 ){
  exit(-1);
  }

//
// Define vector length
//
  RAJA::Timer::ElapsedType runTime; 
  const RAJA::Index_type N = atoi(argv[1]);
  //const RAJA::Index_type N = 2048;  // RAJA code runs slower

#if defined(ADD_ALIGN_HINT)
  std::cout << "\n\nRAJA vector addition benchmark with alignment hint...\n";
#else
  std::cout << "\n\nRAJA vector addition benchmark...\n";
#endif
  std::cout<<"No of entries "<<N<<"\n\n"<<std::endl;
  
  auto timer = RAJA::Timer();
  const RAJA::Index_type Niter = 50000;

  TFloat a = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  TFloat b = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  TFloat c = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  
  //Intialize data 
  for(RAJA::Index_type i=0; i<N; ++i)
    {
      a[i] = i*.10;
      b[i] = -i*.10;
    }
  
  //---------------------------------------------------------
  std::cout<<"Native C - strictly sequential"<<std::endl;
  //---------------------------------------------------------

  for(RAJA::Index_type it = 0; it < Niter; ++it){

    timer.start();
    vecAdd_noVec(a, b, c, N);
    timer.stop();
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, N);


  //---------------------------------------------------------
  std::cout<<"Native C - raw loop"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    vecAdd_native(a, b, c, N);
    timer.stop();    
  }

  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, N);

  //---------------------------------------------------------
  std::cout<<"Native C - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    vecAdd_simd(a, b, c, N);
    timer.stop();
    
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, N);

  //---------------------------------------------------------
  std::cout<<"RAJA - strictly sequential"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    vecAdd_RAJA<RAJA::seq_exec>(a, b, c, N);
     timer.stop(); 
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, N);

  //---------------------------------------------------------
  std::cout<<"RAJA - raw loop"<<std::endl;
  //---------------------------------------------------------  
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    vecAdd_RAJA<RAJA::loop_exec>(a, b, c, N);
     timer.stop(); 
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, N);

  //---------------------------------------------------------
  std::cout<<"RAJA - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    vecAdd_RAJA<RAJA::simd_exec>(a, b, c, N);
    timer.stop(); 
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, N);
  //---------------------------------------------------------

  
  std::cout << "\n DONE!...\n";
  return 0;
}

//
// Function to check result and report P/F.
//
template<typename T>
void checkResult(T res, RAJA::Index_type len) 
{
  bool correct = true;
  for (RAJA::Index_type i = 0; i < len; i++) {
    if ( std::abs(res[i]) > 1e-9 ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\t result -- PASS\n\n";
  } else {
    std::cout << "\t result -- FAIL\n\n";
  }
}
