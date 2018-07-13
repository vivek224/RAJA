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
 *  Simd benchmark 3 - dot product
 */

//#define ADD_ALIGN_HINT


#if defined(ADD_ALIGN_HINT)

#define DOT_BODY \
   dot += x[i]*y[i];
#else
#define DOT_BODY \
  dot += a[i]*b[i];
#endif

using realType = double;
using TFloat = realType * const RAJA_RESTRICT;

RAJA_INLINE
void dot_noVec(TFloat a, TFloat b, double &dot, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(a);
#endif

  RAJA_NO_SIMD
  for(RAJA::Index_type i=0; i<N; ++i){
    DOT_BODY
  }
}

RAJA_INLINE
void dot_native(TFloat a, TFloat b, double &dot, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif

  for(RAJA::Index_type i=0; i<N; ++i){
    DOT_BODY;
  }

}

RAJA_INLINE
void dot_simd(TFloat a, TFloat b, double &dot, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif
  RAJA_SIMD
  for(RAJA::Index_type i=0; i<N; ++i){
    DOT_BODY;
  }
  
}

template<typename RDot>
RAJA_INLINE
void dot_seq_RAJA(TFloat a, TFloat b, RDot &dot, RAJA::Index_type N)
{

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif

#pragma forceinline recursive
  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N), [=] (RAJA::Index_type i) {
      DOT_BODY
    });

}


template<typename RDot>
RAJA_INLINE
void dot_loop_RAJA(TFloat a, TFloat b, RDot &dot, RAJA::Index_type N)
{

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif

#pragma forceinline recursive
  RAJA::forall<RAJA::loop_exec>(RAJA::RangeSegment(0, N), [=] (RAJA::Index_type i) {
      DOT_BODY
    });

}


template<typename RDot>
RAJA_INLINE
void dot_simd_RAJA(TFloat a, TFloat b, RDot &dot, RAJA::Index_type N)
{

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif

#pragma forceinline recursive
  RAJA::forall<RAJA::simd_exec>(RAJA::RangeSegment(0, N), [=] (RAJA::Index_type i) {
      DOT_BODY
    });

}

void checkResult(double dot, RAJA::Index_type len);

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
  std::cout << "\n\nRAJA dot product benchmark with alignment hint...\n";
#else
  std::cout << "\n\nRAJA dot product addition benchmark...\n";
#endif
  std::cout<<"No of entries "<<N<<"\n\n"<<std::endl;
  
  auto timer = RAJA::Timer();
  const RAJA::Index_type Niter = 50000;

  double dot = 0.0;
  TFloat a = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  TFloat b = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  
  //Intialize data 
  for(RAJA::Index_type i=0; i<N; ++i)
    {
      a[i] = 1./N;
      b[i] = 1.0;
    }

  //---------------------------------------------------------
  std::cout<<"Native C - strictly sequential"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    dot = 0.0;
    timer.start();
    dot_noVec(a, b, dot, N);
    timer.stop();    
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);


  //---------------------------------------------------------
  std::cout<<"Native C - raw loop"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    dot = 0.0;
    timer.start();
    dot_native(a, b, dot, N);
    timer.stop();
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);

  //---------------------------------------------------------
  std::cout<<"Native C - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    dot = 0.0;
    timer.start();
    dot_simd(a, b, dot, N);
    timer.stop();
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);

  //---------------------------------------------------------
  std::cout<<"RAJA - strictly sequential"<<std::endl;
  //---------------------------------------------------------
  RAJA::ReduceSum<RAJA::seq_reduce, double> rdot(0.0);

  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    dot_seq_RAJA(a, b, rdot, N);
    timer.stop();
    dot = rdot.get();
    rdot.reset(0.0);
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);


  //---------------------------------------------------------
  std::cout<<"RAJA - raw loop"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    dot_loop_RAJA(a, b, rdot, N);
    timer.stop();
    dot = rdot.get();
    rdot.reset(0.0);
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);

  //---------------------------------------------------------
  std::cout<<"RAJA - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    timer.start();
    dot_simd_RAJA(a, b, rdot, N);
    timer.stop();
    dot = rdot.get();
    rdot.reset(0.0);
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);
  //---------------------------------------------------------

  std::cout << "\n DONE!...\n";

}

void checkResult(double dot, RAJA::Index_type len){
  bool correct = true;
  for (RAJA::Index_type i = 0; i < len; i++) {
    if ( std::abs(dot - 1) > 1e-9 ) { correct = false; }
  }

  if ( correct ) {
    std::cout << "\t result -- PASS\n\n";
  } else {
    std::cout << "\t result -- FAIL\n\n";
  }

}
