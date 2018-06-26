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

#define DOT_BODY \
   dot += x[i] ;
#else
#define DOT_BODY \
  dot += a[i];
#endif

using realType = double;
using TFloat = realType * const RAJA_RESTRICT;

RAJA_INLINE
void dot_noVec(TFloat a, double &dot, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif

  RAJA_NO_SIMD
  for(RAJA::Index_type i=0; i<N; ++i){
    DOT_BODY
  }
}

RAJA_INLINE
void dot_native(TFloat a, double &dot, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif

  for(RAJA::Index_type i=0; i<N; ++i){
    DOT_BODY;
  }

}

RAJA_INLINE
void dot_simd(TFloat a, double &dot, RAJA::Index_type N) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif
  RAJA_SIMD
  for(RAJA::Index_type i=0; i<N; ++i){
    DOT_BODY;
  }
  
}

template<typename POL, typename redPOL>
RAJA_INLINE
void dot_RAJA(TFloat a, double &rdot, RAJA::Index_type N)
{

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
#endif
  RAJA::ReduceSum<redPOL, double> dot(0.0);

  RAJA::forall<POL>(RAJA::RangeSegment(0, N), [=] (RAJA::Index_type i) {
      DOT_BODY
    });

  rdot = dot.get();
  
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
  std::cout << "\n\nRAJA vector addition benchmark with alignment hint...\n";
#else
  std::cout << "\n\nRAJA vector addition benchmark...\n";
#endif
  std::cout<<"No of entries "<<N<<"\n\n"<<std::endl;
  
  auto timer = RAJA::Timer();
  const RAJA::Index_type Niter = 1000000;

  double dot = 0.0;
  TFloat a = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*sizeof(realType));
  
  //Intialize data 
  for(RAJA::Index_type i=0; i<N; ++i)
    {
      a[i] = 1./N;
    }

  //---------------------------------------------------------
  std::cout<<"Native C - strictly sequential"<<std::endl;
  //---------------------------------------------------------
  timer.start();
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    dot = 0.0;
    dot_noVec(a, dot, N);
    
  }
  timer.stop();
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);


  //---------------------------------------------------------
  std::cout<<"Native C - raw loop"<<std::endl;
  //---------------------------------------------------------

  timer.start();
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    dot = 0.0;
    dot_native(a, dot, N);    
  }
  timer.stop();
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);

  //---------------------------------------------------------
  std::cout<<"Native C - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  timer.start();
  for(RAJA::Index_type it = 0; it < Niter; ++it){
    dot = 0.0;
    dot_simd(a, dot, N);    
  }
  timer.stop();
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);


  //---------------------------------------------------------
  std::cout<<"RAJA - strictly sequential"<<std::endl;
  //---------------------------------------------------------
  timer.start();
  for(RAJA::Index_type it = 0; it < Niter; ++it){

    dot_RAJA<RAJA::seq_exec, RAJA::seq_reduce>(a, dot, N);
    
  }
  timer.stop();
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);

  //---------------------------------------------------------
  std::cout<<"RAJA - raw loop"<<std::endl;
  //---------------------------------------------------------
  timer.start();
  for(RAJA::Index_type it = 0; it < Niter; ++it){

    dot_RAJA<RAJA::loop_exec, RAJA::seq_reduce>(a, dot, N);
    
  }
  timer.stop();
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);

  //---------------------------------------------------------
  std::cout<<"RAJA - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  timer.start();
  for(RAJA::Index_type it = 0; it < Niter; ++it){

    dot_RAJA<RAJA::simd_exec, RAJA::seq_reduce>(a, dot, N);
    
  }
  timer.stop();
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(dot, N);
  //---------------------------------------------------------

  std::cout << "\n DONE!...\n";
  return 0;
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
