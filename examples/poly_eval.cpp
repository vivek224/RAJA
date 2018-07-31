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
#include "RAJA/util/RestrictView.hpp"


#include "memoryManager.hpp"

template < typename T, size_t N >
struct Array {
  T arr[N];
  template < typename I >
  RAJA_HOST_DEVICE RAJA_INLINE
  T& operator[](I&& i) { return arr[std::forward<I>(i)]; }
  template < typename I >
  RAJA_HOST_DEVICE RAJA_INLINE
  T const& operator[](I&& i) const { return arr[std::forward<I>(i)]; }
};

/*
  CUDA_BLOCK_SIZE - specifies the number of threads in a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

//
// By default a RAJA::Index_type
// is a long int
//
using RAJA::Index_type;

//
//Function for checking results
//
template <typename T0, typename T1, typename T2>
void checkResult(T0* Y, T1 const& coef, T2* X, Index_type N);

int main(int argc, char **argv)
{

  std::cout << "\n\nRAJA polynomial evaluation example...\n";

// Dimensions of matrices
  Array<double, 6> coef {
    1.2,
    -0.54,
    1.24,
    -2.53,
    2.41,
    5.132
  };

// Number of matrices
  Index_type N = 8000000;
  if (argc >= 2) N = atol(argv[1]);

// Number of iterations
  int NITER = 20;
  if (argc >= 3) NITER = atoi(argv[2]);

  std::cout << "\n Number of polynomials to be evaluated: " << N << " \n \n";

//
// Initialize a RAJA timer object
// and variable to store minimum run time
//
  auto timer = RAJA::Timer();
  double minRun;

//
// Allocate space for data
//
  double *data = memoryManager::allocate<double>(N + N);

  double *Y = data;
  double *X = data + N;

  RAJA::RestrictValue<double, 1> *Y2 = (RAJA::RestrictValue<double, 1>*)Y;
  RAJA::RestrictValue<double, 2> *X2 = (RAJA::RestrictValue<double, 2>*)X;

//
// Initialize data
//
#if defined(RAJA_ENABLE_OPENMP)
  using INIT_POL = RAJA::omp_parallel_for_exec;
#else
  using INIT_POL = RAJA::loop_exec;
#endif

  RAJA::forall<INIT_POL>(
      RAJA::TypedRangeSegment<Index_type>(0, N), [=](Index_type i) {
    Y[i] = 0.0;
    X[i] = 0.0;
  });

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " manually optimized (c style - sequential) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    for (Index_type i = 0; i < N; ++i) {

      double y = 0.0;
      y += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[2] * ( X[i] * X[i] * X[i] ) ;
      y += coef[3] * ( X[i] * X[i] ) ;
      y += coef[4] * ( X[i] ) ;
      y += coef[5] ;
      Y[i] = y;

    };
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " unoptimized (c style - sequential) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    for (Index_type i = 0; i < N; ++i) {

      Y[i] = 0.0;
      Y[i] += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[2] * ( X[i] * X[i] * X[i] ) ;
      Y[i] += coef[3] * ( X[i] * X[i] ) ;
      Y[i] += coef[4] * ( X[i] ) ;
      Y[i] += coef[5] ;

    };
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " rclass optimized (c style - sequential) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    for (Index_type i = 0; i < N; ++i) {

      Y2[i] = 0.0;
      Y2[i] += coef[0] * ( X2[i] * X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[1] * ( X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[2] * ( X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[3] * ( X2[i] * X2[i] ) ;
      Y2[i] += coef[4] * ( X2[i] ) ;
      Y2[i] += coef[5] ;

    };
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " manually optimized (RAJA - sequential) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::loop_exec>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=](Index_type i) {

      double y = 0.0;
      y += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[2] * ( X[i] * X[i] * X[i] ) ;
      y += coef[3] * ( X[i] * X[i] ) ;
      y += coef[4] * ( X[i] ) ;
      y += coef[5] ;
      Y[i] = y;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " unoptimized (RAJA - sequential) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::loop_exec>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=](Index_type i) {

      Y[i] = 0.0;
      Y[i] += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[2] * ( X[i] * X[i] * X[i] ) ;
      Y[i] += coef[3] * ( X[i] * X[i] ) ;
      Y[i] += coef[4] * ( X[i] ) ;
      Y[i] += coef[5] ;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " rclass optimized (RAJA - sequential) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::loop_exec>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=](Index_type i) {

      Y2[i] = 0.0;
      Y2[i] += coef[0] * ( X2[i] * X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[1] * ( X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[2] * ( X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[3] * ( X2[i] * X2[i] ) ;
      Y2[i] += coef[4] * ( X2[i] ) ;
      Y2[i] += coef[5] ;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << " \n Performing polynomial evaluation"
            << " manually optimized (c style - omp parallel for) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    #pragma omp parallel for
    for (Index_type i = 0; i < N; ++i) {

      double y = 0.0;
      y += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[2] * ( X[i] * X[i] * X[i] ) ;
      y += coef[3] * ( X[i] * X[i] ) ;
      y += coef[4] * ( X[i] ) ;
      y += coef[5] ;
      Y[i] = y;

    };
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " unoptimized (c style - omp parallel for) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    #pragma omp parallel for
    for (Index_type i = 0; i < N; ++i) {

      Y[i] = 0.0;
      Y[i] += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[2] * ( X[i] * X[i] * X[i] ) ;
      Y[i] += coef[3] * ( X[i] * X[i] ) ;
      Y[i] += coef[4] * ( X[i] ) ;
      Y[i] += coef[5] ;

    };
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " rclass optimized (c style - omp parallel for) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    #pragma omp parallel for
    for (Index_type i = 0; i < N; ++i) {

      Y2[i] = 0.0;
      Y2[i] += coef[0] * ( X2[i] * X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[1] * ( X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[2] * ( X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[3] * ( X2[i] * X2[i] ) ;
      Y2[i] += coef[4] * ( X2[i] ) ;
      Y2[i] += coef[5] ;

    };
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
    
  std::cout << "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " manually optimized (RAJA - omp parallel for) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=](Index_type i) {

      double y = 0.0;
      y += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[2] * ( X[i] * X[i] * X[i] ) ;
      y += coef[3] * ( X[i] * X[i] ) ;
      y += coef[4] * ( X[i] ) ;
      y += coef[5] ;
      Y[i] = y;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " unoptimized (RAJA - omp parallel for) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=](Index_type i) {

      Y[i] = 0.0;
      Y[i] += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[2] * ( X[i] * X[i] * X[i] ) ;
      Y[i] += coef[3] * ( X[i] * X[i] ) ;
      Y[i] += coef[4] * ( X[i] ) ;
      Y[i] += coef[5] ;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " rclass optimized (RAJA - omp parallel for) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::omp_parallel_for_exec>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=](Index_type i) {

      Y2[i] = 0.0;
      Y2[i] += coef[0] * ( X2[i] * X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[1] * ( X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[2] * ( X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[3] * ( X2[i] * X2[i] ) ;
      Y2[i] += coef[4] * ( X2[i] ) ;
      Y2[i] += coef[5] ;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }
  
  std::cout<< "\trun time : " << minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << " \n Performing polynomial evaluation"
            << " manually optimized (RAJA - cuda) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=] RAJA_HOST_DEVICE (Index_type i) {

      double y = 0.0;
      y += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      y += coef[2] * ( X[i] * X[i] * X[i] ) ;
      y += coef[3] * ( X[i] * X[i] ) ;
      y += coef[4] * ( X[i] ) ;
      y += coef[5] ;
      Y[i] = y;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }

  std::cout<< "\trun time: "<< minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " unoptimized (RAJA - cuda) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=] RAJA_HOST_DEVICE (Index_type i) {

      Y[i] = 0.0;
      Y[i] += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ;
      Y[i] += coef[2] * ( X[i] * X[i] * X[i] ) ;
      Y[i] += coef[3] * ( X[i] * X[i] ) ;
      Y[i] += coef[4] * ( X[i] ) ;
      Y[i] += coef[5] ;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }

  std::cout<< "\trun time: "<< minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

//----------------------------------------------------------------------------//

  std::cout << " \n Performing polynomial evaluation"
            << " rclass optimized (RAJA - cuda) ... " << std::endl;

  minRun = std::numeric_limits<double>::max();
  for (int iter = 0; iter < NITER; ++iter) {

    timer.start();
    RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(
        RAJA::TypedRangeSegment<Index_type>(0, N), [=] RAJA_HOST_DEVICE (Index_type i) {

      Y2[i] = 0.0;
      Y2[i] += coef[0] * ( X2[i] * X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[1] * ( X2[i] * X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[2] * ( X2[i] * X2[i] * X2[i] ) ;
      Y2[i] += coef[3] * ( X2[i] * X2[i] ) ;
      Y2[i] += coef[4] * ( X2[i] ) ;
      Y2[i] += coef[5] ;

    });
    timer.stop();

    RAJA::Timer::ElapsedType tMin = timer.elapsed();
    if (tMin < minRun) minRun = tMin;
    timer.reset();
  }

  std::cout<< "\trun time: "<< minRun << " seconds" << std::endl;
  checkResult(Y, coef, X, N);

#endif

//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(data);

  std::cout << "\n DONE!...\n";
  return 0;
}

//
// check result
//
template <typename T0, typename T1, typename T2>
void checkResult(T0* Y, T1 const& coef, T2* X, Index_type N)
{

  bool status = true;
  for (int i = 0; i < N; ++i) {
    double y = 0.0;
    y += coef[0] * ( X[i] * X[i] * X[i] * X[i] * X[i] ) ; 
    y += coef[1] * ( X[i] * X[i] * X[i] * X[i] ) ; 
    y += coef[2] * ( X[i] * X[i] * X[i] ) ; 
    y += coef[3] * ( X[i] * X[i] ) ; 
    y += coef[4] * ( X[i] ) ;  
    y += coef[5] ; 
    if (std::abs(Y[i] - y) > 10e-12) {
      status = false;
    }
  }

  if ( status ) {
    std::cout << "\tresult -- PASS\n";
  } else {
    std::cout << "\tresult -- FAIL\n";
  }
}
