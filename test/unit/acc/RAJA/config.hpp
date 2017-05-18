/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for basic RAJA configuration options.
 *
 *          This file contains platform-specific parameters that control
 *          aspects of compilation of application code using RAJA. These
 *          parameters specify: SIMD unit width, data alignment information,
 *          inline directives, etc.
 *
 *          IMPORTANT: These options are set by CMake and depend on the options
 *          passed to it.
 *
 *          IMPORTANT: Exactly one e RAJA_COMPILER_* option must be defined to
 *          ensure correct behavior.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_config_HXX
#define RAJA_config_HXX

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

#define RAJA_USE_DOUBLE
#define RAJA_USE_RESTRICT_PTR

/*
 * Programming models
 */

#define RAJA_ENABLE_OPENMP 1
#define RAJA_ENABLE_NESTED 1

/*
 * Timer options
 */

#define RAJA_USE_CHRONO 1

/*
 * Detect the host C++ compiler we are using.
 */
#define RAJA_COMPILER_PGI

namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief RAJA software version number.
 *
 ******************************************************************************
 */
  const int RAJA_VERSION_MAJOR = 0;
  const int RAJA_VERSION_MINOR = 2;
  const int RAJA_VERSION_PATCHLEVEL = 5;


/*!
 ******************************************************************************
 *
 * \brief Useful macros.
 *
 ******************************************************************************
 */

//
//  Platform-specific constants for range index set and data alignment:
//
//     RANGE_ALIGN - alignment of begin/end indices in range segments
//                   (i.e., starting index and length of range segments
//                    constructed by index set builder methods will
//                    be multiples of this value)
//
//     RANGE_MIN_LENGTH - used in index set builder methods
//                        as min length of range segments (an integer multiple
//                        of RANGE_ALIGN)
//
//     DATA_ALIGN - used in compiler-specific intrinsics and typedefs
//                  to specify alignment of data, loop bounds, etc.;
//                  units of "bytes"

  const int RANGE_ALIGN = 4;
  const int RANGE_MIN_LENGTH = 8;
  const int DATA_ALIGN = 64;
  const int COHERENCE_BLOCK_SIZE = 64;

#if defined (_WIN32)
#define RAJA_RESTRICT __restrict
#else
#define RAJA_RESTRICT __restrict__
#endif


//
//  Compiler-specific definitions for inline directives, data alignment
//  intrinsics, and SIMD vector pragmas
//
//  Variables for compiler instrinsics, directives, typedefs
//
//     RAJA_INLINE - macro to enforce method inlining
//
//     RAJA_ALIGN_DATA(<variable>) - macro to express alignment of data,
//                              loop bounds, etc.
//
//     RAJA_SIMD - macro to express SIMD vectorization pragma to force
//                 loop vectorization
//
//     RAJA_ALIGNED_ATTR(<alignment>) - macro to express type or variable alignments
//

#if defined(RAJA_COMPILER_GNU) || defined(RAJA_COMPILER_PGI)
#define RAJA_ALIGNED_ATTR(N) __attribute__((aligned(N)))
#else
#define RAJA_ALIGNED_ATTR(N) alignas(N)
#endif


#if defined(RAJA_COMPILER_ICC)
//
// Configuration options for Intel compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(RAJA_ENABLE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else

#if __ICC < 1300  // use alignment intrinsic
#define RAJA_ALIGN_DATA(d) __assume_aligned(d, DATA_ALIGN)
#else
#define RAJA_ALIGN_DATA(d)  // TODO: Define this...
#endif

#endif

#define RAJA_SIMD  // TODO: Define this...


#elif defined(RAJA_COMPILER_GNU) || defined (RAJA_COMPILER_PGI)
//
// Configuration options for GNU compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(RAJA_ENABLE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else

#define RAJA_ALIGN_DATA(d) __builtin_assume_aligned(d, DATA_ALIGN)

#endif

#define RAJA_SIMD  // TODO: Define this...


#elif defined(RAJA_COMPILER_XLC)
//
// Configuration options for xlc compiler (i.e., bgq/sequoia).
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#define RAJA_ALIGN_DATA(d) __alignx(DATA_ALIGN, d)

//#define RAJA_SIMD  _Pragma("simd_level(10)")
#define RAJA_SIMD   // TODO: Define this...


#elif defined(RAJA_COMPILER_CLANG)
//
// Configuration options for clang compilers
//

#define RAJA_INLINE inline  __attribute__((always_inline))

#if defined(RAJA_ENABLE_CUDA)
#define RAJA_ALIGN_DATA(d)
#else

#define RAJA_ALIGN_DATA(d) // TODO: Define this...

#endif

#define RAJA_SIMD  // TODO: Define this...

#else

#pragma message("RAJA_COMPILER unknown, using default empty macros.")

#define RAJA_INLINE inline
#define RAJA_ALIGN_DATA(d)
#define RAJA_SIMD

#endif

}  // closing brace for RAJA namespace

#endif // closing endif for header file include guard
