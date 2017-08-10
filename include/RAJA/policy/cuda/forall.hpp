/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via CUDA kernel launch.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cuda_HPP
#define RAJA_forall_cuda_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/forall.hpp"


#if defined(RAJA_ENABLE_CUDA)

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

#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

#include <algorithm>
#include <type_traits>

namespace RAJA
{

namespace cuda
{

enum struct GridStrideMode : int {
  disabled,
  static_size,
  occupancy_size
};

RAJA_INLINE
GridStrideMode& getGridStrideMode()
{
  static GridStrideMode mode = GridStrideMode::disabled;
  return mode;
}

namespace impl
{

RAJA_INLINE
int getNumSm(int device)
{
  int numSm;
  cudaErrchk(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, device));
  return numSm;
}

template <typename Func>
RAJA_INLINE
int getNumBlocksPerSm(Func func, size_t block_size, size_t dynSmem)
{
  int numBlocksPerSm;
  cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &numBlocksPerSm, func, block_size, dynSmem, cudaOccupancyDefault));
  return numBlocksPerSm;
}

/*!
 ******************************************************************************
 *
 * \brief calculate gridDim from length of iteration and blockDim
 *
 ******************************************************************************
 */
template <typename Func>
RAJA_INLINE
dim3 getGridDim(Func func, size_t len, dim3 blockDim, size_t dynSmem)
{
  size_t block_size = blockDim.x * blockDim.y * blockDim.z;
  size_t gridSize = (len + block_size-1) / block_size;

  if (getGridStrideMode() != GridStrideMode::disabled) {
    static int numSm = getNumSm(0);

    int numBlocksPerSm = 1;
    if (getGridStrideMode() == GridStrideMode::static_size) {
      numBlocksPerSm = (2048/block_size);
    } else if (getGridStrideMode() == GridStrideMode::occupancy_size) {
      static int bpsm = getNumBlocksPerSm(func, block_size, dynSmem);
      numBlocksPerSm = bpsm;
    } else {
      printf("Unknown grid stride mode\n");
    }

    size_t max_concurrent_blocks = numSm * numBlocksPerSm;

    return std::min(gridSize, max_concurrent_blocks);
  }

  return gridSize;
}

/*!
 ******************************************************************************
 *
 * \brief calculate global thread index from 1D grid of 1D blocks
 *
 ******************************************************************************
 */
__device__ __forceinline__ unsigned int getGlobalIdx_1D_1D()
{
  unsigned int blockId = blockIdx.x;
  unsigned int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned int getGlobalNumThreads_1D_1D()
{
  unsigned int numThreads = blockDim.x * gridDim.x;
  return numThreads;
}

/*!
 ******************************************************************************
 *
 * \brief calculate global thread index from 3D grid of 3D blocks
 *
 ******************************************************************************
 */
__device__ __forceinline__ unsigned int getGlobalIdx_3D_3D()
{
  unsigned int blockId =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  unsigned int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                          + (threadIdx.z * (blockDim.x * blockDim.y))
                          + (threadIdx.y * blockDim.x) + threadIdx.x;
  return threadId;
}
__device__ __forceinline__ unsigned int getGlobalNumThreads_3D_3D()
{
  unsigned int numThreads =
      blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
  return numThreads;
}

//
//////////////////////////////////////////////////////////////////////
//
// CUDA kernel templates.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall template for indirection array.
 *
 ******************************************************************************
 */
template <size_t BlockSize, typename Iterator, typename LOOP_BODY, typename IndexType>
__launch_bounds__ (BlockSize, 1)
__global__ void forall_cuda_kernel(LOOP_BODY loop_body,
                                   const Iterator idx,
                                   IndexType length)
{
  auto body = loop_body;
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
  if (ii < length) {
    body(idx[ii]);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall template for indirection array using grid-stride.
 *
 ******************************************************************************
 */
template <size_t BlockSize, typename Iterator, typename LOOP_BODY, typename IndexType>
__launch_bounds__ (BlockSize, 1)
__global__ void forall_cuda_kernel_gridstride(LOOP_BODY loop_body,
                                   const Iterator idx,
                                   IndexType length)
{
  auto body = loop_body;
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
  auto gridThreads = static_cast<IndexType>(getGlobalNumThreads_1D_1D());
  for (;ii < length;ii+=gridThreads) {
    body(idx[ii]);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall_Icount template for indiraction array.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BlockSize,
          typename Iterator,
          typename LoopBody,
          typename IndexType,
          typename IndexType2>
__launch_bounds__ (BlockSize, 1)
__global__ void forall_Icount_cuda_kernel(LoopBody loop_body,
                                          const Iterator idx,
                                          IndexType length,
                                          IndexType2 icount)
{
  auto body = loop_body;
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
  if (ii < length) {
    body(static_cast<IndexType>(ii + icount), idx[ii]);
  }
}

/*!
 ******************************************************************************
 *
 * \brief  CUDA kernal forall_Icount template for indiraction array using grid-stride.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <size_t BlockSize,
          typename Iterator,
          typename LoopBody,
          typename IndexType,
          typename IndexType2>
__launch_bounds__ (BlockSize, 1)
__global__ void forall_Icount_cuda_kernel_gridstride(LoopBody loop_body,
                                          const Iterator idx,
                                          IndexType length,
                                          IndexType2 icount)
{
  auto body = loop_body;
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
  auto gridThreads = static_cast<IndexType>(getGlobalNumThreads_1D_1D());
  for (;ii < length;ii+=gridThreads) {
    body(static_cast<IndexType>(ii + icount), idx[ii]);
  }
}

}  // closing brace for impl namespace

}  // closing brace for cuda namespace

namespace impl
{

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

template <typename Iterable, typename LoopBody, size_t BlockSize, bool Async>
RAJA_INLINE void forall(cuda_exec<BlockSize, Async>,
                        Iterable&& iter,
                        LoopBody&& loop_body)
{
  auto begin = std::begin(iter);
  auto end   = std::end(iter);

  auto len = std::distance(begin, end);

  if (len > 0 && BlockSize > 0) {

    size_t dynSmem = 0;
    cudaStream_t stream = 0;

    RAJA_FT_BEGIN;

    if (cuda::getGridStrideMode() == cuda::GridStrideMode::disabled) {
      auto func = cuda::impl::forall_cuda_kernel<
          BlockSize,
          typename std::remove_reference<decltype(begin)>::type,
          typename std::remove_reference<decltype(loop_body)>::type,
          typename std::remove_reference<decltype(len)>::type>;

      auto gridSize = cuda::impl::getGridDim(func, len, BlockSize, dynSmem);


      func<<<gridSize, BlockSize, dynSmem, stream>>>(
          cuda::make_launch_body(gridSize, BlockSize, dynSmem, stream,
                                 std::forward<LoopBody>(loop_body)),
          std::move(begin), len);

    } else {
      auto func = cuda::impl::forall_cuda_kernel_gridstride<
          BlockSize,
          typename std::remove_reference<decltype(begin)>::type,
          typename std::remove_reference<decltype(loop_body)>::type,
          typename std::remove_reference<decltype(len)>::type>;

      auto gridSize = cuda::impl::getGridDim(func, len, BlockSize, dynSmem);


      func<<<gridSize, BlockSize, dynSmem, stream>>>(
          cuda::make_launch_body(gridSize, BlockSize, dynSmem, stream,
                                 std::forward<LoopBody>(loop_body)),
          std::move(begin), len);

    }
    cuda::peekAtLastError();

    cuda::launch(stream);
    if (!Async) cuda::synchronize(stream);

    RAJA_FT_END;
  }
}


template <typename Iterable,
          typename IndexType,
          typename LoopBody,
          size_t BlockSize,
          bool Async>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(cuda_exec<BlockSize, Async>,
              Iterable&& iter,
              IndexType icount,
              LoopBody&& loop_body)
{
  auto begin = std::begin(iter);
  auto end   = std::end(iter);

  auto len = std::distance(begin, end);

  if (len > 0 && BlockSize > 0) {

    size_t dynSmem = 0;
    cudaStream_t stream = 0;

    RAJA_FT_BEGIN;

    if (cuda::getGridStrideMode() == cuda::GridStrideMode::disabled) {
      auto func = cuda::impl::forall_Icount_cuda_kernel<
          BlockSize,
          typename std::remove_reference<decltype(begin)>::type,
          typename std::remove_reference<decltype(loop_body)>::type,
          typename std::remove_reference<decltype(len)>::type,
          typename std::remove_reference<decltype(icount)>::type>;

      auto gridSize = cuda::impl::getGridDim(func, len, BlockSize, dynSmem);


      func<<<gridSize, BlockSize, dynSmem, stream>>>(
          cuda::make_launch_body(gridSize, BlockSize, dynSmem, stream,
                                 std::forward<LoopBody>(loop_body)),
          std::move(begin), len, icount);

    } else {
      auto func = cuda::impl::forall_Icount_cuda_kernel_gridstride<
          BlockSize,
          typename std::remove_reference<decltype(begin)>::type,
          typename std::remove_reference<decltype(loop_body)>::type,
          typename std::remove_reference<decltype(len)>::type,
          typename std::remove_reference<decltype(icount)>::type>;

      auto gridSize = cuda::impl::getGridDim(func, len, BlockSize, dynSmem);


      func<<<gridSize, BlockSize, dynSmem, stream>>>(
          cuda::make_launch_body(gridSize, BlockSize, dynSmem, stream,
                                 std::forward<LoopBody>(loop_body)),
          std::move(begin), len, icount);

    }
    cuda::peekAtLastError();

    cuda::launch(stream);
    if (!Async) cuda::synchronize(stream);

    RAJA_FT_END;
  }
}

//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as CUDA kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         CUDA execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE void forall(
    ExecPolicy<seq_segit, cuda_exec<BlockSize, Async>>,
    const StaticIndexSet<SegmentTypes...>& iset,
    LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(isi,
                     CallForall(),
                     cuda_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) cuda::synchronize();
}


/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         CUDA execution for segments.
 *
 *         This method passes index count to segment iteration.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE void forall_Icount(
    ExecPolicy<seq_segit, cuda_exec<BlockSize, Async>>,
    const StaticIndexSet<SegmentTypes...>& iset,
    LoopBody&& loop_body)
{
  auto num_seg = iset.getNumSegments();
  for (decltype(num_seg) isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(isi,
                     CallForallIcount(iset.getStartingIcount(isi)),
                     cuda_exec<BlockSize, true>(),
                     loop_body);

  }  // iterate over segments of index set

  if (!Async) cuda::synchronize();
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
