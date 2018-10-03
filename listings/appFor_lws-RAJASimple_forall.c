// --sched. strategy library and variables --
#include "vSched.h"
// in the below macros, strat is how we specify the library
#define FORALL_BEGIN(strat, s,e, start, end, tid, numThds )  loop_start_ ## strat (s,e ,&start, &end, tid, numThds);
do { 
#define FORALL_END(strat, start, end, tid)  } while( loop_next_ ## strat (&start, &end, tid));

/// OpenMP parallel lws policy implementation
template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const omp_lws<&, Iterable&& iter, Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
  int startInd, endInd;
  int threadNum = omp_get_thread_num();
  int numThreads = omp_get_num_threads();  
  FORALL_BEGIN(statdynstaggered, 0, distance_it, startInd, endInd, threadNum, numThreads)
  for (decltype(distance_it) i = startInd; i < endInd; ++i) {
    loop_body(begin_it[i]);
  }
  FORALL_END(statdynstaggered, startInd, endInd, threadNum)
}
