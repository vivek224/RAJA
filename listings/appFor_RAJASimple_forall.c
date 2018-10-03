// OpenMP parallel lws policy implementation                                                              
template <typename Iterable, typename Func>
  RAJA_INLINE void forall_impl(const omp_for<&, Iterable&& iter, Func&& loop_body)
{
  RAJA_EXTRACT_BED_IT(iter);
#pragma omp for
    for (decltype(distance_it) i = 0; i < distance_it; ++i) {
      loop_body(begin_it[i]);
    }
}
