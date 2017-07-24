#ifndef RAJA_ColorSet_Builder_HPP_
#define RAJA_ColorSet_Builder_HPP_

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"

namespace RAJA
{

namespace detail
{

namespace impl
{

template <typename Array>
constexpr unsigned long countElements(const Array &,
                                      const Array &,
                                      VarOps::index_sequence<>)
{
  return 1;
}

template <typename Array, size_t Curr, size_t... Ids>
constexpr unsigned long countElements(const Array &lower,
                                      const Array &upper,
                                      VarOps::index_sequence<Curr, Ids...>)
{
  using Seq = VarOps::index_sequence<Ids...>;
  return (upper[Curr] - lower[Curr]) * countElements(lower, upper, Seq());
}

template <typename Array>
constexpr unsigned long maxBufferSizeRequired(const Array &,
                                              const Array &,
                                              VarOps::index_sequence<>)
{
  return 1;
}

template <typename Array, size_t Curr, size_t... Ids>
constexpr unsigned long maxBufferSizeRequired(
    const Array &lower,
    const Array &upper,
    VarOps::index_sequence<Curr, Ids...>)
{
  using Seq = VarOps::index_sequence<Ids...>;
  return (upper[Curr] - lower[Curr] + 1) / 2
         * maxBufferSizeRequired(lower, upper, Seq());
}

template <typename Array>
constexpr unsigned long computeOffset(const Array &,
                                      const Array &,
                                      VarOps::index_sequence<>)
{
  return 0;
}

template <typename Array, size_t Curr, size_t... Ids>
constexpr unsigned long computeOffset(const Array &offsets,
                                      const Array &inds,
                                      VarOps::index_sequence<Curr, Ids...>)
{
  using Seq = VarOps::index_sequence<Ids...>;
  return (inds[Curr] * offsets[Curr]) + computeOffset(offsets, inds, Seq());
}

template <typename Array, typename IndexType, typename... Offsets>
constexpr const Array make_offsets_for(const Array &,
                                       IndexType,
                                       VarOps::index_sequence<>,
                                       Offsets... offsets)
{
  return Array{{offsets...}};
}

template <typename Array,
          typename IndexType,
          size_t Curr,
          size_t... Rest,
          typename... Offsets>
constexpr const Array make_offsets_for(const Array &extents,
                                       IndexType factor,
                                       VarOps::index_sequence<Curr, Rest...>,
                                       Offsets... offsets)
{
  return make_offsets_for(extents,
                          factor * std::get<Curr>(extents),
                          VarOps::index_sequence<Rest...>(),
                          factor,
                          offsets...);
}

}  // closing brace for impl namespace


template <typename>
struct reverse;

/// reverse an @index_sequence
template <size_t... Ids>
struct reverse<VarOps::index_sequence<Ids...>> {
  using type = VarOps::index_sequence<((sizeof...(Ids) - 1) - Ids)...>;
};

/// type helper alias for reversing @index_sequence
template <typename T>
using reverse_t = typename reverse<T>::type;

/// gets number of elements in [@lower, @upper) ND-range
template <typename Array>
constexpr unsigned long countElements(const Array &lower, const Array &upper)
{
  using Seq = VarOps::make_index_sequence<std::tuple_size<Array>::value>;
  return impl::countElements(lower, upper, Seq());
}

/// determines the largest buffer required to compute a ListSegment space for
/// coloring in the [@lower, @upper) ND-range
template <typename Array>
constexpr unsigned long maxBufferSizeRequired(const Array &lower,
                                              const Array &upper)
{
  using Seq = VarOps::make_index_sequence<std::tuple_size<Array>::value>;
  return impl::maxBufferSizeRequired(lower, upper, Seq());
}

/// compute the distance @inds array is from virtual zero given @offset array
template <typename Array>
constexpr unsigned long computeOffset(const Array &offsets, const Array &inds)
{
  using Seq = VarOps::make_index_sequence<std::tuple_size<Array>::value>;
  return impl::computeOffset(offsets, inds, Seq());
}

/// create an offsets array from an @extents array
template <typename Array>
constexpr const Array make_offsets_for(const Array &extents)
{
  using IndexType = typename Array::value_type;
  using Sequence = VarOps::make_index_sequence<std::tuple_size<Array>::value>;
  using Reversed = reverse_t<Sequence>;
  return impl::make_offsets_for(extents, static_cast<IndexType>(1), Reversed());
}

template <typename, typename, typename IndexType>
struct ColorSet_BuildList;

template <template <size_t...> class Vals,
          size_t Curr,
          size_t... Rest,
          size_t... All,
          typename IndexType>
struct ColorSet_BuildList<Vals<Curr, Rest...>, Vals<All...>, IndexType> {
  using Next = ColorSet_BuildList<Vals<Rest...>, Vals<All...>, IndexType>;
  using Buffer = IndexType *;

  template <typename Array, typename... Idx>
  static inline void generate(Buffer &buffer,
                              const Array &offsets,
                              const Array &lower,
                              const Array &upper,
                              const Array &&dim,
                              const Idx... ids)
  {
    for (auto i = lower[Curr] + dim[Curr]; i < upper[Curr]; i += 2) {
      Next::generate(buffer,
                     offsets,
                     lower,
                     upper,
                     std::forward<const Array>(dim),
                     ids...,
                     i);
    }
  }
};

template <template <size_t...> class Vals, size_t... All, typename IndexType>
struct ColorSet_BuildList<Vals<>, Vals<All...>, IndexType> {
  using Next = ColorSet_BuildList<Vals<All...>, Vals<All...>, IndexType>;
  using Buffer = IndexType *;

  template <typename Array, typename... Idx>
  static inline void generate(Buffer &buffer,
                              const Array &offsets,
                              const Array &,
                              const Array &,
                              const Array &&,
                              const Idx &... ids)
  {
    *buffer = computeOffset(offsets, Array{{ids...}}), ++buffer;
  }
};

template <typename, typename, typename IndexType>
struct ColorSet_EachColor;

template <template <size_t...> class Vals,
          size_t Curr,
          size_t... Rest,
          size_t... All,
          typename IndexType>
struct ColorSet_EachColor<Vals<Curr, Rest...>, Vals<All...>, IndexType> {
  using Next = ColorSet_EachColor<Vals<Rest...>, Vals<All...>, IndexType>;
  using Buffer = IndexType *;

  template <typename IdxSet, typename Array, typename... Idx>
  static inline void generate(IdxSet &iset,
                              Buffer buffer,
                              const Array &offsets,
                              const Array &lower,
                              const Array &upper,
                              const Idx &... ids)
  {
    for (auto i : {0, 1}) {
      Next::generate(iset, buffer, offsets, lower, upper, ids..., i);
    }
  }
};

template <template <size_t...> class Vals, size_t... All, typename IndexType>
struct ColorSet_EachColor<Vals<>, Vals<All...>, IndexType> {
  using Next = ColorSet_BuildList<Vals<All...>, Vals<All...>, IndexType>;
  using Buffer = IndexType *;

  template <typename IdxSet, typename Array, typename... Idx>
  static inline void generate(IdxSet &iset,
                              Buffer buffer,
                              const Array &offsets,
                              const Array &lower,
                              const Array &upper,
                              const Idx &... ids)
  {
    Buffer currentBuffer = buffer;
    Next::generate(currentBuffer, offsets, lower, upper, Array{{ids...}});
    using ListSeg = RAJA::TypedListSegment<IndexType>;
    iset.push_back(ListSeg(buffer, std::distance(buffer, currentBuffer)));
  }
};

template <typename T>
struct TempBuffer {
  T *data;
  constexpr TempBuffer(unsigned long N) : data(new T[N]) {}
  ~TempBuffer()
  {
    delete[] data;
  }
  constexpr operator T *() const { return data; }
};

template <typename IdxSet, typename Array>
static inline IdxSet build_colorset(typename Array::value_type *buffer,
                                    const Array &offsets,
                                    const Array &lower,
                                    const Array &upper)
{
  using IndexType = typename Array::value_type;
  using Seq = VarOps::make_index_sequence<std::tuple_size<Array>::value>;
  using ColorSet = ColorSet_EachColor<Seq, Seq, IndexType>;
  IdxSet result;
  ColorSet::generate(result, buffer, offsets, lower, upper);
  return result;
}

}  // closing brace for detail namespace

/// create a ColorSet @IndexSet from @extents, @lower, and @upper bounds
/*!
 *  \param[in] extents ND-range specifying the extents in each dimension
 *  \param[in] lower ND-point specifying the lower iteration space limit
 * (inclusive)
 *  \param[in] upper ND-point specifying the upper iteration space limit
 * (exclusive)
 */
template <unsigned long N, typename IndexType>
RAJA::StaticIndexSet<RAJA::TypedListSegment<IndexType>> make_colorset(
    const std::array<IndexType, N> &extents,
    const std::array<IndexType, N> &lower,
    const std::array<IndexType, N> &upper)
{
  using IndexSet = RAJA::StaticIndexSet<RAJA::TypedListSegment<IndexType>>;
  IndexType count = detail::maxBufferSizeRequired(lower, upper);
  detail::TempBuffer<IndexType> buffer(count);
  const std::array<IndexType, N> offsets = detail::make_offsets_for(extents);
  return detail::build_colorset<IndexSet>(buffer, offsets, lower, upper);
}

/// create a ColorSet @IndexSet from @extents
/*!
 *  \param[in] extents ND-range specifying the extents in each dimension
 */
template <typename Array,
          typename IndexType = typename Array::value_type,
          size_t N = std::tuple_size<Array>::value>
RAJA::StaticIndexSet<RAJA::TypedListSegment<IndexType>> make_colorset(
    const Array &extents)
{
  return make_colorset(extents, Array{}, extents);
}

}  // closing brace for RAJA namespace

#endif
