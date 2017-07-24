#include "RAJA/RAJA.hpp"
#include "RAJA/util/ColorSetBuilder.hpp"

#include <cstdio>

int main()
{

  puts("Iteration for extents={{4, 4}}");
  {
    auto Result = RAJA::make_colorset(std::array<RAJA::Index_type, 2>{{4, 4}});
    for (auto i : Result) {
      auto seg = Result.getSegment<RAJA::TypedListSegment<RAJA::Index_type>>(i);
      printf("Segment %lu:\n", i);
      RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type idx) {
        printf("%lu ", idx);
      });
      putchar('\n');
    }
  }
  puts("Iteration for extents={{6, 6}}, lower={{1, 1}}, upper={{5, 5}}");
  {
    auto Result = RAJA::make_colorset(std::array<RAJA::Index_type, 2>{{6, 6}},
                                std::array<RAJA::Index_type, 2>{{1, 1}},
                                std::array<RAJA::Index_type, 2>{{5, 5}});
    for (auto i : Result) {
      auto seg = Result.getSegment<RAJA::TypedListSegment<RAJA::Index_type>>(i);
      printf("Segment %lu:\n", i);
      RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type idx) {
        printf("%lu ", idx);
      });
      putchar('\n');
    }
  }
  puts("Iteration for extents={{5}}, lower={{1}}, upper={{4}}");
  {
    auto Result = RAJA::make_colorset(std::array<RAJA::Index_type, 1>{{5}},
                                      std::array<RAJA::Index_type, 1>{{1}},
                                      std::array<RAJA::Index_type, 1>{{4}});
    for (auto i : Result) {
      auto seg = Result.getSegment<RAJA::TypedListSegment<RAJA::Index_type>>(i);
      printf("Segment %lu:\n", i);
      RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type idx) {
        printf("%lu ", idx);
      });
      putchar('\n');
    }
  }
  puts("Iteration for extents={{5, 5}}, lower={{1, 1}}, upper={{4, 4}}");
  {
    auto Result = RAJA::make_colorset(std::array<RAJA::Index_type, 2>{{5, 5}},
                                      std::array<RAJA::Index_type, 2>{{1, 1}},
                                      std::array<RAJA::Index_type, 2>{{4, 4}});
    for (auto i : Result) {
      auto seg = Result.getSegment<RAJA::TypedListSegment<RAJA::Index_type>>(i);
      printf("Segment %lu:\n", i);
      RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type idx) {
        printf("%lu ", idx);
      });
      putchar('\n');
    }
  }
  puts("Iteration for extents={{5, 5, 5}}, lower={{1, 1, 1}}, upper={{4, 4, 4}}");
  {
    auto Result = RAJA::make_colorset(std::array<RAJA::Index_type, 3>{{5, 5, 5}},
                                      std::array<RAJA::Index_type, 3>{{1, 1, 1}},
                                      std::array<RAJA::Index_type, 3>{{4, 4, 4}});
    for (auto i : Result) {
      auto seg = Result.getSegment<RAJA::TypedListSegment<RAJA::Index_type>>(i);
      printf("Segment %lu:\n", i);
      RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type idx) {
        printf("%lu ", idx);
      });
      putchar('\n');
    }
  }
  puts("Iteration for extents={{5, 5, 5, 5}}, lower={{1, 1, 1, 1}}, upper={{4, 4, 4, 4}}");
  {
    auto Result = RAJA::make_colorset(std::array<RAJA::Index_type, 4>{{5, 5, 5, 5}},
                                      std::array<RAJA::Index_type, 4>{{1, 1, 1, 1}},
                                      std::array<RAJA::Index_type, 4>{{4, 4, 4, 4}});
    for (auto i : Result) {
      auto seg = Result.getSegment<RAJA::TypedListSegment<RAJA::Index_type>>(i);
      printf("Segment %lu:\n", i);
      RAJA::forall<RAJA::seq_exec>(seg, [=](RAJA::Index_type idx) {
        printf("%lu ", idx);
      });
      putchar('\n');
    }
  }
}
