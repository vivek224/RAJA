/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining an alias specifying view class.
 *
 ******************************************************************************
 */

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

#ifndef RAJA_RESTRICTVIEW_HPP
#define RAJA_RESTRICTVIEW_HPP

#include <type_traits>

#include "RAJA/config.hpp"

namespace RAJA
{

template < typename T, long rclass = 0, bool is_integral = std::is_integral<T>::value >
struct RestrictValue {
  using value_type = T;
  constexpr static long restrict_class = rclass;

  value_type item;

  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue() : item() {}

  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(value_type const& v) : item(v) {}
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(value_type&& v) : item(std::move(v)) {}

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue<T2, r2> const& v) : item(v.item) {}
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue const& v) : item(v.item) {}

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue<T2, r2>&& v) : item(std::move(v.item)) {}
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue&& v) : item(std::move(v.item)) {}

  // unary operators
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue operator+() const { return RestrictValue{+item}; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue operator-() const { return RestrictValue{-item}; }

  // assignment operators
  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator= (RestrictValue<T2, r2> const& v) { item =  v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator= (RestrictValue const& v        ) { item =  v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator+=(RestrictValue<T2, r2> const& v) { item += v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator+=(RestrictValue const&         v) { item += v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator-=(RestrictValue<T2, r2> const& v) { item -= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator-=(RestrictValue const&         v) { item -= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator*=(RestrictValue<T2, r2> const& v) { item *= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator*=(RestrictValue const&         v) { item *= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator/=(RestrictValue<T2, r2> const& v) { item /= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator/=(RestrictValue const&         v) { item /= v.item; return *this; }

  RAJA_HOST_DEVICE RAJA_INLINE operator T() const { return item; }

  RAJA_HOST_DEVICE RAJA_INLINE ~RestrictValue() {}
};

// specialization for integral types
template < typename T, long rclass >
struct RestrictValue<T, rclass, true> {
  using value_type = T;
  constexpr static long restrict_class = rclass;

  value_type item;

  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue() : item() {}

  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(value_type const& v) : item(v) {}
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(value_type&& v) : item(std::move(v)) {}

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue<T2, r2> const& v) : item(v.item) {}
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue const& v) : item(v.item) {}

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue<T2, r2>&& v) : item(std::move(v.item)) {}
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue(RestrictValue&& v) : item(std::move(v.item)) {}

  // unary operators
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue operator+() const { return RestrictValue{+item}; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue operator-() const { return RestrictValue{-item}; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue operator~() const { return RestrictValue{~item}; }

  // assignment operators
  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator= (RestrictValue<T2, r2> const& v) { item =  v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator= (RestrictValue const& v        ) { item =  v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator+=(RestrictValue<T2, r2> const& v) { item += v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator+=(RestrictValue const&         v) { item += v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator-=(RestrictValue<T2, r2> const& v) { item -= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator-=(RestrictValue const&         v) { item -= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator*=(RestrictValue<T2, r2> const& v) { item *= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator*=(RestrictValue const&         v) { item *= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator/=(RestrictValue<T2, r2> const& v) { item /= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator/=(RestrictValue const&         v) { item /= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator%=(RestrictValue<T2, r2> const& v) { item %= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator%=(RestrictValue const&         v) { item %= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator&=(RestrictValue<T2, r2> const& v) { item &= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator&=(RestrictValue const&         v) { item &= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator|=(RestrictValue<T2, r2> const& v) { item |= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator|=(RestrictValue const&         v) { item |= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator^=(RestrictValue<T2, r2> const& v) { item ^= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator^=(RestrictValue const&         v) { item ^= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator<<=(RestrictValue<T2, r2> const& v) { item <<= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator<<=(RestrictValue const&         v) { item <<= v.item; return *this; }

  template < typename T2, long r2 >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator>>=(RestrictValue<T2, r2> const& v) { item >>= v.item; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue& operator>>=(RestrictValue const&         v) { item >>= v.item; return *this; }

  RAJA_HOST_DEVICE RAJA_INLINE operator T() const { return item; }

  RAJA_HOST_DEVICE RAJA_INLINE ~RestrictValue() {}
};

}  // namespace RAJA


// binary arithmetic operators
template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator+(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item + rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator-(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item - rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator*(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item * rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator/(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item / rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator%(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item % rhs.item}; }

// binary comparison operators
template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE bool operator==(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return lhs.item == rhs.item; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE bool operator!=(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return lhs.item != rhs.item; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE bool operator< (RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return lhs.item <  rhs.item; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE bool operator> (RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return lhs.item >  rhs.item; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE bool operator<=(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return lhs.item <= rhs.item; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE bool operator>=(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return lhs.item >= rhs.item; }

// binary bitwise operators

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator&(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item & rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator|(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item | rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator^(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item ^ rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator<<(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item << rhs.item}; }

template < typename T1, long r1, typename T2, long r2 >
RAJA_HOST_DEVICE RAJA_INLINE RAJA::RestrictValue<T1, r1> operator>>(RAJA::RestrictValue<T1, r1> const& lhs, RAJA::RestrictValue<T2, r2> const& rhs)
{ return RAJA::RestrictValue<T1, r1>{lhs.item >> rhs.item}; }


namespace RAJA
{

template < typename T, long rclass = 0 >
class RestrictView
{
public:
  using value_type = T;
  constexpr static long restrict_class = rclass;

  using base_type = T;
  using RestrictValue_type = RestrictValue<T, restrict_class>;
  using base_ptr_type = base_type*;
  using RestrictValue_ptr_type = RestrictValue_type*;

  RAJA_HOST_DEVICE RAJA_INLINE RestrictView() : r_ptr(nullptr) {}

  // construction from T*
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView(base_ptr_type const& ptr) : r_ptr((RestrictValue_ptr_type)ptr) {}

  // construction from ritem*
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView(RestrictValue_ptr_type const& ptr) : r_ptr(ptr) {}

  // copy construction from RestrictView
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView(RestrictView const& o) : r_ptr(o.r_ptr) {}
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView(RestrictView&& o) : r_ptr(o.r_ptr) {}

  // assignment from T*
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView& operator=(base_ptr_type const& rhs) { r_ptr = (RestrictValue_ptr_type)rhs; return *this; }

  // assignment from RestrictValue*
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView& operator=(RestrictValue_ptr_type const& rhs) { r_ptr = rhs; return *this; }

  // copy assignment from RestrictView
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView& operator=(RestrictView const& rhs) { r_ptr = rhs.r_ptr; return *this; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView& operator=(RestrictView&& rhs) { r_ptr = rhs.r_ptr; return *this; }

  // binary assignment operators
  template < typename I >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView& operator+=(I const& rhs) { r_ptr += rhs; return *this; }

  template < typename I >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView& operator-=(I const& rhs) { r_ptr -= rhs; return *this; }

  // binary operators
  template < typename I >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView operator+(I const& rhs) const { return {r_ptr + rhs}; }

  template < typename I >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictView operator-(I const& rhs) const { return {r_ptr - rhs}; }

  // indirection operator
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue_type& operator*() const { return *r_ptr; }

  // index operator
  template < typename I >
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue_type& operator[](T const& i) const { return r_ptr[i]; }

  // explicit conversion to RestrictValue ptr type
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue_ptr_type& get() { return r_ptr; }
  RAJA_HOST_DEVICE RAJA_INLINE RestrictValue_ptr_type const& get() const { return r_ptr; }

  // explicit conversion to base ptr type
  RAJA_HOST_DEVICE RAJA_INLINE base_ptr_type get_base() const { return reinterpret_cast<base_ptr_type>(r_ptr); }

  // implicit conversion to base ptr type (for compatibility)
  RAJA_HOST_DEVICE RAJA_INLINE operator base_ptr_type() const { return reinterpret_cast<base_ptr_type>(r_ptr); }

  RAJA_HOST_DEVICE RAJA_INLINE ~RestrictView() {}
private:
  RestrictValue_ptr_type r_ptr;

  static_assert(sizeof(RestrictValue_type) == sizeof(base_type),
      "Size of RestrictValue differs from size of base type");
  static_assert(alignof(RestrictValue_type) == alignof(base_type),
      "Alignment of RestrictValue differs from alignment of base type");
  static_assert(alignof(RestrictValue_type) == alignof(base_type),
      "Alignment of RestrictValue differs from alignment of base type");
  static_assert(std::is_standard_layout<RestrictValue_type>::value == std::is_standard_layout<base_type>::value,
      "layout of RestrictValue differs from layout of base type");
};

}  // namespace RAJA

#endif /* RAJA_RESTRICTVIEW_HPP */
