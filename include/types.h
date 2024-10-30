// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef TYPES_H_
#define TYPES_H_

#include <type_traits>
#include <variant>

namespace detail {

template <typename T, typename... Types>
struct unique {
  using type = T;
};

template <
    template <class...> class Tuple, typename... TupleTypes,
    typename U, typename... Us>
struct unique<Tuple<TupleTypes...>, U, Us...>
    : std::conditional<
          (std::is_same_v<U, TupleTypes> || ...), unique<Tuple<TupleTypes...>, Us...>, unique<Tuple<TupleTypes..., U>, Us...>>::type {};

} // namespace detail

template <typename... Types>
using unique_variant = typename detail::unique<std::variant<>, Types...>::type;

#endif