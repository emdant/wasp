// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef CONTAINERS_BITMAP_H_
#define CONTAINERS_BITMAP_H_

#include <cstddef>
#include <cstdint>

#define INT_BITS 32
#define INT_LOG 5

class bitmap {
public:
  explicit bitmap(std::size_t n) : bits(new unsigned int[(n + INT_BITS - 1) >> INT_LOG]()) {}

  inline bool test(std::size_t i) const {
    return ((bits[i >> INT_LOG] >> i) & 1) != 0;
  }

  inline void set(std::size_t i) {
    bits[i >> INT_LOG] |= 1 << (i & (INT_BITS - 1));
  }

  inline void reset(std::size_t i) {
    bits[i >> INT_LOG] &= ~(1 << (i & (INT_BITS - 1)));
  }

private:
  uint32_t* bits;
};

#endif
