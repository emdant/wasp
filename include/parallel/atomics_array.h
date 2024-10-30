// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef PARALLEL_ATOMICS_ARRAY_H_
#define PARALLEL_ATOMICS_ARRAY_H_

#include <atomic>
#include <cstddef>

namespace parallel {

template <typename T>
class atomics_array {
private:
  std::size_t size_;
  std::atomic<T>* data_;

public:
  atomics_array(std::size_t n) : size_(n), data_(new std::atomic<T>[size_]) {}

  atomics_array(std::size_t n, T init_value) : size_(n), data_(new std::atomic<T>[size_]) {
#pragma omp parallel for
    for (size_t i = 0; i < size_; i++) {
      data_[i] = init_value;
    }
  }

  ~atomics_array() {
    delete[] data_;
  }

  std::atomic<T>& operator[](std::size_t index) { return data_[index]; }
  const std::atomic<T>& operator[](std::size_t index) const { return data_[index]; }

  std::atomic<T>* begin() { return data_; }
  std::atomic<T>* end() { return data_ + size_; }
  const std::atomic<T>* begin() const { return data_; }
  const std::atomic<T>* end() const { return data_ + size_; }
};

}; // namespace parallel

#endif