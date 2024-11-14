// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef PARALLEL_PADDED_ARRAY_H_
#define PARALLEL_PADDED_ARRAY_H_

#include <algorithm>

#include "alignment.h"

namespace parallel {

template <typename T>
class padded_array {
public:
  static inline constexpr size_t stride() {
    if constexpr (sizeof(T) > hardware_destructive_interference_size) {
      return 1 + (sizeof(T) % hardware_destructive_interference_size != 0 ? 1 : 0);
    } else {
      constexpr int trunc = hardware_destructive_interference_size / sizeof(T);
      constexpr float f = hardware_destructive_interference_size / sizeof(T);
      return f > trunc ? trunc + 1 : trunc;
    }
  }

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;

    inline iterator(pointer ptr) : m_ptr(ptr) {}

    inline reference operator*() const { return *m_ptr; }
    inline pointer operator->() { return m_ptr; }

    // Prefix increment
    inline iterator& operator++() {
      m_ptr += stride();
      return *this;
    }

    // Postfix increment
    inline iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    inline friend bool operator==(const iterator& a, const iterator& b) { return a.m_ptr == b.m_ptr; };
    inline friend bool operator!=(const iterator& a, const iterator& b) { return a.m_ptr != b.m_ptr; };

  private:
    pointer m_ptr;
  };

  padded_array() : start_(nullptr), end_size_(nullptr) {}

  explicit padded_array(size_t num_elements) {
    start_ = new T[num_elements * stride()];
    end_size_ = start_ + (num_elements * stride());
  }

  padded_array(size_t num_elements, const T& init_val) : padded_array(num_elements) {
    fill(init_val);
  }

  padded_array(iterator copy_begin, iterator copy_end)
      : padded_array(copy_end - copy_begin) {
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < size(); i++)
      start_[i * stride()] = copy_begin[i * stride()];
  }

  // don't want this to be copied, too much data to move
  padded_array(const padded_array& other) = delete;

  // prefer move because too much data to copy
  padded_array(padded_array&& other)
      : start_(other.start_), end_size_(other.end_size_) {
    other.start_ = nullptr;
    other.end_size_ = nullptr;
  }

  // want move assignment
  padded_array& operator=(padded_array&& other) {
    if (this != &other) {
      release_resources();
      start_ = other.start_;
      end_size_ = other.end_size_;
      other.start_ = nullptr;
      other.end_size_ = nullptr;
    }
    return *this;
  }

  void release_resources() {
    if (start_ != nullptr) {
      delete[] start_;
    }
  }

  ~padded_array() {
    release_resources();
  }

  T& operator[](size_t n) {
    return start_[n * stride()];
  }

  const T& operator[](size_t n) const {
    return start_[n * stride()];
  }

  void fill(T init_val) {
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < size(); i++)
      start_[i * stride()] = init_val;
  }

  size_t size() const {
    return (end_size_ - start_) / stride();
  }

  iterator begin() const {
    return iterator(start_);
  }

  iterator end() const {
    return iterator(end_size_);
  }

  T* data() const {
    return start_;
  }

  void swap(padded_array& other) {
    std::swap(start_, other.start_);
    std::swap(end_size_, other.end_size_);
  }

private:
  T* start_;
  T* end_size_;
};

} // namespace parallel

#endif // PARALLEL_PADDED_ARRAY_H_