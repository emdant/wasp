// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef PARALLEL_PADDED_ARRAY_H_
#define PARALLEL_PADDED_ARRAY_H_

#include <algorithm>
#include <cstddef>

#include "alignment.h"

namespace parallel {

template <typename T>
class padded_array {
public:
  struct alignas(hardware_destructive_interference_size) padded_element {
    T element;
  };

  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = padded_element;
    using pointer = value_type*;
    using reference = T&;

    inline iterator(pointer ptr) : ptr_(ptr) {}

    inline reference operator*() const { return ptr_->element; }
    inline pointer operator->() { return ptr_; }

    // Prefix increment
    inline iterator& operator++() {
      ptr_++;
      return *this;
    }

    // Postfix increment
    inline iterator operator++(int) {
      iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    inline friend bool operator==(const iterator& a, const iterator& b) { return a.ptr_ == b.ptr_; };
    inline friend bool operator!=(const iterator& a, const iterator& b) { return a.ptr_ != b.ptr_; };
    inline difference_type operator-(const iterator& other) const {
      return ptr_ - other.ptr_;
    }
    inline reference operator[](difference_type n) const {
      return (ptr_ + n)->element;
    }

  private:
    pointer ptr_;
  };

  padded_array() : start_(nullptr), end_size_(nullptr) {}

  explicit padded_array(size_t num_elements) {
    start_ = new padded_element[num_elements];
    end_size_ = start_ + num_elements;
  }

  padded_array(size_t num_elements, const T& init_val) : padded_array(num_elements) {
    fill(init_val);
  }

  padded_array(iterator copy_begin, iterator copy_end)
      : padded_array(copy_end - copy_begin) {
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < size(); i++)
      start_[i].element = copy_begin[i];
  }

  // don't want this to be copied, too much data to move
  padded_array(const padded_array& other) = delete;
  padded_array& operator=(const padded_array& other) = delete;

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
    return start_[n].element;
  }

  const T& operator[](size_t n) const {
    return start_[n].element;
  }

  void fill(T init_val) {
#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < size(); i++)
      start_[i].element = init_val;
  }

  size_t size() const {
    return end_size_ - start_;
  }

  const iterator begin() const {
    return iterator(start_);
  }

  const iterator end() const {
    return iterator(end_size_);
  }

  padded_element* data() const {
    return start_;
  }

  void swap(padded_array& other) {
    std::swap(start_, other.start_);
    std::swap(end_size_, other.end_size_);
  }

private:
  padded_element* start_;
  padded_element* end_size_;
};

} // namespace parallel

#endif // PARALLEL_PADDED_ARRAY_H_