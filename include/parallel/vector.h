// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef PARALLEL_VECTOR_H_
#define PARALLEL_VECTOR_H_

#include <algorithm>

/*
GAP Benchmark Suite
Class:  parallel vector
Author: Scott Beamer

Vector class with ability to not initialize or do initialization in parallel
 - std::vector (when resizing) will always initialize, and does so serially
 - When parallel::vector is resized, new elements are uninitialized
 - Resizing is not thread-safe
*/

namespace parallel {

template <typename T>
class vector {
public:
  typedef T* iterator;

  vector() : start_(nullptr), end_size_(nullptr), end_capacity_(nullptr) {}

  explicit vector(size_t num_elements) {
    start_ = new T[num_elements];
    end_size_ = start_ + num_elements;
    end_capacity_ = end_size_;
  }

  vector(size_t num_elements, T init_val) : vector(num_elements) {
    fill(init_val);
  }

  vector(iterator copy_begin, iterator copy_end)
      : vector(copy_end - copy_begin) {
#pragma omp parallel for
    for (size_t i = 0; i < capacity(); i++)
      start_[i] = copy_begin[i];
  }

  // don't want this to be copied, too much data to move
  vector(const vector& other) = delete;

  // prefer move because too much data to copy
  vector(vector&& other)
      : start_(other.start_), end_size_(other.end_size_),
        end_capacity_(other.end_capacity_) {
    other.start_ = nullptr;
    other.end_size_ = nullptr;
    other.end_capacity_ = nullptr;
  }

  // want move assignment
  vector& operator=(vector&& other) {
    if (this != &other) {
      release_resources();
      start_ = other.start_;
      end_size_ = other.end_size_;
      end_capacity_ = other.end_capacity_;
      other.start_ = nullptr;
      other.end_size_ = nullptr;
      other.end_capacity_ = nullptr;
    }
    return *this;
  }

  void release_resources() {
    if (start_ != nullptr) {
      delete[] start_;
    }
  }

  ~vector() {
    release_resources();
  }

  // not thread-safe
  void reserve(size_t num_elements) {
    if (num_elements > capacity()) {
      T* new_range = new T[num_elements];
#pragma omp parallel for
      for (size_t i = 0; i < size(); i++)
        new_range[i] = start_[i];
      end_size_ = new_range + size();
      delete[] start_;
      start_ = new_range;
      end_capacity_ = start_ + num_elements;
    }
  }

  // prevents internal storage from being freed when this parallel::vector is desctructed
  // - used by Builder to reuse an EdgeList's space for in-place graph building
  void leak() {
    start_ = nullptr;
  }

  bool empty() {
    return end_size_ == start_;
  }

  void clear() {
    end_size_ = start_;
  }

  void resize(size_t num_elements) {
    reserve(num_elements);
    end_size_ = start_ + num_elements;
  }

  T& operator[](size_t n) {
    return start_[n];
  }

  const T& operator[](size_t n) const {
    return start_[n];
  }

  void push_back(T val) {
    if (size() == capacity()) {
      size_t new_size = capacity() == 0 ? 1 : capacity() * growth_factor;
      reserve(new_size);
    }
    *end_size_ = val;
    end_size_++;
  }

  void fill(T init_val) {
#pragma omp parallel for
    for (T* ptr = start_; ptr < end_size_; ptr++)
      *ptr = init_val;
  }

  size_t capacity() const {
    return end_capacity_ - start_;
  }

  size_t size() const {
    return end_size_ - start_;
  }

  iterator begin() const {
    return start_;
  }

  iterator end() const {
    return end_size_;
  }

  T* data() const {
    return start_;
  }

  void swap(vector& other) {
    std::swap(start_, other.start_);
    std::swap(end_size_, other.end_size_);
    std::swap(end_capacity_, other.end_capacity_);
  }

private:
  T* start_;
  T* end_size_;
  T* end_capacity_;
  static const size_t growth_factor = 2;
};

} // namespace parallel

#endif // PVECTOR_H_