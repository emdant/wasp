// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef CONTAINERS_STATIC_BUFFER_H_
#define CONTAINERS_STATIC_BUFFER_H_

#include <cstddef>
#include <cstdint>

namespace containers {

template <typename DataT>
class static_fifo {
public:
  static_fifo(std::uint32_t capacity)
      : cap_(capacity),
        data_(new DataT[cap_]) {}

  ~static_fifo() {
    delete[] data_;
  }

  static_fifo(static_fifo&) = delete;
  static_fifo(static_fifo&&) = delete;

  bool empty() const noexcept {
    return head_ == tail_;
  }

  bool full() const noexcept {
    return tail_ == cap_;
  }

  std::size_t size() const noexcept {
    return tail_;
  }

  // precondition: !*this.full()
  void push_back(DataT value) {
    data_[tail_++] = value;
  }

  // precondition: !*this.empty()
  DataT pop_front() {
    return data_[head_++];
  }

  void clear() noexcept {
    tail_ = head_ = 0;
  }

private:
  std::uint32_t cap_;
  std::uint32_t head_ = 0;
  std::uint32_t tail_ = 0;
  DataT* data_;
};

}; // namespace containers

#endif