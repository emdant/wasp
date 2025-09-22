// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef PARALLEL_DEQUE_H_
#define PARALLEL_DEQUE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "../alignment.h"

namespace parallel {

// Implementation of the Chase-Lev Work Stealing Deque
template <typename DataT>
class deque {

  class ring_buffer {
    int64_t cap_;
    int64_t mask_;
    std::atomic<DataT>* buf_;

  public:
    explicit ring_buffer(int64_t cap)
        : cap_{cap},
          mask_{cap - 1},
          buf_{new std::atomic<DataT>[static_cast<size_t>(cap)]} {
    }

    ~ring_buffer() {
      delete[] buf_;
    }

    int64_t capacity() const noexcept {
      return cap_;
    }

    void push(int64_t i, DataT o) noexcept {
      buf_[i & mask_].store(o, std::memory_order_relaxed);
    }

    DataT pop(int64_t i) noexcept {
      return buf_[i & mask_].load(std::memory_order_relaxed);
    }

    ring_buffer* resize(int64_t b, int64_t t, int delta) {
      ring_buffer* ptr = new ring_buffer{(1 << delta) * cap_};
      for (int64_t i = t; i != b; ++i) {
        ptr->push(i, pop(i));
      }
      return ptr;
    }
  };

public:
  explicit deque(std::int64_t capacity = 1024)
      : buffer_(new ring_buffer(capacity)) {
    old_.reserve(32);
  }

  ~deque() noexcept {
    for (auto a : old_) {
      delete a;
    }
    delete buffer_.load();
  }

  deque(deque const& other) = delete;
  deque& operator=(deque const& other) = delete;

  int64_t capacity() const noexcept {
    return buffer_.load(std::memory_order_relaxed)->capacity();
  }

  bool empty() const noexcept {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_relaxed);
    return t >= b;
  }

  void push(DataT value) noexcept {
    std::int64_t b = bottom_.load(std::memory_order_relaxed);
    std::int64_t t = top_.load(std::memory_order_acquire);
    auto buf = buffer_.load(std::memory_order_relaxed);

    auto size = b - t;
    if (buf->capacity() < size + 1) {
      buf = resize_array(buf, b, t, 1);
    }

    buf->push(b, value);
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
  }

  DataT pop() noexcept {
    std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    auto buf = buffer_.load(std::memory_order_relaxed);

    bottom_.store(b, std::memory_order_relaxed); // Stealers can no longer steal
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::int64_t t = top_.load(std::memory_order_relaxed);

    DataT item{nullptr};

    if (t <= b) {
      item = buf->pop(b);
      if (t == b) {
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
          item = nullptr;
        }
        bottom_.store(b + 1, std::memory_order_relaxed);
      }
    } else {
      bottom_.store(b + 1, std::memory_order_relaxed);
    }
    return item;
  }

  DataT steal() noexcept {
    std::int64_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::int64_t b = bottom_.load(std::memory_order_acquire);

    DataT item{nullptr};
    if (t < b) {
      auto* buf = buffer_.load(std::memory_order_consume);
      item = buf->pop(t);

      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
        return nullptr;
      }
    }
    return item;
  }

  // (*ContainerT::iterator) must be a pointer to the Deque::DataT
  template <typename ContainerT>
  void push(ContainerT& container, std::size_t size) {
    std::int64_t b = bottom_.load(std::memory_order_relaxed);
    std::int64_t t = top_.load(std::memory_order_acquire);
    auto buf = buffer_.load(std::memory_order_relaxed);

    auto desired = (b - t) + size;
    if (buf->capacity() < desired) {
      int delta = 1;
      while ((buf->capacity() << delta) < desired)
        delta++;
      buf = resize_array(buf, b, t, delta);
    }

    for (auto it = container.begin(); it != container.end(); ++it) {
      buf->push(b, *it);
      b++;
    }

    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b, std::memory_order_relaxed);
  }

private:
  alignas(hardware_destructive_interference_size) std::atomic<std::int64_t> top_{0};
  alignas(hardware_destructive_interference_size) std::atomic<std::int64_t> bottom_{0};
  alignas(hardware_destructive_interference_size) std::atomic<ring_buffer*> buffer_;
  std::vector<ring_buffer*> old_;

  ring_buffer* resize_array(ring_buffer* a, std::int64_t b, std::int64_t t, int delta) {
    ring_buffer* tmp = a->resize(b, t, delta);
    old_.push_back(a);
    std::swap(a, tmp);
    buffer_.store(a, std::memory_order_release);
    return a;
  }
};

}; // namespace parallel

#endif