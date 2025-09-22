// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef CONTAINERS_CHUNK_H_
#define CONTAINERS_CHUNK_H_

#include <cstddef>
#include <cstdint>

namespace containers {

// chunk capacity N must be power of 2
template <typename DataT, typename PrioT, int N>
class chunk {
public:
  using value_type = DataT;
  using priority_type = PrioT;
  static inline constexpr int capacity = N;
  static inline constexpr int mask = N - 1;

  // chunk() = default;

  chunk(PrioT p)
      : head_(0),
        tail_(0),
        priority(p) {}

  chunk(DataT v, PrioT p)
      : head_(0),
        tail_(0),
        priority(p) {
    data_[0] = v;
    tail_++;
  }
  chunk(DataT v, PrioT p, chunk* n)
      : head_(0),
        tail_(0),
        priority(p),
        next(n) {
    data_[0] = v;
    tail_++;
  }
  chunk(DataT v, PrioT p, std::int64_t b, std::int64_t e)
      : head_(0),
        tail_(0),
        priority(p),
        begin(b),
        end(e) {
    data_[0] = v;
    tail_++;
  }

  chunk(chunk&) = delete;
  chunk(chunk&&) = delete;

  bool empty() const noexcept {
    return head_ == tail_;
  }

  bool full() const noexcept {
    return size() == capacity;
  }

  std::size_t size() const noexcept {
    return tail_ - head_;
  }

  // precondition: !*this.full()
  void push_back(DataT value) {
    data_[tail_ & mask] = value;
    tail_++;
  }

  // precondition: !*this.empty()
  DataT pop_front() {
    DataT front = data_[head_ & mask];
    head_++;
    return front;
  }

  void clear() noexcept {
    tail_ = head_ = 0;
  }

private:
  std::uint32_t head_;
  std::uint32_t tail_;

public:
  PrioT priority;
  chunk* next = nullptr;
  std::int64_t begin = 0;
  std::int64_t end = 0;

private:
  DataT data_[N];
};

}; // namespace containers

#endif
