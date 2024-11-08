// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_FRONTIER_H_
#define BUCKETING_FRONTIER_H_

#include <atomic>
#include <limits>
#include <memory>
#include <vector>

#include "base.h"
#include "current_bucket.h"
#include "deque_bucket.h"

namespace bucketing {

template <typename ChunkT = nodes_chunk>
class frontier {
private:
  using value_type = typename ChunkT::value_type;
  using priority_type = typename ChunkT::priority_type;

public:
  frontier() {
    buckets_.reserve(1024);
    buckets_.resize(128);

    stealing_deque_.store(buckets_[0].deque());
  }

  inline void push(value_type value, priority_type priority) {
    auto current_size = buckets_.size();
    if (priority >= current_size) {
      while ((current_size <<= 1) <= priority)
        ;

      buckets_.resize(current_size);
    }

    buckets_[priority].push(value, priority);
  }

  inline std::optional<node_prio> pop() {
    auto curr = current_.load(std::memory_order_relaxed);
    if (curr == EMPTY_BUCKETS)
      return std::nullopt;
    return buckets_[curr].pop();
  }

  inline ChunkT* steal() {
    auto sq = stealing_deque_.load();
    if (sq == nullptr)
      return nullptr;

    return sq->steal();
  }

  inline bucket_index current_index() const {
    return current_.load();
  }

  bool current_empty() {
    return buckets_[current_.load(std::memory_order_relaxed)].empty();
  }

  inline void set_current(bucket_index index) {
    if (index == EMPTY_BUCKETS)
      stealing_deque_.store(nullptr);
    else
      stealing_deque_.store(buckets_[index].deque());

    current_.store(index, std::memory_order_release);
  }

  bucket_index next_index() {
    for (auto i = 0; i < buckets_.size(); i++) {
      if (!buckets_[i].empty()) {
        return i;
      }
    }
    return EMPTY_BUCKETS;
  }

private:
  std::atomic<bucket_index> current_{0};
  std::vector<bucket<ChunkT>> buckets_;
  std::atomic<parallel::deque<ChunkT*>*> stealing_deque_;
};

}; // namespace bucketing

#endif