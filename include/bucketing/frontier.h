// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_FRONTIER_H_
#define BUCKETING_FRONTIER_H_

#include <atomic>
#include <limits>
#include <memory>
#include <vector>

#include "base.h"
#include "bucket.h"
#include "current-bucket.h"

namespace bucketing {

template <typename ChunkT = nodes_chunk>
class frontier {
private:
  using value_type = typename ChunkT::value_type;
  using priority_type = typename ChunkT::priority_type;

public:
  frontier() {
    next_buckets_.reserve(1024);
    next_buckets_.resize(128);
  }

  inline void push(value_type value, priority_type priority) {
    if (current_.load(std::memory_order_relaxed) == priority) {
      current_bucket_.push(value, priority);
      return;
    }

    auto current_size = next_buckets_.size();
    if (priority >= current_size) {
      while ((current_size <<= 1) <= priority)
        ;
      next_buckets_.resize(current_size);
    }

    next_buckets_[priority].push_value(value, priority);
  }

  inline std::optional<node_prio> pop() {
    return current_bucket_.pop();
  }

  inline ChunkT* steal() {
    return current_bucket_.steal();
  }

  inline bucket_index current_index() const {
    return current_.load(std::memory_order_acquire);
  }

  inline bucket_index next_index() {
    for (auto i = 0; i < next_buckets_.size(); i++) {
      if (!next_buckets_[i].empty()) {
        return i;
      }
    }
    return EMPTY_BUCKETS;
  }

  bool current_empty() {
    if (local_chunk_ != nullptr)
      return false;

    return current_bucket_.empty();
  }

  void push_from(bucket_index index) {
    auto& bucket = next_buckets_[index];

    while (!bucket.empty())
      current_bucket_.push(bucket.pop_chunk());
  }

  inline void set_current(bucket_index index) {
    current_.store(index, std::memory_order_release);
  }

private:
  std::atomic<bucket_index> current_{0};
  ChunkT* local_chunk_ = nullptr;
  current_bucket<ChunkT> current_bucket_;
  std::vector<bucket<ChunkT>> next_buckets_;
};

}; // namespace bucketing

#endif